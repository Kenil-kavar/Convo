from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import torch
import numpy as np
import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from .models import build_model  # Updated import
from .kokoro import generate  # Updated import
from scipy.io.wavfile import write

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = "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"  # Replace with your GROQ API key
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Global variables for the chat model
model = None
tokenizer = None
messages = []

def initialize_chat_model():
    """Initialize the chat model and tokenizer."""
    global model, tokenizer, messages

    # Load the saved model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # Enable faster inference
    FastLanguageModel.for_inference(model)

    # Initialize system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly and professional customer care assistant. "
            "Assist users with clear and accurate guidance for their requests."
        )
    }

    # Initialize messages
    messages = [
        system_prompt,
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]

def transcribe_audio_file(audio_file_path):
    """Transcribe an audio file using the GROQ API."""
    # Read the audio file
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # Send the audio to the GROQ API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    files = {
        "file": ("audio.wav", audio_data, "audio/wav"),
        "model": (None, "whisper-large-v3"),
    }

    response = requests.post(URL, headers=headers, files=files)

    if response.status_code == 200:
        transcription = response.json()
        return transcription["text"]  # Return the transcription text
    else:
        return None  # Return None if there's an error

def chat_with_model(user_input):
    """Chat with the language model and return the AI's response."""
    global messages

    # Append user input to messages
    messages.append({"role": "user", "content": user_input})

    # Tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate the output
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=1.0,
        min_p=0.1
    )

    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = decoded_output.split("assistant")[-1].strip()

    # Append AI response to messages
    messages.append({"role": "assistant", "content": ai_response})

    return ai_response

def generate_and_play_audio(text, model, voicepack, voice_name):
    """Generate audio from text and play it."""
    # Split the text into chunks
    chunk_size = 490  # Set the chunk size
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Generate audio for each chunk and concatenate the results
    audio_chunks = []
    for chunk in text_chunks:
        audio, _ = generate(model, chunk, voicepack, lang=voice_name[0])
        audio_chunks.append(audio)

    # Combine audio chunks
    combined_audio = np.concatenate(audio_chunks)

    # Save the combined audio
    write("output.wav", 24000, combined_audio)

@csrf_exempt
def handle_audio(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        audio_file_path = f"media/{audio_file.name}"
        with open(audio_file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # Initialize the chat model
        initialize_chat_model()

        # Transcribe the audio file
        user_input = transcribe_audio_file(audio_file_path)
        if not user_input:
            return JsonResponse({'error': 'Transcription failed'}, status=500)

        # Get AI response
        ai_response = chat_with_model(user_input)

        # Generate and play audio for the AI response
        if ai_response:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            kokoro_model = build_model('kokoro-v0_19.pth', device)
            voice_name = [
                'af',  # Default voice is a 50-50 mix of Bella & Sarah
                'af_bella', 'af_sarah', 'am_adam', 'am_michael',
                'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
                'af_nicole', 'af_sky',
            ][0]
            voicepack = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)
            generate_and_play_audio(ai_response, kokoro_model, voicepack, voice_name)

            return JsonResponse({'response': ai_response, 'audio_file': 'output.wav'})
        else:
            return JsonResponse({'error': 'No response from the chat model'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)
