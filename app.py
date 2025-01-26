import requests
import torch
import numpy as np
from IPython.display import Audio, display
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from models import build_model
from kokoro import generate

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = "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"  # Replace with your GROQ API key
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Global variables for the chat model
model = None
tokenizer = None
messages = []


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
        print("Transcription:", transcription["text"])
        return transcription["text"]  # Return the transcription text
    else:
        print("Error:", response.status_code, response.text)
        return None  # Return None if there's an error


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

    print(f"AI: {ai_response}")
    return ai_response


def split_text_into_chunks(text, chunk_size=500):
    """Split the text into chunks of a specified length."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def play_audio(audio_data, sample_rate=24000):
    """Play audio using IPython's Audio widget."""
    display(Audio(data=audio_data, rate=sample_rate, autoplay=True))


def generate_and_play_audio(text, model, voicepack, voice_name):
    """Generate audio from text and play it."""
    # Split the text into chunks
    
    chunk_size = 490  # Set the chunk size
    text_chunks = split_text_into_chunks(text, chunk_size)

    # Generate audio for each chunk and concatenate the results
    audio_chunks = []
    if not text_chunks:
      print("ERROR---------------")
    for chunk in text_chunks:
        audio, _ = generate(model, chunk, voicepack, lang=voice_name[0])
        audio_chunks.append(audio)

    # Combine audio chunks
    combined_audio = np.concatenate(audio_chunks)

    # Play the combined audio
    play_audio(combined_audio)
    for chunk in text_chunks:
      _, out_ps = generate(model, chunk, voicepack, lang=voice_name[0])
      print(out_ps)

 
def main():
    """Main function to run the transcription and chat."""
    # Initialize the chat model
    initialize_chat_model()

    # Path to the audio file
    audio_file_path = "sound.wav"  # Replace with the actual path to your audio file

    # Load the Kokoro model and voicepack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kokoro_model = build_model('kokoro-v0_19.pth', device)
    voice_name = [
        'af',  # Default voice is a 50-50 mix of Bella & Sarah
        'af_bella', 'af_sarah', 'am_adam', 'am_michael',
        'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
        'af_nicole', 'af_sky',
    ][0]
    voicepack = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)

    # Main loop
    print("AI: Hi, how can I help you?")
    # Transcribe the audio file
    user_input = transcribe_audio_file(audio_file_path)
    if not user_input:
        print("Error: Transcription failed.")
        

    # Check for exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat. Goodbye!")

    # Get AI response
    ai_response = chat_with_model(user_input)
    # Generate and play audio for the AI response
    if ai_response:
        generate_and_play_audio(ai_response, kokoro_model, voicepack, voice_name)
    else:
        print("Error: No response from the chat model.")


if __name__ == "__main__":
    main()
