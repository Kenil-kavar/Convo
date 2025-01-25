import requests
from io import BytesIO
import wave
import os
from models import build_model
import torch
from kokoro import generate
import numpy as np
from IPython.display import Audio, display

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Replace with your GROQ API key
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

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
    else:
        print("Error:", response.status_code, response.text)

def main():
    """Main function to run the transcription."""
    # Path to the audio file
    audio_file_path = "output.wav"  # Replace with the actual path to your audio file

    # Transcribe the audio file
    transcribe_audio_file(audio_file_path)















device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)
VOICE_NAME = [
    'af', # Default voice is a 50-50 mix of Bella & Sarah
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
][0]
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)


# Function to split text into chunks of a specified length
def split_text_into_chunks(text, chunk_size=500):
    """Split the text into chunks of a specified length."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


text=      """Pass your LLM response"""
# Split the text into chunks
chunk_size = 500  # Set the chunk size
text_chunks = split_text_into_chunks(text, chunk_size)

# Generate audio for each chunk and concatenate the results
audio_chunks = []
for chunk in text_chunks:
    audio, out_ps = generate(MODEL, chunk, VOICEPACK, lang=VOICE_NAME[0])
    audio_chunks.append(audio)
    
# Combine Audio
combined_audio = np.concatenate(audio_chunks)

# Playing the combined audio using IPython's Audio widget
def play_audio(audio_data, sample_rate=24000):
    """Playing audio using IPython's Audio widget."""
    display(Audio(data=audio_data, rate=sample_rate, autoplay=True))

# Play the combined audio
play_audio(combined_audio)


# Note: Phonemes will be specific to each chunk
for chunk in text_chunks:
    _, out_ps = generate(MODEL, chunk, VOICEPACK, lang=VOICE_NAME[0])
    print(out_ps)


if __name__ == "__main__":
    main()
