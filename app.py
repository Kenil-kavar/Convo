import requests
from io import BytesIO
import wave
import os
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

if __name__ == "__main__":
    main()
