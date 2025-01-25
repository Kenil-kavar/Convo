# Convolabs AI

#Kokoro-82M: Text-to-Speech Project

Welcome to the **Kokoro-82M** project! This repository contains a powerful text-to-speech (TTS) model that converts text into high-quality speech. Follow the instructions below to set up and run the project.

---

## 🚀 Getting Started

To get started with the Kokoro-82M project, follow the steps below.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Git**: To clone the repository.
- **Git LFS**: To handle large files.
- **Python 3.8+**: To run the project.
- **espeak-ng**: For phonemization.

---

## 🛠️ Installation

### 1️⃣ Install Dependencies

Run the following commands to install the required dependencies:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the Kokoro-82M repository
git clone https://huggingface.co/hexgrad/Kokoro-82M

# Navigate to the project directory
cd Kokoro-82M

# Install espeak-ng (for phonemization)
apt-get -qq -y install espeak-ng > /dev/null 2>&1

# Clone the repo
git clone https://github.com/Kenil-kavar/Convolabs-Bot.git

# Install Python dependencies
pip install -q phonemizer torch transformers scipy munch

# Install additional dependencies from requirements.txt
pip install -r requirements.txt

# Finally run the command
python app.py
