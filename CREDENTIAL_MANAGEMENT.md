# Credential Management

This document explains how to properly manage API credentials and secrets for this project.

## ⚠️ Important Security Notes

- **NEVER commit API keys, tokens, or secrets to git**
- **NEVER push files containing hardcoded credentials**
- Always use environment variables for sensitive data
- Rotate credentials immediately if they are accidentally exposed

## Setup Instructions

### 1. Copy the environment template
```bash
cp .env.template .env
```

### 2. Fill in your credentials
Edit the `.env` file with your actual API credentials:

```bash
# Google OAuth Credentials (for Gemini API)
GOOGLE_CLIENT_ID=your_actual_client_id
GOOGLE_CLIENT_SECRET=your_actual_client_secret

# Hugging Face Token
HUGGINGFACE_TOKEN=hf_your_actual_token
```

### 3. Install python-dotenv (if not already installed)
```bash
pip install python-dotenv
```

### 4. Use credentials in your code
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
```

## For Jupyter Notebooks

Add this cell at the beginning of your notebook:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access credentials
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
```

## Getting Credentials

### Google OAuth (for Gemini API)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Gemini API
4. Go to "Credentials" → "Create Credentials" → "OAuth client ID"
5. Copy the client ID and client secret

### Hugging Face Token
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with appropriate permissions
3. Copy the token (starts with `hf_`)

## File Status

The following files were removed from git history due to containing secrets:
- `Audio-Textual Based Violence Detection/gemini-api.json`
- `Audio-Textual Based Violence Detection/speaker-test.py`
- `Audio-Textual Based Violence Detection/speaker_diarization.ipynb`
- `Audio-Textual Based Violence Detection/youtube-video.ipynb`
- `Audio-Textual Based Violence Detection/audio-textual-analysis.ipynb`

If you need these notebooks for development, recreate them using the environment variable approach described above.
