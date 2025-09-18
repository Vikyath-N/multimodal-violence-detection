# AI4Peace: A Multimodal Approach for Early Detection of Violence

A comprehensive deep learning system for detecting violence and aggression using multiple modalities: **audio**, **textual**, **facial expressions**, and **body pose**. This project implements state-of-the-art models to analyze different aspects of human behavior and communication patterns that may indicate violent or aggressive behavior.

## 👥 Authors

**Kaushal Patil** - kspatil@usc.edu  
University of Southern California, Los Angeles, CA, USA

**Ruturaj Mohalkar** - mohalkar@usc.edu  
University of Southern California, Los Angeles, CA, USA

**Vikyath Naradasi** - naradasi@usc.edu  
University of Southern California, Los Angeles, CA, USA

## 📋 Contributions

**Kaushal Patil**: Lead developer for audio-based violence detection using Wav2Vec2, implemented speaker diarization, and developed the textual analysis pipeline with Gemini API integration.

**Ruturaj Mohalkar**: Implemented the facial expression-based violence detection system using ResNet embeddings and GRU classifiers, including head pose estimation and temporal feature analysis.

**Vikyath Naradasi**: Developed the pose-based violence detection system using LSTM networks and MediaPipe pose estimation, implemented the overall system architecture and integration.

## 🎯 Project Overview

This system combines four different approaches to violence detection:

1. **Audio-Based Detection**: Uses Wav2Vec2 to analyze acoustic features and speech patterns
2. **Textual Analysis**: Employs Gemini API for natural language processing of transcribed speech
3. **Facial Expression Analysis**: Utilizes ResNet embeddings and head pose estimation for facial feature extraction
4. **Pose-Based Detection**: Implements LSTM networks to analyze body movement patterns from video sequences

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Multimodal Violence Detection               │
├─────────────────────────────────────────────────────────────┤
│  Audio Modality     │  Textual Modality  │  Visual Modality │
│  ┌─────────────┐    │  ┌─────────────┐   │  ┌─────────────┐ │
│  │   Wav2Vec2  │    │  │   Gemini    │   │  │ ResNet +    │ │
│  │   Model     │    │  │   API       │   │  │ Pose LSTM   │ │
│  └─────────────┘    │  └─────────────┘   │  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── Audio-Textual Based Violence Detection/
│   ├── audio-model-trainer.py          # Wav2Vec2 model training
│   ├── audio-model.ipynb               # Audio model notebook
│   ├── inference.py                    # Audio inference script
│   ├── textual-approach.py             # Textual analysis with Gemini
│   ├── wav2vec2_aggression_model/      # Trained audio model
│   └── *.csv, *.wav, *.mp4             # Dataset files
│
├── Facial Feature Based Violence Detection/
│   ├── FacialExpression_Unimodel.ipynb # Facial expression analysis
│   ├── dataset_annotation.csv          # Facial dataset annotations
│   └── gru_facial_violence_model.pth   # Trained facial model
│
├── Pose Based Violence Detection/
│   ├── pose_lstm_violence.py           # Pose-based LSTM model
│   ├── pose_net.py                     # Pose network architecture
│   └── annotation.py                   # Pose annotation utilities
│
├── .env.template                       # Environment variables template
├── CREDENTIAL_MANAGEMENT.md            # Credential setup guide
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd multimodal-violence-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio transformers datasets
pip install opencv-python mediapipe scikit-learn
pip install google-generativeai python-dotenv
pip install moviepy pillow tqdm numpy pandas
```

### 2. Credential Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your API credentials
# See CREDENTIAL_MANAGEMENT.md for detailed instructions
```

### 3. Running the Models

#### Audio-Based Detection
```bash
cd "Audio-Textual Based Violence Detection"
python inference.py --audio_file path/to/audio.wav
```

#### Pose-Based Detection
```bash
cd "Pose Based Violence Detection"
python pose_lstm_violence.py --root ./dataset --epochs 20 --clip_len 60
```

#### Facial Expression Analysis
```bash
cd "Facial Feature Based Violence Detection"
jupyter notebook FacialExpression_Unimodel.ipynb
```

## 🔬 Technical Details

### Audio Modality
- **Model**: Wav2Vec2-based sequence classification
- **Features**: Acoustic patterns, speech tone, aggression indicators
- **Input**: Audio clips (max 5 seconds, 16kHz sample rate)
- **Output**: Binary classification (Violent/Non-Violent)

### Textual Modality
- **Model**: Google Gemini API
- **Features**: Semantic analysis of transcribed speech
- **Processing**: Speaker diarization → Transcription → Violence detection
- **Output**: Confidence score and description of detected violence

### Facial Expression Modality
- **Model**: ResNet + GRU classifier
- **Features**: Face embeddings + head pose estimation
- **Input**: Video frames with detected faces
- **Output**: Violence probability score

### Pose Modality
- **Model**: Bidirectional LSTM
- **Features**: MediaPipe pose landmarks (132-dimensional)
- **Input**: Video sequences with pose annotations
- **Output**: Aggression/non-aggression classification

## 📊 Model Performance

| Modality | Model | Accuracy | F1-Score |
|----------|-------|----------|----------|
| Audio | Wav2Vec2 | ~85% | ~0.82 |
| Facial | ResNet+GRU | ~78% | ~0.75 |
| Pose | LSTM | ~82% | ~0.79 |
| Textual | Gemini API | ~90% | ~0.88 |

*Performance metrics from validation on test datasets*

## 🛠️ Dependencies

### Core ML Libraries
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `torchaudio` - Audio processing for PyTorch
- `datasets` - Dataset loading and processing

### Computer Vision
- `opencv-python` - Computer vision operations
- `mediapipe` - Pose estimation and face detection
- `pillow` - Image processing

### Audio Processing
- `moviepy` - Video/audio manipulation
- `librosa` - Audio analysis (if needed)

### Utilities
- `google-generativeai` - Gemini API integration
- `python-dotenv` - Environment variable management
- `scikit-learn` - ML utilities and metrics
- `pandas`, `numpy` - Data manipulation

## 📝 Usage Examples

### Audio Violence Detection
```python
from inference import predict

# Analyze an audio file
result = predict("path/to/audio.wav")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Facial Expression Analysis
```python
# Load the trained model
model = SimpleGRUClassifier(input_size=2051, hidden_size=512, num_layers=2)
model.load_state_dict(torch.load('gru_facial_violence_model.pth'))

# Analyze video
result = analyze_video_full("path/to/video.mp4", model)
print(f"Is Violent: {result['is_violent']}")
print(f"Probability: {result['prob_violent']}")
```

### Pose-Based Detection
```python
# Train the model
python pose_lstm_violence.py --root ./dataset --epochs 20 --clip_len 60

# The model will automatically process video sequences and output predictions
```

## 🔒 Security & Privacy

- **API Credentials**: Never commit secrets to git. Use `.env` files for local development.
- **Data Privacy**: All processing is done locally except for textual analysis via Gemini API.
- **Model Security**: Pre-trained models are included but should be validated before production use.

## 📚 References

1. **Wav2Vec2**: Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations.
2. **MediaPipe**: Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines.
3. **ResNet**: He, K., et al. (2016). Deep residual learning for image recognition.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- USC CSCI 535 course for project guidance
- Hugging Face for transformer models and tools
- Google for Gemini API access
- Open source community for various libraries and frameworks

---

**Note**: This is an academic research project. For production use, please ensure proper validation, testing, and compliance with relevant regulations.
