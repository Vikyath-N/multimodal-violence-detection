import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Config
MODEL_DIR       = "wav2vec2_aggression_model"
AUDIO_DIR       = "test_dataset"         # or wherever your test clips live
TARGET_SR       = 16_000
MAX_DUR_SEC     = 5
ID2LABEL        = {0: "Non Violent", 1: "Violent"}

# 1) Load model & feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 2) Preprocessing function
def preprocess_file(path):
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
        sr = TARGET_SR
    audio = waveform[0].numpy()
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        padding="max_length",
        truncation=True,
        max_length=sr * MAX_DUR_SEC,
        return_tensors="pt"
    )
    return inputs.input_values, inputs.attention_mask

# 3) Inference on a single file
def predict(path):
    input_values, attention_mask = preprocess_file(path)
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (1, 2)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], float(probs[pred_id])

# 4) Test on a batch of files
if __name__ == "__main__":
    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(AUDIO_DIR, fname)
        label, confidence = predict(path)
        print(f"{fname} → {label} (confidence {confidence:.2%})")
