# %%
import os
import torch
import numpy as np
import torchaudio
from datasets import load_dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# %%
AUDIO_DIR = "audio_clips"
METADATA_CSV = "clip_list - clip_list.csv"  # contains filename,label
OUTPUT_DIR = "wav2vec2_aggression_model"
MODEL_CHECKPOINT = "facebook/wav2vec2-base"
TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 5               # max length for padding/truncation
NUM_LABELS = 2                     # violent vs non-violent
LABEL2ID = {"Non Violent": 0, "Violent": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# %%
# 1. Load dataset from CSV
print("Loading metadata...")
ds = load_dataset(
    "csv",
    data_files=METADATA_CSV,
    split="train"
)

# %%
# 2. Rename clip_name to filename
print("Renaming columns...")
ds = ds.rename_column("clip_name", "filename")

# %%
print("Mapping labels...")
def map_labels(example):
    example["labels"] = LABEL2ID[example["class"]]
    return example

ds = ds.map(map_labels, remove_columns=["class"])

# %%
ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]

# %%
print("Loading model and feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT, return_attention_mask=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)

# %%
def preprocess(batch):
    # Load waveform
    path = os.path.join(AUDIO_DIR, batch["filename"])
    waveform, sr = torchaudio.load(path)  # waveform shape: (channels, samples)
    # If necessary, resample to target sample rate
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = TARGET_SAMPLE_RATE
    audio = waveform[0].numpy()

    # Feature extraction with padding/truncation
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        padding="max_length",
        truncation=True,
        max_length=sr * MAX_DURATION_SEC,
        return_tensors="pt"
    )
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    return batch

print("Preprocessing and training datasets...")
train_ds = train_ds.map(preprocess, remove_columns=["filename"])
eval_ds  = eval_ds.map(preprocess, remove_columns=["filename"])

# %%
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# %%
# 8. TrainingArguments & Trainer setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=50,
    eval_steps=100,
    save_steps=500,
    num_train_epochs=5,
    learning_rate=1e-5,
    do_train=True,
    fp16=torch.cuda.is_available(),
    do_eval=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

# %%
print("Starting training...")
trainer.train()
os.makedirs(OUTPUT_DIR, exist_ok=True)
state_path = os.path.join(OUTPUT_DIR, "aggression_detector_state.pt")
torch.save(trainer.model.state_dict(), state_path)
full_model_path = os.path.join(OUTPUT_DIR, "aggression_detector_full.pt")
torch.save(trainer.model, full_model_path)
print(f"Saved state_dict to {state_path}")
print(f"Saved full model to {full_model_path}")

# %%
metrics = trainer.evaluate()
print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"Test F1 Score: {metrics['eval_f1']:.4f}")

# %%



