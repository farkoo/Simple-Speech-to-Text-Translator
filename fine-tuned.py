

import pandas as pd
from sklearn.model_selection import train_test_split
from hezar.models import Model
from hezar.trainer import Trainer, TrainerConfig
from hezar.preprocessors import Preprocessor
from datasets import Dataset, load_metric
from hezar.data import Dataset
import torch

# # Load the Excel file
# file_path = r"C:\Users\ACER\Downloads\archive (9)\final_df.xlsx"
# df = pd.read_excel(file_path)

# # Split the dataset into training (60%) and testing (40%) sets
# train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)

# # Convert the DataFrame to a Dataset
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

dataset_path = "hezarai/common-voice-13-fa"
base_model_path = "hezarai/whisper-small"

train_dataset = Dataset.load(dataset_path, preprocessor=base_model_path, split="train")
eval_dataset = Dataset.load(dataset_path, preprocessor=base_model_path, split="test")

# Load the pre-trained model
model = Model.load("hezarai/whisper-small")

# Define training arguments
train_config = TrainerConfig(
    output_dir="result",
    task="speech_recognition",
    mixed_precision="bf16",           # Use bf16 for mixed precision training
    resume_from_checkpoint=True,
    gradient_accumulation_steps=64,    # Accumulate gradients over 8 steps
    batch_size=1,                     # Reduce batch size to 1
    log_steps=100,
    save_steps=1000,
    num_epochs=5,
    metrics=["cer", "wer"],
)

# Initialize the Trainer
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Clear CUDA cache
torch.cuda.empty_cache()

# Train the model
trainer.train()

#%%

# Save the model
model.save_pretrained("./fine-tuned-whisper-small-fa")

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
#%%
from hezar.data import Dataset

commonvoice_dataset = Dataset.load("hezarai/common-voice-13-fa", preprocessor="hezarai/whisper-small-fa")  # A SpeechRecognitionDataset instance
