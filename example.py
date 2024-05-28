import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertForMaskedLM
import mlflow
import mlflow.pytorch
import os

# Set CUDA_LAUNCH_BLOCKING to 1 for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up DagsHub as the remote tracking server
MLFLOW_TRACKING_URI = "https://dagshub.com/karmakaragradwip02/sequence_classification-MLFlow-.mlflow"
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = 'karmakaragradwip02'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '9ccb0f28354fcca6469017b32544fa0704b9c343'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze()
        return input_ids

# Load dataset
from datasets import load_dataset
dataset = load_dataset('ag_news', split='train[:1%]') 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128
texts = dataset['text']
text_dataset = TextDataset(texts, tokenizer, max_length)
# Reduce batch size to 4 for debugging
text_loader = DataLoader(text_dataset, batch_size=4, shuffle=True)

class SimpleLM(nn.Module):
    def __init__(self):
        super(SimpleLM, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        return loss

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)

# Start the MLflow run
with mlflow.start_run() as run:
    try:
        # Log hyperparameters
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("epsilon", 1e-8)
        mlflow.log_param("weight_decay", 0.01)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("max_length", max_length)

        # Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in text_loader:
                input_ids = batch.to(device)
                print(f"Processing batch with shape: {input_ids.shape}")
                optimizer.zero_grad()
                loss = model(input_ids)
                print(f"Loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Log the average loss for this epoch
            avg_loss = running_loss / len(text_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Log the model
        mlflow.pytorch.log_model(model, "model")
    except Exception as e:
        print(f"Exception during training: {e}")
    finally:
        mlflow.end_run()