from torch.utils.data import Dataset, DataLoader
import torch
from huggingface_api import tokenizer
from pandas import read_csv

class TextValueDataset(Dataset):
    def __init__(self, texts, values, tokenizer, max_length=512):
        self.texts = texts
        self.values = values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Retrieve the single text entry at the specific index
        text = self.texts.iloc[idx]  # Ensure using .iloc for accurate indexing in pandas
        value = self.values.iloc[idx]

        # Tokenize the text
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        
        return input_ids, torch.tensor([value], dtype=torch.float32)

# Set the EOS token as the padding token if it isn't already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainset = read_csv('training_data.csv')
# Assuming you have a tokenizer for your base model
dataset = TextValueDataset(trainset['text'], trainset['value'], tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

