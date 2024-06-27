from torch.utils.data import Dataset, DataLoader
import torch
from huggingface_api import tokenizer, model
from pandas import read_csv

class TextValueDataset(Dataset):
    def __init__(self, texts, values, tokenizer):
        self.texts = texts
        self.values = values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Retrieve the single text entry at the specific index
        text = self.texts.iloc[idx]  # Ensure using .iloc for accurate indexing in pandas
        value = self.values.iloc[idx]

        # Tokenize the text
        encoding = self.tokenizer(text, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        
        return input_ids, torch.tensor([value], dtype=torch.float32)

# Set the EOS token as the padding token if it isn't already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainset = read_csv('training_final.csv')
# Assuming you have a tokenizer for your base model
dataset = TextValueDataset(trainset['text'], trainset['value'], tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def train_value_model(dataloader, model_load_path="value_model.pth", lr=0.0001):
    from huggingface_api import ValueModel
    vmodel = ValueModel.load_model(model, model_load_path)
    # freeze base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    vmodel.train()
    # Assuming 'dataloader' and 'value_model' are defined and 'value_model' is already loaded
    optimizer = torch.optim.Adam(vmodel.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    num_gpus = torch.cuda.device_count()
    last_gpu = f'cuda:{num_gpus-1}'

    for epoch in range(10):
        short_running_loss = 0.0
        running_loss=0.0
        i = 0
        for inputs, labels in dataloader:
                i+=1
                if i%1000==0:
                    print(f'Iteration: {i}')
                    avg_sloss = short_running_loss/1000
                    print(f'Avg Loss (last 1000): {avg_sloss}')
                    short_running_loss = 0.0
                inputs = inputs.to('cuda:0')
                labels.to(dtype=torch.bfloat16)
                labels = labels.to(last_gpu)
                optimizer.zero_grad()
                # fwd pass
                outputs = vmodel(inputs)
                outputs = outputs.to(last_gpu)
                outputs = outputs.float()                
                # Check data types right before loss calculation
                # print(f'Inputs dtype: {inputs.dtype}, Outputs dtype: {outputs.dtype}, Labels dtype: {labels.dtype}')
                loss = criterion(outputs, labels)
                # print(f'Loss: {loss.item()}')
                # print(f'Loss dtype: {loss.dtype}')  # Check the dtype of loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                short_running_loss += loss.item()
        avg_loss = running_loss/len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # Save the model after each epoch
        try:
            vmodel.save_model(f"checkpoints/model_epoch_{epoch+1}_{avg_loss}.pth")
            print("\033[92mModel saved at checkpoints/model_epoch_{epoch+1}.pth\033[0m\n")
        except:
            torch.save(vmodel.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")