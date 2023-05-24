import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import HateSpeechDataset


def predict_message(message):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('model.pt')
    msg_encoded = HateSpeechDataset(text=message)
    input_ids, attention_masks, token_type_ids = msg_encoded.embed_msg()
    
    output_chunks = []
    model.to(device)
    model.eval()
    for ids, mask, tt_ids in zip(input_ids, attention_masks, token_type_ids):
        ids = ids.to(device)
        mask = mask.to(device)
        tt_ids = tt_ids.to(device)
        
        with torch.no_grad():
            output = model(ids, 
                           attention_mask=mask,
                           token_type_ids=tt_ids)
        
        logits = output.logits.detach().cpu().numpy()
        prediction = np.argmax(logits, axis=1).flatten()
        output_chunks.append(bool(prediction))
    
    return output_chunks 


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(model, optimizer, scheduler, loader, device):
    total_loss = 0
    total_acc = 0
    model.to(device)
    model.train()
    
    progress_bar = tqdm(loader, total=len(loader))
    for batch in progress_bar:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device) 
        
        model.zero_grad()
        
        output = model(ids,
                       attention_mask=mask,
                       token_type_ids=token_type_ids,
                       labels=labels)
        
        loss = output.loss
        logits = output.logits
        
        total_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        total_acc += flat_accuracy(logits, labels)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    
    return avg_loss, avg_acc


def evaluate(model, loader, device):
    total_loss = 0
    total_acc = 0
    model.to(device)
    model.eval()
    
    progress_bar = tqdm(loader, total=len(loader))
    for batch in progress_bar:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device) 
        
        with torch.no_grad():
            output = model(ids,
                           attention_mask=mask,
                           token_type_ids=token_type_ids,
                           labels=labels)
        loss = output.loss  
        logits = output.logits
        
        total_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        
        total_acc += flat_accuracy(logits, labels)
        
    avg_acc = total_acc / len(loader)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, avg_acc
    

# https://github.com/Vicomtech/hate-speech-dataset
def get_data(data_dir: str):
    df = pd.read_csv('data/annotations_metadata.csv')
    classes = {'hate': 0, 'noHate': 1}
    annots = dict()
    for _, row in df.iterrows():
        if str(row[-1]) in classes:
            annots[str(row[0])] = classes[str(row[-1])]
        
    text = []
    labels = []
    for file in os.listdir(data_dir):
        filename = os.path.basename(file)
        with open(os.path.join(data_dir, file), 'r') as f:
            text.append(f.read())
        labels.append(annots[os.path.splitext(filename)[0]])
        
    return text, labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 128
    batch_size= 8
    learning_rate = 5e-5
    num_epochs = 3
    
    train_text, train_labels = get_data('data/sampled_train')
    val_text, val_labels = get_data('data/sampled_train')
    train_dataset = HateSpeechDataset(text=train_text, max_len=max_len, classes=train_labels)
    eval_dataset = HateSpeechDataset(text=val_text, max_len=max_len, classes=val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate, 
                                  eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_epochs * len(train_dataloader))
    
    for epoch in range(num_epochs):
        print(f'\n===Epoch {epoch + 1}===')
        
        loss_train, acc_train = train(model, optimizer, scheduler, train_dataloader, device)
        print(f'Acc: {acc_train} | Loss: {loss_train}')
        
        loss_eval, acc_eval = evaluate(model, eval_dataloader, device)
        print(f'Acc: {acc_eval} | Loss: {loss_eval}')
        
    torch.save(model, 'model.pt')
    
if __name__ == '__main__':
    main()
