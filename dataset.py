import re

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class HateSpeechDataset(Dataset):
    def __init__(self, text, max_len=None, classes=None) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        
        self.text = text
        self.classes = classes
            
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.clean_text(self.text[index])
        if len(self.classes) > 0:
            label = self.classes[index]
        else:
            label = None
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        if label is not None:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'labels': label
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }
        
    def embed_msg(self):
        text = self.clean_text(self.text)
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            truncation=False,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for i in range(0, len(encoding['input_ids']), 512):
            input_ids.append(encoding['input_ids'][i:i+512])
            attention_masks.append(encoding['attention_mask'][i:i+512])
            token_type_ids.append(encoding['token_type_ids'][i:i+512])
            
        return input_ids, attention_masks, token_type_ids
        
        
    def clean_text(self, text):
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove emoji
        text = re.sub(r'\:.*?\:', r'', text)
        
        return text
        














