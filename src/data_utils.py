import re
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
class BasicTokenizer:
    def __init__(self, lower=True, max_length=128):
        self.lower = lower
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.inv_vocab = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_size = 4
        self.max_length=max_length

    def tokenize(self, text):
        if self.lower:
            text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocab(self, texts, min_freq=2):
        freq = defaultdict(int)
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                freq[token] += 1
        for token, count in freq.items():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.inv_vocab[self.vocab_size] = token
                self.vocab_size += 1

    def encode(self, text):
        tokens = ['[CLS]'] + self.tokenize(text) + ['[SEP]']
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab['[PAD]']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        attention_mask = [1 if id != self.vocab['[PAD]'] else 0 for id in token_ids]
        return token_ids, attention_mask

class QbiasDataset(Dataset):
    def __init__(self, texts, labels, domains, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids, attn_mask = self.tokenizer.encode(self.texts[idx])
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'domains': torch.tensor(self.domains[idx], dtype=torch.long)
        }


def load_data(config, tokenizer):
    df = pd.read_csv(config.dataset_path)
    df = df.dropna(subset=[config.label_column, config.domain_column])
    df = df[df[config.label_column].isin(config.bias_classes)]

    label_map = {label: i for i, label in enumerate(config.bias_classes)}
    df['label'] = df[config.label_column].map(label_map)
    domain_map = {config.source_domain: 0, config.target_domain: 1}
    df['domain_id'] = df[config.domain_column].map(domain_map)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)

    # Split source and target separately to maintain class balance
    source_df = df[df[config.domain_column] == config.source_domain]
    target_df = df[df[config.domain_column] == config.target_domain]

    print("\nClass distribution:")
    print("Source domain:")
    print(source_df[config.label_column].value_counts())
    print("\nTarget domain:")
    print(target_df[config.label_column].value_counts())

    train_source, test_source = train_test_split(
        source_df, test_size=config.test_size, stratify=source_df['label'], random_state=config.seed
    )
    train_target, test_target = train_test_split(
        target_df, test_size=config.test_size, stratify=target_df['label'], random_state=config.seed
    )

    train_df = pd.concat([train_source, train_target])
    test_df = pd.concat([test_source, test_target])

    def create_loader(data, shuffle=False):
        return DataLoader(
            QbiasDataset(
                texts=data[config.text_column].tolist(),
                labels=data['label'].tolist(),
                domains=data['domain_id'].tolist(),
                tokenizer=tokenizer,
                max_length=config.max_length
            ),
            batch_size=config.batch_size,
            shuffle=shuffle
        )

    train_loader = create_loader(train_df, shuffle=True)
    val_loader = create_loader(test_df, shuffle=False)
    return train_loader, val_loader, class_weights
