import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
        token_ids, attn_mask = self.tokenizer.encode(self.texts[idx], max_len=self.max_length)
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.float),
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

    source_df = df[df[config.domain_column] == config.source_domain]
    target_df = df[df[config.domain_column] == config.target_domain]

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

    source_loader = create_loader(train_df[train_df[config.domain_column] == config.source_domain], shuffle=True)
    target_loader = create_loader(train_df[train_df[config.domain_column] == config.target_domain])
    val_loader = create_loader(test_df[test_df[config.domain_column] == config.source_domain])
    test_loader = create_loader(test_df[test_df[config.domain_column] == config.target_domain])

    return source_loader, target_loader, val_loader, test_loader
