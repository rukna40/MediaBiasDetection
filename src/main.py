from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import shap
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------

@dataclass
class QbiasConfig:
    dataset_path: str = 'data/qbias.csv'
    output_dir: str = 'results'
    text_column: str = 'heading'
    label_column: str = 'bias_rating'
    domain_column: str = 'source'
    source_domain: str = 'New York Times (News)'
    target_domain: str = 'Fox News (Online News)'
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-4
    test_size: int = 0.2
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    bias_classes: list = field(default_factory=lambda: ['left', 'center', 'right'])

# -------------------------------
# Tokenizer
# -------------------------------

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

# -------------------------------
# Dataset
# -------------------------------

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

# -------------------------------
# Data Loading
# -------------------------------

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

# -------------------------------
# Metrics
# -------------------------------

def calculate_metrics(y_true, y_pred, target_names=None, labels=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        labels=labels if labels is not None else list(range(len(target_names))),
        zero_division=0
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "classification_report": report
    }

# -------------------------------
# Model
# -------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out   = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask: (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(B, T, C)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class BERTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256, num_layers=2, max_len=512, num_classes=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # attention_mask: (B, T) -> (B, 1, 1, T)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

# -------------------------------
# Training & Evaluation
# -------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

# -------------------------------
# SHAP Explainability
# -------------------------------

class BiasExplainer:
    def __init__(self, model, tokenizer, device, class_names):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names
        self.model.eval()

        # Create full SHAP-compatible tokenizer
        class ShapTokenizer:
            def __init__(self, base_tokenizer):
                self.base = base_tokenizer
                self.mask_token = "[MASK]"

            def __call__(self, text, return_offsets_mapping=False):
                # Tokenize and build offset mapping
                tokens = ['[CLS]'] + self.base.tokenize(text) + ['[SEP]']
                token_ids = [self.base.vocab.get(token, self.base.vocab['[UNK]']) for token in tokens]

                offset_mapping = []
                pos = 0
                text_lower = text.lower() if self.base.lower else text
                for token in tokens:
                    if token in ['[CLS]', '[SEP]']:
                        offset_mapping.append((0, 0))
                    else:
                        # Find token in text
                        start = text_lower.find(token, pos)
                        if start == -1:
                            offset_mapping.append((0, 0))
                        else:
                            end = start + len(token)
                            offset_mapping.append((start, end))
                            pos = end

                out = {"input_ids": token_ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = offset_mapping
                return out

            def decode(self, ids):
                return [self.base.inv_vocab.get(int(id), "[UNK]") for id in ids]

        self.shap_tokenizer = ShapTokenizer(tokenizer)

    def predictor(self, texts):
        input_ids = []
        attention_masks = []
        for text in texts:
            ids, mask = self.tokenizer.encode(text)
            input_ids.append(ids)
            attention_masks.append(mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    def explain(self, text):
        explainer = shap.Explainer(
            self.predictor,
            masker=shap.maskers.Text(self.shap_tokenizer),
            output_names=self.class_names
        )
        return explainer([text])

# -------------------------------
# Main
# -------------------------------

def main():
    config = QbiasConfig()
    df = pd.read_csv(config.dataset_path)
    train_texts = df[config.text_column].dropna().tolist()
    tokenizer = BasicTokenizer()
    tokenizer.build_vocab(train_texts)
    
    train_loader, val_loader, class_weights = load_data(config, tokenizer)

    model = BERTModel(tokenizer.vocab_size).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        y_true_val, y_pred_val = evaluate(model, val_loader, config.device)
        metrics_val = calculate_metrics(
            y_true_val, y_pred_val, 
            target_names=config.bias_classes, 
            labels=list(range(len(config.bias_classes)))
        )
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {metrics_val['accuracy']:.4f}")

    torch.save(model.state_dict(), f"{config.output_dir}/qbias_model.pth")
    print(f"Model saved to {config.output_dir}/qbias_model.pth")

    model.load_state_dict(torch.load(f"{config.output_dir}/qbias_model.pth", weights_only=True))

    # Final evaluation
    y_true_test, y_pred_test = evaluate(model, val_loader, config.device)
    metrics_test = calculate_metrics(
        y_true_test, y_pred_test, 
        target_names=config.bias_classes, 
        labels=list(range(len(config.bias_classes)))
    )
    print(f"\nFinal Test Accuracy: {metrics_test['accuracy']:.4f}")
    print(metrics_test["classification_report"])

    # Print sample predictions from validation set
    sample_batch = next(iter(val_loader))
    input_ids = sample_batch['input_ids'].to(config.device)
    attention_mask = sample_batch['attention_mask'].to(config.device)
    labels = sample_batch['labels'].to(config.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    explainer = BiasExplainer(model, tokenizer, config.device, config.bias_classes)
    N = 5
    for sample_idx in range(N):  
        text_tokens = input_ids[sample_idx].cpu().numpy()
        raw_tokens = [tokenizer.inv_vocab.get(int(idx), '[UNK]') for idx in text_tokens]
        filtered_tokens = [t for t in raw_tokens if t != '[PAD]']
        
        true_label = config.bias_classes[labels[sample_idx].item()]
        pred_label = config.bias_classes[preds[sample_idx].item()]
        
        print(f"\nSample {sample_idx+1}:")
        print("Text:", " ".join(filtered_tokens))
        print("True Label:", true_label)
        print("Predicted Label:", pred_label)
        print("Probabilities:", probs[sample_idx].cpu().numpy())

        # Generate SHAP explanation
        explanation = explainer.explain(" ".join(filtered_tokens))
        shap_values = explanation[0].values
        explanation_tokens = explanation[0].data

        print("\nToken-level SHAP contributions:")
        for token_idx, token in enumerate(explanation_tokens):
            contributions = " | ".join(
                [f"{cls}:{shap_values[token_idx, cls_idx]:.2f}" 
                for cls_idx, cls in enumerate(config.bias_classes)]
            )
            print(f"{token}: {contributions}")


    user_text = input("\nEnter your own text to analyze (or press Enter to exit): ").strip()
    while user_text:
        token_ids, attn_mask = tokenizer.encode(user_text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(config.device)
        mask_tensor = torch.tensor([attn_mask], dtype=torch.long).to(config.device)
        
        with torch.no_grad():
            output = model(input_tensor, mask_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
        
        # Decode tokens for display
        raw_tokens = [tokenizer.inv_vocab.get(int(idx), '[UNK]') for idx in token_ids]
        filtered_tokens = [t for t in raw_tokens if t != '[PAD]']
        
        print("\nUser Input Analysis:")
        print("Text:", " ".join(filtered_tokens))
        print("Predicted Label:", config.bias_classes[pred])
        print("Probabilities:", prob[0].cpu().numpy())
        
        # Generate SHAP explanation
        explanation = explainer.explain(user_text)
        shap_values = explanation[0].values
        explanation_tokens = explanation[0].data
        
        print("\nToken-level SHAP contributions:")
        for token_idx, token in enumerate(explanation_tokens):
            contributions = " | ".join(
                [f"{cls}:{shap_values[token_idx, cls_idx]:.2f}" 
                for cls_idx, cls in enumerate(config.bias_classes)]
            )
            print(f"{token}: {contributions}")
        
        user_text = input("\nEnter another text to analyze (or press Enter to exit): ").strip()


if __name__ == "__main__":
    main()
