import re
from collections import defaultdict

class BasicTokenizer:
    def __init__(self, lower=True):
        self.lower = lower
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.inv_vocab = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_size = 4

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

    def encode(self, text, max_len=512):
        tokens = ['[CLS]'] + self.tokenize(text) + ['[SEP]']
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        if len(token_ids) < max_len:
            token_ids += [self.vocab['[PAD]']] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        attention_mask = [1 if id != self.vocab['[PAD]'] else 0 for id in token_ids]
        return token_ids, attention_mask
