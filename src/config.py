from dataclasses import dataclass, field
import torch
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


@dataclass
class MbicConfig:
    dataset_path: str = 'data/mbic.csv'
    output_dir: str = 'results'
    text_column: str = 'text'
    label_column: str = 'type'
    domain_column: str = 'topic'
    source_domain: str = 'environment'
    target_domain: str = 'gun-control'
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-4
    test_size: int = 0.2
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    bias_classes: list = field(default_factory=lambda: ['left', 'center', 'right'])