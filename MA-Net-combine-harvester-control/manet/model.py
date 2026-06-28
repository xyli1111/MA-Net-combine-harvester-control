from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class MANetConfig:
    input_dim: int = 6
    output_dim: int = 6
    hidden_dim: int = 64
    memory_size: int = 256
    siam_heads: int = 4
    dropout: float = 0.1
    lstm_layers: int = 1
    update_threshold: float = 0.5


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class LocalFeatureExtractionUnit(nn.Module):
    """LFEU: residual convolutional feature extractor for local fluctuations."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.resnet = nn.Sequential(
            ResidualConvBlock(hidden_dim, dropout),
            ResidualConvBlock(hidden_dim, dropout),
        )
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input: [batch, seq, features], output: [batch, seq, hidden].
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.resnet(x)
        x = self.pool(x).transpose(1, 2)
        return self.output(x)


class HistoricalKnowledgeVectorDatabase(nn.Module):
    """HKVD with cosine retrieval and optional dynamic memory update."""

    def __init__(self, memory_size: int, hidden_dim: int, update_threshold: float) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.update_threshold = update_threshold
        memory = torch.randn(memory_size, hidden_dim) * 0.02
        self.register_buffer("memory", F.normalize(memory, dim=-1))
        self.register_buffer("write_index", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, vectors: Tensor) -> None:
        vectors = F.normalize(vectors.detach(), dim=-1)
        similarity = vectors @ self.memory.t()
        max_similarity = similarity.max(dim=-1).values
        new_vectors = vectors[max_similarity < self.update_threshold]
        if new_vectors.numel() == 0:
            return

        for vector in new_vectors:
            index = int(self.write_index.item() % self.memory_size)
            self.memory[index].copy_(vector)
            self.write_index.add_(1)

    def retrieve(self, query: Tensor, update: bool = False) -> Tensor:
        query = F.normalize(query, dim=-1)
        memory = F.normalize(self.memory, dim=-1)
        similarity = query @ memory.t()
        indices = similarity.argmax(dim=-1)
        recalled = memory[indices]
        if update and self.training:
            self.update(query)
        return recalled


class SimpleAttention(nn.Module):
    """SIAM: lightweight channel attention plus depthwise temporal convolution."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by siam_heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.channel_score = nn.Linear(hidden_dim, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        value = self.value(x).view(bsz, seq_len, self.heads, self.head_dim)
        score = self.channel_score(x.mean(dim=1))
        score = score.view(bsz, self.heads, self.head_dim)
        score = F.softmax(score / (self.head_dim**0.5), dim=-1)
        attended = value * score.unsqueeze(1)
        attended = attended.reshape(bsz, seq_len, self.hidden_dim)

        local = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        return self.output(attended * local)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GlobalDependencyModelingUnit(nn.Module):
    """GDMU: SIAM and FFN block for global temporal dependency modeling."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = SimpleAttention(hidden_dim, heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(x)
        return x


class HybridFeatureFusionUnit(nn.Module):
    """HFFU: fuse current global features with recalled long-term memory."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, global_features: Tensor, memory: Tensor) -> Tensor:
        memory = memory.unsqueeze(1).expand_as(global_features)
        return self.mlp(torch.cat([global_features, memory], dim=-1))


class BidirectionalMemoryRecallUnit(nn.Module):
    """BMRU: BiLSTM decoder for bidirectional temporal reconstruction."""

    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.lstm(x)
        return self.proj(output)


class MANet(nn.Module):
    def __init__(self, config: MANetConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = MANetConfig(**config)
        self.config = config
        self.lfeu = LocalFeatureExtractionUnit(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.hkvd = HistoricalKnowledgeVectorDatabase(
            config.memory_size,
            config.hidden_dim,
            config.update_threshold,
        )
        self.gdmu = GlobalDependencyModelingUnit(
            config.hidden_dim,
            config.siam_heads,
            config.dropout,
        )
        self.hffu = HybridFeatureFusionUnit(config.hidden_dim, config.dropout)
        self.bmru = BidirectionalMemoryRecallUnit(
            config.hidden_dim,
            config.lstm_layers,
            config.dropout,
        )
        self.output = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: Tensor, pred_length: int | None = None, update_memory: bool = False) -> Tensor:
        local_features = self.lfeu(x)
        query = local_features.mean(dim=1)
        long_term_memory = self.hkvd.retrieve(query, update=update_memory)
        global_features = self.gdmu(local_features)
        fused = self.hffu(global_features, long_term_memory)
        decoded = self.bmru(fused)
        output = self.output(decoded)
        if pred_length is not None:
            output = output[:, -pred_length:, :]
        return output
