from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sparse_adj(n: int, src: torch.Tensor, dst: torch.Tensor, edge_keep_prob: float = 1.0) -> torch.Tensor:
    """Build row-normalized sparse adjacency with self-loops and optional edge dropout."""
    assert src.dtype == torch.long and dst.dtype == torch.long
    if edge_keep_prob < 1.0 and src.numel() > 0:
        keep = torch.rand(src.shape[0], device=src.device) < edge_keep_prob
        src = src[keep]
        dst = dst[keep]
    self_idx = torch.arange(n, device=src.device, dtype=torch.long)
    src2 = torch.cat([src, self_idx], dim=0)
    dst2 = torch.cat([dst, self_idx], dim=0)
    val = torch.ones(src2.shape[0], device=src.device)
    A = torch.sparse_coo_tensor(torch.stack([src2, dst2]), val, (n, n)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    rows = A.indices()[0]
    vals = A.values() / deg[rows]
    return torch.sparse_coo_tensor(A.indices(), vals, (n, n), device=src.device).coalesce()


class MixHopEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, K: int = 2, dropout: float = 0.2):
        super().__init__()
        self.K = int(K)
        self.dropout = float(dropout)
        self.proj = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(self.K + 1)])
        self.out = nn.Linear((self.K + 1) * hid_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        hs = []
        h = X
        hs.append(F.relu(self.proj[0](F.dropout(h, p=self.dropout, training=self.training))))
        for k in range(1, self.K + 1):
            h = torch.sparse.mm(A, h)
            hs.append(F.relu(self.proj[k](F.dropout(h, p=self.dropout, training=self.training))))
        return self.out(torch.cat(hs, dim=1))


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(A, X)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.lin(h))


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = float(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(A, X)
        h = torch.cat([X, neigh], dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.lin(h))


class GATEncoderLite(nn.Module):
    """Dependency-light one-head GAT-style encoder used for reproducible baselines."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

    @staticmethod
    def _scatter_max(dst: torch.Tensor, e: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.full((n,), float("-inf"), device=e.device, dtype=e.dtype)
        if hasattr(out, "scatter_reduce_"):
            out.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        else:
            for d in torch.unique(dst):
                mask = dst == d
                out[d] = torch.max(e[mask])
        return out

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A = A.coalesce()
        idx = A.indices()
        src, dst = idx[0], idx[1]
        n = X.size(0)
        H = F.dropout(self.W(X), p=self.dropout, training=self.training)
        e = F.leaky_relu(self.a_src(H[src]).squeeze(-1) + self.a_dst(H[dst]).squeeze(-1), negative_slope=self.negative_slope)
        e_max = self._scatter_max(dst, e, n)
        exp_e = torch.exp(e - e_max[dst]).clamp(min=0.0)
        denom = torch.zeros((n,), device=exp_e.device, dtype=exp_e.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-12)
        out = torch.zeros((n, H.size(1)), device=H.device, dtype=H.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(1) * H[src])
        return F.elu(out)


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 128, dropout: float = 0.1):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k)
        self.Wk = nn.Linear(d_model, d_k)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.scale = 1.0 / math.sqrt(float(d_k))
        self.last_attention: torch.Tensor | None = None

    def forward(self, z_t: torch.Tensor, z_hist: torch.Tensor) -> torch.Tensor:
        q = self.Wq(z_t).unsqueeze(1)
        k = self.Wk(z_hist)
        v = self.Wv(z_hist)
        attn = torch.softmax((q * k).sum(-1) * self.scale, dim=1)
        self.last_attention = attn.detach()
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return (attn.unsqueeze(-1) * v).sum(1)


class GraphClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, d_k: int = 128, dropout: float = 0.2, temporal: str = "attention"):
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.temp_attn = TemporalAttention(emb_dim, d_k=d_k, dropout=dropout)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.classifier = nn.Linear(emb_dim, 1)

    def encode(self, X, A):
        return self.encoder(X, A)

    def forward(self, z_t: torch.Tensor, z_hist: torch.Tensor):
        if self.temporal == "attention":
            z = self.temp_attn(z_t, z_hist)
        elif self.temporal == "gru":
            z, _ = self.gru(z_hist)
            z = z[:, -1, :]
        elif self.temporal == "none":
            z = z_t
        else:
            raise ValueError(f"Unknown temporal mode {self.temporal}")
        return self.classifier(z).squeeze(-1), z


class STCMixHop(GraphClassifier):
    def __init__(self, in_dim: int, hid_dim: int = 64, emb_dim: int = 64, K: int = 2, d_k: int = 128, dropout: float = 0.2, use_temporal_attention: bool = True):
        super().__init__(MixHopEncoder(in_dim, hid_dim, emb_dim, K=K, dropout=dropout), emb_dim, d_k=d_k, dropout=dropout, temporal="attention" if use_temporal_attention else "none")
        self.K = K


class TemporalGCNGRU(GraphClassifier):
    """T-GCN-style baseline: GCN structural encoder followed by GRU temporal fusion."""
    def __init__(self, in_dim: int, emb_dim: int = 64, dropout: float = 0.2):
        super().__init__(GCNEncoder(in_dim, emb_dim, dropout=dropout), emb_dim, dropout=dropout, temporal="gru")


class DySATLite(GraphClassifier):
    """DySAT-inspired dependency-light baseline: GAT structural encoder + temporal attention."""
    def __init__(self, in_dim: int, emb_dim: int = 64, d_k: int = 128, dropout: float = 0.2):
        super().__init__(GATEncoderLite(in_dim, emb_dim, dropout=dropout), emb_dim, d_k=d_k, dropout=dropout, temporal="attention")


class EvolveGCNLite(GraphClassifier):
    """EvolveGCN-inspired lightweight baseline using SAGE encoding and GRU temporal fusion."""
    def __init__(self, in_dim: int, emb_dim: int = 64, dropout: float = 0.2):
        super().__init__(SAGEEncoder(in_dim, emb_dim, dropout=dropout), emb_dim, dropout=dropout, temporal="gru")


def info_nce_loss(z_anchor: torch.Tensor, z_pos: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    logits = (z_anchor @ z_pos.T) / temperature
    labels = torch.arange(z_anchor.size(0), device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def dgi_loss(z: torch.Tensor, z_corrupt: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z.mean(dim=0, keepdim=True))
    pos = (z * s).sum(dim=1)
    neg = (z_corrupt * s).sum(dim=1)
    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
