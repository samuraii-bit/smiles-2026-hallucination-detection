"""
aggregation.py — Hidden-state aggregation and geometric feature extraction.

Strategy
--------
With ~440 training samples and a 896-dim hidden state, packing many layers into
a single 7000+ dim vector and feeding it to one classifier leads to severe
over-fitting (train AUROC 100%, test AUROC ~72% in the previous run).

The new layout keeps **predictable per-layer / per-pool slices** so the probe in
``probe.py`` can train a *separate* tiny logistic regression on each slice and
ensemble the results.  Each sub-probe only sees 896 features and 440 samples —
a much healthier ratio than 7252 features and 440 samples.

Feature vector layout
---------------------
For each layer L in ``SELECTED_LAYERS`` we emit two 896-dim pooled views:
  * last real-token hidden state (causal-attention summary at the end of the
    assistant response).
  * mean over the last K real tokens at L (response-region pool).

Then we append a compact block of geometric / spectral descriptors (~84 dims).
``SLICE_LAYOUT`` (exported below) maps every slice to its position in the flat
vector, so the probe knows exactly which feature dimensions belong to which
layer/pool.  This module-level metadata is the only "fixed-infrastructure-
compatible" way to share structure with ``probe.py``.

For ``Qwen/Qwen2.5-0.5B`` (24 transformer layers + 1 embedding, hidden_dim=896,
``SELECTED_LAYERS = (8,10,12,14,16,18,20,22,24)``) the resulting feature dim is
9 * 2 * 896 + 84 = 16212.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Sample every other layer in the mid-to-late stack — that is where the
# "truthfulness direction" emerges in decoder LMs (Azaria & Mitchell 2023,
# Marks & Tegmark 2023).  More layers => more ensemble diversity for the
# per-layer linear probes in probe.py, without runtime cost.
SELECTED_LAYERS: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24)

# Number of trailing real tokens treated as the response region for mean
# pooling and spectral analysis.
RESPONSE_TAIL_K = 64

# Layers used for geometric eigen-features (top of the stack).
GEO_LAYERS = (18, 20, 22, 24)
GEO_TOP_SV = 5

# Qwen2.5-0.5B hidden dim — used to declare slice sizes up front.
HIDDEN_DIM = 896

# Pool names (kept stable so probe.py can iterate over them).
POOLS: tuple[str, ...] = ("last", "mean")


# ---------------------------------------------------------------------------
# Slice layout — shared with probe.py
# ---------------------------------------------------------------------------
def _build_slice_layout() -> list[dict]:
    """Return per-slice metadata: name, start, end indices in the flat vector.

    Layout: [layer8_last, layer8_mean, layer10_last, layer10_mean, ..., GEO]
    """
    layout: list[dict] = []
    cur = 0
    for L in SELECTED_LAYERS:
        for pool in POOLS:
            layout.append(
                {
                    "name": f"L{L}_{pool}",
                    "layer": L,
                    "pool": pool,
                    "start": cur,
                    "end": cur + HIDDEN_DIM,
                }
            )
            cur += HIDDEN_DIM
    return layout


SLICE_LAYOUT: list[dict] = _build_slice_layout()
GEO_START: int = SLICE_LAYOUT[-1]["end"]   # everything past this is geo features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.dot(a, b) / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))


def _pooled_features(
    hidden_states: torch.Tensor,
    real_idx: torch.Tensor,
    last_pos: int,
) -> list[torch.Tensor]:
    """last-token + mean-tail pool at each selected layer, in SLICE_LAYOUT order."""
    n_real = int(real_idx.numel())
    tail_n = min(RESPONSE_TAIL_K, n_real)
    tail_pos = real_idx[-tail_n:]

    feats: list[torch.Tensor] = []
    for L in SELECTED_LAYERS:
        layer = hidden_states[L]                              # (seq_len, hidden_dim)
        feats.append(layer[last_pos])                          # last-token
        feats.append(layer.index_select(0, tail_pos).mean(0))  # tail-mean
    return feats


def _geometric_features(
    hidden_states: torch.Tensor,
    real_idx: torch.Tensor,
    last_pos: int,
) -> list[torch.Tensor]:
    """Hand-crafted geometric / spectral descriptors (~84 features)."""
    device = hidden_states.device
    n_layers_total = hidden_states.size(0)
    n_real = int(real_idx.numel())
    tail_n = min(RESPONSE_TAIL_K, n_real)
    tail_pos = real_idx[-tail_n:]

    feats: list[torch.Tensor] = []

    # (1) Per-layer norm of last real token (25 features).
    feats.append(torch.stack(
        [hidden_states[L, last_pos].norm() for L in range(n_layers_total)]
    ))

    # (2) Inter-layer cosine drift of last real token (24 features).
    feats.append(torch.stack(
        [
            _safe_cosine(hidden_states[L, last_pos], hidden_states[L + 1, last_pos])
            for L in range(n_layers_total - 1)
        ]
    ))

    # (3) Token-norm stats at the final layer over the response tail (4).
    final_layer = hidden_states[-1]
    tail_states = final_layer.index_select(0, tail_pos)
    tok_norms = tail_states.norm(dim=-1)
    feats.append(torch.stack([
        tok_norms.mean(),
        tok_norms.std(unbiased=False),
        tok_norms.min(),
        tok_norms.max(),
    ]))

    # (4) Token-to-token cosine drift in the tail at the final layer (2).
    if tail_n >= 2:
        a = tail_states[:-1]
        b = tail_states[1:]
        cos_tt = (a * b).sum(dim=-1) / (
            a.norm(dim=-1).clamp_min(1e-8) * b.norm(dim=-1).clamp_min(1e-8)
        )
        feats.append(torch.stack([cos_tt.mean(), cos_tt.std(unbiased=False)]))
    else:
        feats.append(torch.zeros(2, device=device))

    # (5) EigenScore-style spectral features per upper layer.
    sv_per_layer: list[torch.Tensor] = []
    log_det_proxy: list[torch.Tensor] = []
    eff_rank: list[torch.Tensor] = []
    for L in GEO_LAYERS:
        states = hidden_states[L].index_select(0, tail_pos)
        if tail_n >= 2:
            centered = states - states.mean(dim=0, keepdim=True)
            try:
                sv = torch.linalg.svdvals(centered)
            except Exception:
                sv = torch.zeros(min(tail_n, states.size(-1)), device=device)
        else:
            sv = torch.zeros(1, device=device)

        sv = sv[:GEO_TOP_SV]
        if sv.numel() < GEO_TOP_SV:
            pad = torch.zeros(GEO_TOP_SV - sv.numel(), device=device)
            sv = torch.cat([sv, pad])
        sv_per_layer.append(sv)

        log_det_proxy.append(torch.log(sv.clamp_min(1e-6)).sum().unsqueeze(0))

        sv_norm = sv / sv.sum().clamp_min(1e-8)
        entropy = -(sv_norm.clamp_min(1e-12) * torch.log(sv_norm.clamp_min(1e-12))).sum()
        eff_rank.append(torch.exp(entropy).unsqueeze(0))

    feats.append(torch.cat(sv_per_layer, dim=0))            # 4 * 5 = 20
    feats.append(torch.cat(log_det_proxy, dim=0))           # 4
    feats.append(torch.cat(eff_rank, dim=0))                # 4

    # (6) Sequence length, normalised (1).
    feats.append(torch.tensor([float(n_real) / 512.0], device=device))

    return feats


# ---------------------------------------------------------------------------
# Public functions called by solution.py
# ---------------------------------------------------------------------------
def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the full feature vector: dense pooled embeddings + geometric stats."""
    device = hidden_states.device
    mask = attention_mask.to(device=device, dtype=torch.bool)

    real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        real_idx = torch.tensor([0], device=device)
    last_pos = int(real_idx[-1].item())

    pooled = _pooled_features(hidden_states, real_idx, last_pos)
    geo = _geometric_features(hidden_states, real_idx, last_pos)

    out = torch.cat([t.reshape(-1) for t in pooled + geo], dim=0).float()
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """All geometric features are emitted inside ``aggregate``."""
    return torch.zeros(0, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Concatenate ``aggregate`` output with the (empty) geometric hook."""
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
