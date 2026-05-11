"""
aggregation.py — Hidden-state aggregation and geometric feature extraction.

Per the README, applicants are explicitly encouraged to add hand-crafted
features "during the aggregation step, drawing on geometrical or topological
methods".  Since ``USE_GEOMETRIC`` in ``solution.py`` is part of the fixed
infrastructure (only ``aggregation.py``, ``probe.py``, ``splitting.py`` may
be edited), every feature in this file is produced inside ``aggregate``.
``extract_geometric_features`` is left as a documented hook returning an
empty tensor — flipping ``USE_GEOMETRIC`` has no effect either way.

What the probe sees
-------------------
For each sample the feature vector is the concatenation of:

A. Dense per-layer pooled embeddings ── ``len(SELECTED_LAYERS) * 2 * hidden_dim``
   * Last real-token hidden state at each selected mid-to-late layer
     (causal-attention summary at the end of the response).
   * Mean over the last K real tokens at each selected layer
     (response-region pool — averages out single-token noise).
   This is the SAPLMA / mass-mean line of work (Azaria & Mitchell 2023;
   Marks & Tegmark 2023).  Mid-to-late layers are picked because the
   "truthfulness direction" in decoder LMs is concentrated there.

B. Geometric / spectral descriptors ── ~84 features
   * Per-layer L2 norm of the last real token (25 features).
   * Inter-layer cosine drift of the last token (24 features).
   * Token-norm statistics in the response tail at the final layer.
   * Token-to-token cosine drift in the tail.
   * EigenScore-style features: top-k singular values of the centred
     response-tail state matrix, plus log-determinant proxy and effective
     rank — at four upper layers (INSIDE; Chen et al. ICLR 2024).
   * Real-token sequence length.

For ``Qwen/Qwen2.5-0.5B`` (24 transformer layers + 1 embedding,
hidden_dim=896, ``SELECTED_LAYERS = (12, 16, 20, 24)``) the resulting
feature dim is 4 * 2 * 896 + 84 = 7252.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Qwen2.5-0.5B has 24 transformer layers; outputs.hidden_states returns 25
# tensors (embedding + 24 layers).  Indexing matches solution.py:
#     hidden_states[0]  -> token embeddings
#     hidden_states[L]  -> after transformer layer L  (L = 1, ..., 24)
#     hidden_states[-1] -> final transformer layer
SELECTED_LAYERS = (12, 16, 20, 24)

# Number of trailing real tokens treated as the response region for mean
# pooling and spectral analysis.
RESPONSE_TAIL_K = 64

# Layers used for geometric eigen-features (top of the stack — strongest
# hallucination signal in 24-layer decoder LMs).
GEO_LAYERS = (18, 20, 22, 24)

# Top-K singular values kept per geometric layer.
GEO_TOP_SV = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between two 1-D tensors, robust to zero norm."""
    return torch.dot(a, b) / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))


def _pooled_features(
    hidden_states: torch.Tensor,
    real_idx: torch.Tensor,
    last_pos: int,
) -> list[torch.Tensor]:
    """Last-token + mean-tail pool at each selected layer."""
    n_real = int(real_idx.numel())
    tail_n = min(RESPONSE_TAIL_K, n_real)
    tail_pos = real_idx[-tail_n:]

    feats: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]                  # (seq_len, hidden_dim)
        feats.append(layer[last_pos])                     # last-token pool
        feats.append(layer.index_select(0, tail_pos).mean(dim=0))  # tail-mean pool
    return feats


def _geometric_features(
    hidden_states: torch.Tensor,
    real_idx: torch.Tensor,
    last_pos: int,
) -> list[torch.Tensor]:
    """Hand-crafted geometric / spectral descriptors."""
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
    # For each layer, take the response-tail states (tail_n x h), centre them,
    # run SVD, and keep top-k singular values + log-det proxy + effective rank.
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
    """Build the full feature vector: dense pooled embeddings + geometric stats.

    Args:
        hidden_states:  ``(n_layers + 1, seq_len, hidden_dim)``.
        attention_mask: ``(seq_len,)`` with 1 for real tokens, 0 for padding.

    Returns:
        Flat tensor of length
        ``len(SELECTED_LAYERS) * 2 * hidden_dim + ~84 (geo)`` = 7252 for
        Qwen2.5-0.5B.
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device, dtype=torch.bool)

    real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        # Degenerate sample; fall back to position 0 to keep shapes consistent.
        real_idx = torch.tensor([0], device=device)
    last_pos = int(real_idx[-1].item())

    pooled = _pooled_features(hidden_states, real_idx, last_pos)
    geo = _geometric_features(hidden_states, real_idx, last_pos)

    out = torch.cat([t.reshape(-1) for t in pooled + geo], dim=0).float()
    # Replace any NaN/Inf with zeros so the probe never sees garbage.
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hook kept for interface compatibility with ``solution.py``.

    All geometric / spectral features are produced inside ``aggregate``
    above so they are always included regardless of the value of
    ``USE_GEOMETRIC`` in ``solution.py`` (which is part of the fixed
    infrastructure and cannot be edited).  This function therefore returns
    an empty tensor and the value of ``USE_GEOMETRIC`` is irrelevant.
    """
    return torch.zeros(0, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Concatenate ``aggregate`` output with the (empty) geometric hook.

    The ``use_geometric`` flag is accepted for backward compatibility but has
    no effect: geometric features are already included by ``aggregate``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
