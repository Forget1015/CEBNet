"""
CEB-Net: Cognitive Episodic Buffer Network for Sequential Recommendation

Architecture:
  VQ Encoder (from CCFRec) → Perceptual Encoding → Split →
  WEBD (Working Memory) + SMC (Long-term Memory) → DEBR (Fusion) → z_u
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pywt
import numpy as np

from layers import CrossAttTransformer, Transformer


# ============================================================
# Helpers
# ============================================================
def gather_tensors(t):
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)
    all_tensors[dist.get_rank()] = t
    return torch.cat(all_tensors)


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, x, y, gathered=False):
        if gathered:
            all_y = gather_tensors(y)
        else:
            all_y = y
        x = F.normalize(x, dim=-1)
        all_y = F.normalize(all_y, dim=-1)
        logits = torch.matmul(x, all_y.T) / self.tau
        labels = torch.arange(x.size(0), device=x.device)
        return F.cross_entropy(logits, labels)


# ============================================================
# Module 1: WEBD — Wavelet-Enhanced Burstiness-preserving Denoiser
# ============================================================
class WaveletBurstDenoiser(nn.Module):
    def __init__(self, hidden_size, inner_size=256, n_heads=2, n_layers=1,
                 wavelet='haar', dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.hidden_size = hidden_size
        self.wavelet_name = wavelet

        # Rehearsal Encoder: causal Transformer
        self.rehearsal_encoder = Transformer(
            n_layers=n_layers, n_heads=n_heads,
            hidden_size=hidden_size, inner_size=inner_size,
            hidden_dropout_prob=dropout, attn_dropout_prob=dropout,
            hidden_act="gelu", layer_norm_eps=layer_norm_eps,
        )
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.rehearsal_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.rehearsal_dropout = nn.Dropout(dropout)

        # Register wavelet filters as buffers (created once)
        w = pywt.Wavelet(wavelet)
        self.filter_len = len(w.dec_lo)
        self.register_buffer('dec_lo', torch.tensor(w.dec_lo, dtype=torch.float32).flip(0))
        self.register_buffer('dec_hi', torch.tensor(w.dec_hi, dtype=torch.float32).flip(0))
        self.register_buffer('rec_lo', torch.tensor(w.rec_lo, dtype=torch.float32).flip(0))
        self.register_buffer('rec_hi', torch.tensor(w.rec_hi, dtype=torch.float32).flip(0))

        # Context-aware dynamic threshold
        self.threshold_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.threshold_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.dropout = nn.Dropout(dropout)

        # Attention pooling
        self.attn_pool_q = nn.Linear(hidden_size, hidden_size)
        self.attn_pool_k = nn.Linear(hidden_size, hidden_size)
        self.attn_pool_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _dwt_1d(self, x):
        """x: [B, L, d] -> cA, cD: [B, L', d]"""
        B, L, d = x.shape
        x_t = x.permute(0, 2, 1)  # [B, d, L]
        pad_len = self.filter_len - 1
        if pad_len < L:
            x_t = F.pad(x_t, (pad_len, pad_len), mode='reflect')
        else:
            x_t = F.pad(x_t, (pad_len, pad_len), mode='constant', value=0)
        lo = self.dec_lo.unsqueeze(0).unsqueeze(0).expand(d, -1, -1)
        hi = self.dec_hi.unsqueeze(0).unsqueeze(0).expand(d, -1, -1)
        cA = F.conv1d(x_t, lo, stride=2, groups=d).permute(0, 2, 1)
        cD = F.conv1d(x_t, hi, stride=2, groups=d).permute(0, 2, 1)
        return cA, cD

    def _idwt_1d(self, cA, cD, target_len):
        """cA, cD: [B, L', d] -> x_rec: [B, target_len, d]"""
        B, L_half, d = cA.shape
        cA_t = cA.permute(0, 2, 1)
        cD_t = cD.permute(0, 2, 1)
        up_A = torch.zeros(B, d, L_half * 2, dtype=cA.dtype, device=cA.device)
        up_A[:, :, 0::2] = cA_t
        up_D = torch.zeros(B, d, L_half * 2, dtype=cD.dtype, device=cD.device)
        up_D[:, :, 0::2] = cD_t
        pad_len = self.filter_len - 1
        up_len = L_half * 2
        if pad_len < up_len:
            up_A = F.pad(up_A, (pad_len, pad_len), mode='reflect')
            up_D = F.pad(up_D, (pad_len, pad_len), mode='reflect')
        else:
            up_A = F.pad(up_A, (pad_len, pad_len), mode='constant', value=0)
            up_D = F.pad(up_D, (pad_len, pad_len), mode='constant', value=0)
        lo = self.rec_lo.unsqueeze(0).unsqueeze(0).expand(d, -1, -1)
        hi = self.rec_hi.unsqueeze(0).unsqueeze(0).expand(d, -1, -1)
        rec_A = F.conv1d(up_A, lo, groups=d)
        rec_D = F.conv1d(up_D, hi, groups=d)
        min_len = min(rec_A.size(2), rec_D.size(2))
        rec = rec_A[:, :, :min_len] + rec_D[:, :, :min_len]
        if rec.size(2) >= target_len:
            rec = rec[:, :, :target_len]
        else:
            rec = F.pad(rec, (0, target_len - rec.size(2)))
        return rec.permute(0, 2, 1)

    def forward(self, x_wm, mask=None):
        """
        Args:
            x_wm: [B, m, d] working memory embeddings (already context-encoded)
            mask: [B, m] True=valid
        Returns:
            anchor: [B, d], x_before_dwt: [B, m, d], x_denoised: [B, m, d]
        """
        B, m, d = x_wm.shape

        # Rehearsal Encoding (causal)
        pos_ids = torch.arange(m, device=x_wm.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(pos_ids)
        inp = self.rehearsal_ln(self.rehearsal_dropout(x_wm + pos_emb))

        causal = torch.tril(torch.ones(m, m, device=x_wm.device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            attn_mask = mask.float().unsqueeze(1).unsqueeze(2) * causal
        else:
            attn_mask = causal
        attn_mask = (1.0 - attn_mask) * -1e9

        x_rehearsed = self.rehearsal_encoder(inp, inp, attn_mask)[-1]
        x_before_dwt = x_rehearsed.clone()

        # DWT
        cA, cD = self._dwt_1d(x_rehearsed)

        # Context-aware threshold
        if mask is not None:
            mf = mask.float().unsqueeze(-1)
            context = (x_rehearsed * mf).sum(1) / mf.sum(1).clamp(min=1)
        else:
            context = x_rehearsed.mean(1)
        threshold = self.threshold_net(context) * self.threshold_scale.abs()
        threshold = threshold.unsqueeze(1)

        # Soft thresholding
        cD_clean = torch.sign(cD) * F.relu(torch.abs(cD) - threshold)

        # IDWT
        x_denoised = self._idwt_1d(cA, cD_clean, m)
        x_denoised = self.dropout(x_denoised)

        # Attention pooling → anchor
        q = self.attn_pool_q(x_denoised)
        k = self.attn_pool_k(x_denoised)
        scores = (q * k).sum(-1) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        anchor = (x_denoised * weights).sum(1)
        anchor = self.attn_pool_ln(anchor)

        return anchor, x_before_dwt, x_denoised


# ============================================================
# Module 2: SMC — Semantic Memory Consolidation
# ============================================================
class SemanticMemoryConsolidation(nn.Module):
    def __init__(self, hidden_size, inner_size=256, n_heads=2, n_replay_layers=1,
                 n_prototypes=32, temperature=1.0, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_prototypes = n_prototypes
        self.temperature = temperature

        # Replay Encoder: bidirectional Transformer
        self.replay_encoder = Transformer(
            n_layers=n_replay_layers, n_heads=n_heads,
            hidden_size=hidden_size, inner_size=inner_size,
            hidden_dropout_prob=dropout, attn_dropout_prob=dropout,
            hidden_act="gelu", layer_norm_eps=layer_norm_eps,
        )
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.replay_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.replay_dropout = nn.Dropout(dropout)

        # Learnable prototype matrix P ∈ R^{K×d}
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, hidden_size))
        nn.init.xavier_uniform_(self.prototypes)

        self.proj_item = nn.Linear(hidden_size, hidden_size)
        self.proj_proto = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_long, mask=None):
        """
        Args:
            x_long: [B, n, d]
            mask: [B, n] True=valid
        Returns:
            memory: [B, K, d]
        """
        B, n, d = x_long.shape

        # Replay Encoding (bidirectional)
        pos_ids = torch.arange(n, device=x_long.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(pos_ids)
        inp = self.replay_ln(self.replay_dropout(x_long + pos_emb))

        if mask is not None:
            attn_mask = mask.float().unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attn_mask) * -1e9
        else:
            attn_mask = torch.zeros(B, 1, 1, n, device=x_long.device)

        x_replayed = self.replay_encoder(inp, inp, attn_mask)[-1]

        # Prototype assignment
        item_proj = self.proj_item(x_replayed)
        proto_proj = self.proj_proto(self.prototypes)
        sim = torch.matmul(item_proj, proto_proj.T) / math.sqrt(d)
        assign = F.softmax(sim / self.temperature, dim=-1)  # [B, n, K]

        # Zero out padding positions explicitly
        if mask is not None:
            assign = assign * mask.float().unsqueeze(-1)

        # Weighted aggregation
        assign_T = assign.permute(0, 2, 1)  # [B, K, n]
        memory = torch.bmm(assign_T, x_replayed)  # [B, K, d]
        assign_sum = assign_T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        memory = memory / assign_sum

        return memory

    def compute_ortho_loss(self):
        """Max cosine similarity penalty."""
        P = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(P, P.T)
        eye = torch.eye(self.n_prototypes, device=sim.device).bool()
        sim = sim.masked_fill(eye, -1e9)
        max_sim, _ = sim.max(dim=-1)
        return max_sim.mean()


# ============================================================
# Module 3: DEBR — Decoupled Episodic Buffer Retrieval
# ============================================================
class DecoupledEpisodicBuffer(nn.Module):
    def __init__(self, hidden_size, attn_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_size = attn_size or hidden_size

        self.W_attn_anchor = nn.Linear(hidden_size, self.attn_size)
        self.W_attn_memory = nn.Linear(hidden_size, self.attn_size)
        self.W_repr_memory = nn.Linear(hidden_size, hidden_size)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def retrieve(self, anchor, memory):
        q = self.W_attn_anchor(anchor)
        k = self.W_attn_memory(memory)
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.attn_size)
        w = F.softmax(scores, dim=-1)
        v = self.W_repr_memory(memory)
        return torch.bmm(w.unsqueeze(1), v).squeeze(1)

    def forward(self, anchor, memory):
        """
        Args:
            anchor: [B, d]
            memory: [B, K, d]
        Returns:
            z_u: [B, d]
        """
        z_long = self.retrieve(anchor, memory)
        g = self.fusion_gate(torch.cat([anchor, z_long], dim=-1))
        z_u = g * anchor + (1 - g) * z_long
        return self.layer_norm(z_u)


# ============================================================
# Frequency Consistency Loss
# ============================================================
class FrequencyConsistencyLoss(nn.Module):
    def forward(self, x_before, x_after, mask=None):
        """MSE between magnitude spectra of x_before and x_after on seq dim."""
        F_before = torch.fft.rfft(x_before, dim=1)
        F_after = torch.fft.rfft(x_after, dim=1)
        diff = (torch.abs(F_after) - torch.abs(F_before)) ** 2
        return diff.sum(dim=1).mean(dim=-1).mean()


class SemanticCalibrationLayer(nn.Module):
    def __init__(self, hidden_size, max_seq_length, weight=0.2, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.weight = weight
        self.freq_gate = nn.Parameter(torch.zeros(max_seq_length // 2 + 1, hidden_size))
        self.residual_gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, item_emb, item_seq_len):
        B, L, _ = item_emb.size()
        valid = torch.arange(L, device=item_emb.device).unsqueeze(0) < item_seq_len.unsqueeze(1)
        valid_mask = valid.unsqueeze(-1)

        x = item_emb * valid_mask.to(item_emb.dtype)
        spectrum = torch.fft.rfft(x.float(), dim=1, norm='ortho')
        freq_gate = torch.sigmoid(self.freq_gate[:spectrum.size(1)]).unsqueeze(0)
        calibrated = torch.fft.irfft(spectrum * freq_gate, n=L, dim=1, norm='ortho').to(item_emb.dtype)

        residual_gate = torch.sigmoid(self.residual_gate(item_emb))
        updated = self.layer_norm(item_emb + self.weight * residual_gate * self.dropout(calibrated))
        return torch.where(valid_mask, updated, item_emb)


# ============================================================
# Main Model: CEB-Net
# ============================================================
class CEBNet(nn.Module):
    def __init__(self, args, dataset, index, device):
        super().__init__()

        # === Hyperparams ===
        self.n_layers = args.n_layers
        self.n_layers_cross = args.n_layers_cross
        self.n_heads = args.n_heads
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.neg_num = args.neg_num
        self.text_num = len(args.text_types)
        self.max_seq_length = args.max_his_len
        self.code_level = args.code_level
        self.n_codes_per_lel = args.n_codes_per_lel
        self.hidden_dropout_prob = args.dropout_prob
        self.attn_dropout_prob = args.dropout_prob
        self.hidden_dropout_prob_cross = args.dropout_prob_cross
        self.attn_dropout_prob_cross = args.dropout_prob_cross
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02

        # CEB-Net specific
        self.wm_length = getattr(args, 'wm_length', 5)
        self.n_prototypes = getattr(args, 'n_prototypes', 16)
        self.wavelet = getattr(args, 'wavelet', 'haar')
        self.ortho_weight = getattr(args, 'ortho_weight', 0.1)
        self.freq_weight = getattr(args, 'freq_weight', 0.01)
        self.attn_size = getattr(args, 'attn_size', None)
        self.proto_temperature = getattr(args, 'proto_temperature', 1.0)
        self.n_layers_webd = getattr(args, 'n_layers_webd', 1)
        self.n_layers_smc = getattr(args, 'n_layers_smc', 1)

        # === VQ index ===
        index[0] = [0] * self.code_level
        self.index = torch.tensor(index, dtype=torch.long, device=device)
        for i in range(self.code_level):
            self.index[:, i] += i * self.n_codes_per_lel + 1

        self.n_items = dataset.n_items + 1
        self.n_codes = args.n_codes_per_lel * args.code_level + 1
        self.tau = args.tau
        self.cl_weight = args.cl_weight
        self.mlm_weight = args.mlm_weight
        self.device = device

        # === Embeddings (from CCFRec) ===
        self.query_code_embedding = nn.Embedding(self.n_codes, self.embedding_size, padding_idx=0)
        self.item_text_embedding = nn.ModuleList([
            nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
            for _ in range(self.text_num)
        ])
        self.item_text_embedding.requires_grad_(False)
        self.use_id_residual = getattr(args, 'use_id_residual', False)
        if self.use_id_residual:
            self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
            self.id_residual_gate = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.use_decoupled_trace_id = getattr(args, 'use_decoupled_trace_id', False)
        self.trace_id_gate_bias_init = getattr(args, 'trace_id_gate_bias_init', -2.0)
        if self.use_decoupled_trace_id:
            self.trace_item_id_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
            self.trace_id_gate = nn.Linear(self.embedding_size * 2, self.embedding_size)
            self.trace_id_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        # Q-Former (from CCFRec)
        self.qformer = CrossAttTransformer(
            n_layers=self.n_layers_cross, n_heads=self.n_heads,
            hidden_size=self.embedding_size, inner_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob_cross,
            attn_dropout_prob=self.attn_dropout_prob_cross,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps,
        )

        # === Position embedding for sequence ===
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # === CEB-Net core modules ===
        self.webd = WaveletBurstDenoiser(
            self.embedding_size, inner_size=self.hidden_size,
            n_heads=self.n_heads, n_layers=self.n_layers_webd,
            wavelet=self.wavelet, dropout=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.smc = SemanticMemoryConsolidation(
            self.embedding_size, inner_size=self.hidden_size,
            n_heads=self.n_heads, n_replay_layers=self.n_layers_smc,
            n_prototypes=self.n_prototypes, temperature=self.proto_temperature,
            dropout=self.hidden_dropout_prob, layer_norm_eps=self.layer_norm_eps,
        )
        self.debr = DecoupledEpisodicBuffer(
            self.embedding_size, attn_size=self.attn_size
        )

        self.use_semantic_calibration = getattr(args, 'use_semantic_calibration', False)
        self.calibration_mode = getattr(args, 'calibration_mode', 'fft')
        self.calibration_weight = getattr(args, 'calibration_weight', 0.2)
        self.calibration_gate_bias_init = getattr(args, 'calibration_gate_bias_init', -2.0)
        if self.use_semantic_calibration:
            self.semantic_calibration = SemanticCalibrationLayer(
                self.embedding_size, self.max_seq_length,
                weight=self.calibration_weight,
                dropout=self.hidden_dropout_prob,
                layer_norm_eps=self.layer_norm_eps,
            )

        self.use_seq_branch = getattr(args, 'use_seq_branch', False)
        self.n_layers_seq = getattr(args, 'n_layers_seq', 2)
        self.seq_fusion_mode = getattr(args, 'seq_fusion_mode', 'gate')
        self.seq_gate_bias_init = getattr(args, 'seq_gate_bias_init', -2.0)
        self.seq_add_weight = getattr(args, 'seq_add_weight', 0.2)
        self.trace_memory_gate_bias_init = getattr(args, 'trace_memory_gate_bias_init', -2.0)
        self.trace_aux_rec_weight = getattr(args, 'trace_aux_rec_weight', 0.0)
        self.history_neg_weight = getattr(args, 'history_neg_weight', 0.0)
        self.history_neg_num = getattr(args, 'history_neg_num', 20)
        if self.use_seq_branch:
            self.seq_encoder = Transformer(
                n_layers=self.n_layers_seq, n_heads=self.n_heads,
                hidden_size=self.embedding_size, inner_size=self.hidden_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps,
            )
            self.seq_layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
            self.seq_dropout = nn.Dropout(self.hidden_dropout_prob)
            self.seq_output_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
            if self.seq_fusion_mode == 'gate':
                self.seq_fusion_gate = nn.Linear(self.embedding_size * 2, self.embedding_size)
            if self.seq_fusion_mode == 'trace_residual_debr':
                self.trace_memory_gate = nn.Linear(self.embedding_size * 2, self.embedding_size)
            self.seq_fusion_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        self.freq_loss_fn = FrequencyConsistencyLoss()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Pre-cache for negative sampling optimization
        self._all_item_array = None

        self.apply(self._init_weights)
        if self.use_decoupled_trace_id:
            nn.init.constant_(self.trace_id_gate.bias, self.trace_id_gate_bias_init)
        if self.use_semantic_calibration:
            nn.init.constant_(self.semantic_calibration.residual_gate.bias, self.calibration_gate_bias_init)
        if self.use_seq_branch and self.seq_fusion_mode == 'gate':
            nn.init.constant_(self.seq_fusion_gate.bias, self.seq_gate_bias_init)
        if self.use_seq_branch and self.seq_fusion_mode == 'trace_residual_debr':
            nn.init.constant_(self.trace_memory_gate.bias, self.trace_memory_gate_bias_init)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _apply_id_residual(self, item_ids, semantic_emb):
        if not self.use_id_residual:
            return semantic_emb
        id_emb = self.item_id_embedding(item_ids)
        gate = torch.sigmoid(self.id_residual_gate(torch.cat([semantic_emb, id_emb], dim=-1)))
        return gate * semantic_emb + (1.0 - gate) * id_emb

    def _apply_decoupled_trace_id(self, item_ids, semantic_emb):
        if not self.use_decoupled_trace_id:
            return semantic_emb
        id_emb = self.trace_item_id_embedding(item_ids)
        gate = torch.sigmoid(self.trace_id_gate(torch.cat([semantic_emb, id_emb], dim=-1)))
        return self.trace_id_norm(semantic_emb + gate * id_emb)

    def _encode_full_seq_branch(self, item_emb, item_seq_len):
        B, L, _ = item_emb.size()
        pos_ids = torch.arange(L, device=item_emb.device).unsqueeze(0).expand(B, -1)
        seq_inp = self.seq_layer_norm(self.seq_dropout(item_emb + self.position_embedding(pos_ids)))

        valid = torch.arange(L, device=item_emb.device).unsqueeze(0) < item_seq_len.unsqueeze(1)
        causal = torch.tril(torch.ones(L, L, device=item_emb.device, dtype=torch.bool))
        attn_mask = valid.unsqueeze(1).unsqueeze(2) & causal.unsqueeze(0).unsqueeze(0)
        attn_mask = (1.0 - attn_mask.float()) * -1e9

        seq_hidden = self.seq_encoder(seq_inp, seq_inp, attn_mask)[-1]
        return self.seq_output_norm(seq_hidden)

    def _encode_seq_branch(self, item_emb, item_seq_len):
        B, _, _ = item_emb.size()
        seq_hidden = self._encode_full_seq_branch(item_emb, item_seq_len)
        last_idx = item_seq_len.clamp(min=1) - 1
        batch_idx = torch.arange(B, device=item_emb.device)
        return seq_hidden[batch_idx, last_idx]

    def _fuse_seq_branch(self, z_ceb, z_seq):
        if self.seq_fusion_mode == 'seq_only':
            return self.seq_fusion_norm(z_seq)
        if self.seq_fusion_mode == 'add':
            return self.seq_fusion_norm(z_ceb + self.seq_add_weight * z_seq)
        gate = torch.sigmoid(self.seq_fusion_gate(torch.cat([z_ceb, z_seq], dim=-1)))
        return self.seq_fusion_norm((1.0 - gate) * z_ceb + gate * z_seq)

    def _fuse_trace_memory(self, z_trace, memory):
        r_mem = self.debr.retrieve(z_trace, memory)
        gate = torch.sigmoid(self.trace_memory_gate(torch.cat([z_trace, r_mem], dim=-1)))
        return self.seq_fusion_norm(z_trace + gate * r_mem)

    def _encode_items(self, item_seq, code_seq):
        """VQ Encoder: Q-Former + query residual. Returns item_emb [B,L,d] and code_emb [B*L,C,d]."""
        B, L = item_seq.size()
        item_flatten = item_seq.reshape(-1)
        query_emb = self.query_code_embedding(code_seq)  # [B*L, C, d]

        text_embs = []
        for j in range(self.text_num):  # fixed: use j not i
            text_embs.append(self.item_text_embedding[j](item_flatten))
        encoder_output = torch.stack(text_embs, dim=1)  # [B*L, text_num, d]

        item_seq_emb = self.qformer(query_emb, encoder_output)[-1]  # [B*L, C, d]
        item_emb = self._apply_id_residual(item_flatten, item_seq_emb.mean(dim=1) + query_emb.mean(dim=1))  # [B*L, d]
        item_emb = item_emb.view(B, L, -1)

        return item_emb, item_seq_emb

    def forward(self, item_seq, item_seq_len, code_seq, return_trace=False):
        """
        CEB-Net forward pass:
        1. VQ encode → item embeddings (with query residual)
        2. Split into working memory + long-term history (no outer PE)
        3. WEBD: working memory denoising → anchor
        4. SMC: long-term history consolidation → memory
        5. DEBR: decoupled retrieval + fusion → z_u

        Returns: z_u [B,d], code_emb [B*L,C,d], x_before_dwt [B,m,d], x_denoised [B,m,d]
        """
        B, L = item_seq.size()
        d = self.embedding_size
        m = min(self.wm_length, L)

        # Step 1: VQ encode
        item_emb, code_emb = self._encode_items(item_seq, code_seq)
        if self.use_semantic_calibration:
            item_emb = self.semantic_calibration(item_emb, item_seq_len)
        trace_item_emb = self._apply_decoupled_trace_id(item_seq, item_emb)
        if self.use_seq_branch and self.seq_fusion_mode == 'seq_first_debr':
            split_item_emb = self._encode_full_seq_branch(trace_item_emb, item_seq_len)
        else:
            split_item_emb = item_emb

        # Step 2: Split (no outer PE — WEBD and SMC handle their own)
        # For-loop split: right-aligned working memory
        x_wm = torch.zeros(B, m, d, device=self.device)
        wm_valid = torch.zeros(B, m, dtype=torch.bool, device=self.device)

        for i in range(B):
            seq_len = item_seq_len[i].item()
            wm_start = max(0, seq_len - m)
            actual_wm_len = min(m, seq_len)
            x_wm[i, m - actual_wm_len:] = split_item_emb[i, wm_start:seq_len]
            wm_valid[i, m - actual_wm_len:] = True

        x_wm_raw = x_wm.clone()

        long_len = L - m
        if long_len > 0:
            actual_long_lens = (item_seq_len - m).clamp(min=0)
            x_long = split_item_emb[:, :long_len, :]
            long_valid = torch.zeros(B, long_len, dtype=torch.bool, device=self.device)
            for i in range(B):
                al = actual_long_lens[i].item()
                if al > 0:
                    long_valid[i, :al] = True
                else:
                    last_valid_idx = item_seq_len[i].item() - 1
                    if last_valid_idx >= 0:
                        x_long[i, 0] = split_item_emb[i, last_valid_idx]
                        long_valid[i, 0] = True
        else:
            x_long = x_wm_raw.clone()
            long_valid = wm_valid.clone()

        # Step 4: WEBD — working memory denoising
        anchor, x_before_dwt, x_denoised = self.webd(x_wm, mask=wm_valid)

        # Step 5: SMC — long-term memory consolidation
        memory = self.smc(x_long, mask=long_valid)

        # Step 6: DEBR — decoupled retrieval and fusion
        z_trace = None
        if self.use_seq_branch and self.seq_fusion_mode == 'trace_residual_debr':
            z_trace = self._encode_seq_branch(trace_item_emb, item_seq_len)
            working_memory = x_denoised * wm_valid.unsqueeze(-1).to(x_denoised.dtype)
            trace_memory = torch.cat([working_memory, memory], dim=1)
            z_u = self._fuse_trace_memory(z_trace, trace_memory)
        else:
            z_ceb = self.debr(anchor, memory)
            if self.use_seq_branch:
                if self.seq_fusion_mode == 'seq_first_debr':
                    last_idx = item_seq_len.clamp(min=1) - 1
                    batch_idx = torch.arange(B, device=item_emb.device)
                    z_trace = split_item_emb[batch_idx, last_idx]
                    z_u = z_ceb
                else:
                    z_seq = self._encode_seq_branch(trace_item_emb, item_seq_len)
                    z_trace = z_seq
                    z_u = self._fuse_seq_branch(z_ceb, z_seq)
            else:
                z_u = z_ceb

        if return_trace:
            return z_u, code_emb, x_before_dwt, x_denoised, z_trace
        return z_u, code_emb, x_before_dwt, x_denoised

    @torch.no_grad()
    def get_item_embedding(self):
        """Compute all item embeddings for full-sort prediction."""
        batch_size = 1024
        all_items = torch.arange(self.n_items, device=self.device)
        n_batches = (self.n_items + batch_size - 1) // batch_size

        item_embedding = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.n_items)
            batch_item = all_items[start:end]
            batch_query = self.index[batch_item]
            batch_query_emb = self.query_code_embedding(batch_query)

            text_embs = []
            for j in range(self.text_num):  # fixed: j not i
                text_embs.append(self.item_text_embedding[j](batch_item))
            batch_encoder_output = torch.stack(text_embs, dim=1)

            batch_item_seq_emb = self.qformer(batch_query_emb, batch_encoder_output)[-1]
            batch_item_emb = self._apply_id_residual(
                batch_item, batch_item_seq_emb.mean(dim=1) + batch_query_emb.mean(dim=1))
            batch_item_emb = self._apply_decoupled_trace_id(batch_item, batch_item_emb)
            item_embedding.append(batch_item_emb)

        return torch.cat(item_embedding, dim=0)

    def _encode_item_ids(self, batch_item):
        batch_query = self.index[batch_item]
        batch_query_emb = self.query_code_embedding(batch_query)

        text_embs = []
        for j in range(self.text_num):
            text_embs.append(self.item_text_embedding[j](batch_item))
        batch_encoder_output = torch.stack(text_embs, dim=1)

        batch_item_seq_emb = self.qformer(batch_query_emb, batch_encoder_output)[-1]
        batch_item_emb = self._apply_id_residual(
            batch_item, batch_item_seq_emb.mean(dim=1) + batch_query_emb.mean(dim=1))
        return self._apply_decoupled_trace_id(batch_item, batch_item_emb)

    def encode_item(self, pos_items):
        """Encode pos + neg items for training loss."""
        pos_list = pos_items.cpu().tolist()

        # Optimized negative sampling: cache item array
        if self._all_item_array is None or len(self._all_item_array) != self.n_items - 1:
            self._all_item_array = np.arange(1, self.n_items)

        # Exclude positive items
        neg_pool = self._all_item_array
        if self.neg_num > 0:
            candidates = np.random.choice(neg_pool, size=min(self.neg_num, len(neg_pool)),
                                          replace=False).tolist()
        else:
            candidates = []

        B = len(pos_list)
        batch_item = torch.tensor(pos_list + candidates, device=self.device)
        batch_item_emb = self._encode_item_ids(batch_item)

        return batch_item_emb[:B], batch_item_emb[B:]

    def _history_negative_loss(self, z_u, item_seq, item_seq_len, pos_logits):
        if self.history_neg_weight <= 0 or self.history_neg_num <= 0:
            return z_u.new_tensor(0.0)

        hist_items = []
        hist_rows = []
        for row in range(item_seq.size(0)):
            hist_len = item_seq_len[row].item()
            if hist_len <= 0:
                continue
            start = max(0, hist_len - self.history_neg_num)
            row_items = item_seq[row, start:hist_len]
            if row_items.numel() == 0:
                continue
            hist_items.append(row_items)
            hist_rows.append(torch.full((row_items.numel(),), row, device=self.device, dtype=torch.long))

        if not hist_items:
            return z_u.new_tensor(0.0)

        hist_items = torch.cat(hist_items, dim=0)
        hist_rows = torch.cat(hist_rows, dim=0)
        hist_emb = self._encode_item_ids(hist_items)
        hist_emb = F.normalize(hist_emb, dim=-1)
        hist_logits = (z_u[hist_rows] * hist_emb).sum(dim=-1) / self.tau
        pos_margin = pos_logits.squeeze(-1).detach()[hist_rows]
        return F.softplus(hist_logits - pos_margin).mean()

    def calculate_loss(self, item_seq, item_seq_len, pos_items, code_seq_mask, labels_mask):
        B, L = item_seq.size()
        code_seq = self.index[item_seq].reshape(B * L, -1)

        # Normal forward
        use_trace_aux = self.trace_aux_rec_weight > 0 and self.use_seq_branch
        if use_trace_aux:
            z_u, code_output, x_before_dwt, x_denoised, z_trace = self.forward(
                item_seq, item_seq_len, code_seq, return_trace=True)
        else:
            z_u, code_output, x_before_dwt, x_denoised = self.forward(item_seq, item_seq_len, code_seq)
            z_trace = None
        # Masked forward (for contrastive learning)
        z_u_mask, code_output_mask, _, _ = self.forward(item_seq, item_seq_len, code_seq_mask)

        z_u = F.normalize(z_u, dim=-1)
        z_u_mask = F.normalize(z_u_mask, dim=-1)
        if z_trace is not None:
            z_trace = F.normalize(z_trace, dim=-1)

        # 1. Recommendation loss
        trace_aux_rec_loss = z_u.new_tensor(0.0)
        history_neg_loss = z_u.new_tensor(0.0)
        if self.neg_num > 0:
            pos_emb, neg_emb = self.encode_item(pos_items)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)
            labels = torch.zeros(B, device=self.device).long()

            pos_logits = torch.bmm(z_u.unsqueeze(1), pos_emb.unsqueeze(2)).squeeze(-1) / self.tau
            neg_logits = torch.matmul(z_u, neg_emb.T) / self.tau
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            rec_loss = self.loss_fct(logits, labels)
            history_neg_loss = self._history_negative_loss(z_u, item_seq, item_seq_len, pos_logits)

            if z_trace is not None:
                trace_pos_logits = torch.bmm(z_trace.unsqueeze(1), pos_emb.unsqueeze(2)).squeeze(-1) / self.tau
                trace_neg_logits = torch.matmul(z_trace, neg_emb.T) / self.tau
                trace_logits = torch.cat([trace_pos_logits, trace_neg_logits], dim=1)
                trace_aux_rec_loss = self.loss_fct(trace_logits, labels)
        else:
            all_emb = F.normalize(self.get_item_embedding(), dim=-1)
            logits = torch.matmul(z_u, all_emb.T) / self.tau
            rec_loss = self.loss_fct(logits, pos_items)
            pos_logits = logits.gather(1, pos_items.unsqueeze(1))
            history_neg_loss = self._history_negative_loss(z_u, item_seq, item_seq_len, pos_logits)
            if z_trace is not None:
                trace_logits = torch.matmul(z_trace, all_emb.T) / self.tau
                trace_aux_rec_loss = self.loss_fct(trace_logits, pos_items)

        # 2. Contrastive loss
        gathered = dist.is_initialized()
        cl_fn = ContrastiveLoss(tau=self.tau)
        cl_loss = (cl_fn(z_u, z_u_mask, gathered=gathered) +
                   cl_fn(z_u_mask, z_u, gathered=gathered)) / 2

        # 3. Masked code modeling loss
        H = z_u.shape[-1]
        code_emb_weight = F.normalize(self.query_code_embedding.weight, dim=-1)
        code_out_flat = F.normalize(code_output_mask.view(-1, H), dim=-1)
        mlm_logits = torch.matmul(code_out_flat, code_emb_weight.T) / self.tau
        mlm_loss = self.loss_fct(mlm_logits, labels_mask)

        # 4. Orthogonal loss
        ortho_loss = self.smc.compute_ortho_loss()

        # 5. Frequency consistency loss
        freq_loss = self.freq_loss_fn(x_before_dwt, x_denoised)

        # Total
        loss = (rec_loss
                + self.trace_aux_rec_weight * trace_aux_rec_loss
                + self.history_neg_weight * history_neg_loss
                + self.cl_weight * cl_loss
                + self.mlm_weight * mlm_loss
                + self.ortho_weight * ortho_loss
                + self.freq_weight * freq_loss)

        return dict(loss=loss, rec_loss=rec_loss, trace_aux_rec_loss=trace_aux_rec_loss,
                    history_neg_loss=history_neg_loss, cl_loss=cl_loss, mlm_loss=mlm_loss,
                    ortho_loss=ortho_loss, freq_loss=freq_loss)

    def full_sort_predict(self, item_seq, item_seq_len, code_seq):
        z_u, _, _, _ = self.forward(item_seq, item_seq_len, code_seq)
        z_u = F.normalize(z_u, dim=-1)
        item_emb = F.normalize(self.get_item_embedding(), dim=-1)
        return torch.matmul(z_u, item_emb.T)
