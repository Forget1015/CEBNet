"""
Case 3: ML-1M SMC prototype visualization.
Position = Σ_k (proto_profile[k] * prototype_emb[k])
Color    = argmax(proto_profile)
Run from: /data0/yejinxuan/workspace/CEBNet
"""
import sys, os
os.chdir('/data0/yejinxuan/workspace/CEBNet')
sys.path.insert(0, '.')

import numpy as np
import json, math, types
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from data import load_split_data, CEBNetDataset, Collator
from model import CEBNet
from utils import load_json

CKPT = ('./myckpt/ML-1M_TedRec/'
        'Jun-27-2026_05-32-c9d1c8_wm20_K16_wavhaar_mlm0.6_cl0.4_drop0.2_'
        'dpcross0.2_seqL2_trace_residual_debr/best_model.pth')
DEVICE     = 'cuda:0'
N_BATCHES  = 400
BATCH_SIZE = 64

plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 10,
                     'axes.linewidth': 1.0})

# ── Load model ────────────────────────────────────────────────────────────────
state = torch.load(CKPT, map_location='cpu', weights_only=False)
args  = state['args']
args.device = DEVICE; args.neg_num = 0
args.batch_size = BATCH_SIZE; args.num_workers = 2
if not hasattr(args, 'no_webd_rehearsal'):
    setattr(args, 'no_webd_rehearsal', False)

device = torch.device(DEVICE)
item2id, n_items, train, val, test = load_split_data(args)
index = load_json(f'./dataset/{args.dataset}/{args.dataset}{args.text_index_path}')
train_dataset = CEBNetDataset(args, n_items, train, index, 'train')
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=Collator(args), num_workers=2)

text_embs_list = []
for ttype in args.text_types:
    emb = np.load(f'./dataset/{args.dataset}/{args.dataset}.t5.{ttype}.emb.npy')
    emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(emb)
    text_embs_list.append(emb)
args.text_embedding_size = text_embs_list[0].shape[-1]

model = CEBNet(args, train_dataset, index, device).to(device)
for i, emb in enumerate(text_embs_list):
    model.item_text_embedding[i].weight.data[1:] = torch.tensor(
        emb, dtype=torch.float32, device=device)
model.load_state_dict(state['state_dict'], strict=False)
model.eval()

WM_LEN = args.wm_length   # 20
K      = args.n_prototypes  # 16

# ── Patch SMC forward to capture assign [B, n_long, K] ────────────────────────
captured = {}

def patched_smc(self, x_long, mask=None):
    B, n, d = x_long.shape
    pos_ids = torch.arange(n, device=x_long.device).unsqueeze(0).expand(B, -1)
    pos_emb = self.position_embedding(pos_ids)
    inp = self.replay_ln(self.replay_dropout(x_long + pos_emb))
    if mask is not None:
        attn_mask = mask.float().unsqueeze(1).unsqueeze(2)
        attn_mask = (1.0 - attn_mask) * -1e9
    else:
        attn_mask = torch.zeros(B, 1, 1, n, device=x_long.device)
    x_replayed = self.replay_encoder(inp, inp, attn_mask)[-1]
    item_proj  = self.proj_item(x_replayed)
    proto_proj = self.proj_proto(self.prototypes)
    sim    = torch.matmul(item_proj, proto_proj.T) / math.sqrt(d)
    assign = F.softmax(sim / self.temperature, dim=-1)   # [B, n, K]
    if mask is not None:
        assign = assign * mask.float().unsqueeze(-1)
    captured['assign'] = assign.detach().cpu().numpy()   # [B, n, K]
    captured['mask']   = mask.cpu().numpy() if mask is not None else None
    assign_T = assign.permute(0, 2, 1)
    memory   = torch.bmm(assign_T, x_replayed)
    assign_sum = assign_T.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return memory / assign_sum

model.smc.forward = types.MethodType(patched_smc, model.smc)

# Get prototype embeddings [K, d]
proto_emb = model.smc.prototypes.detach().cpu().numpy()  # [K, d]
print(f'Prototype embeddings: {proto_emb.shape}')

# ── Collect per-user prototype profiles ───────────────────────────────────────
all_proto_profiles = []   # [K] per user (mean assign over long-term items)
all_item_seqs      = []   # item ids for genre heatmap

with torch.no_grad():
    for i, batch in enumerate(tqdm(loader, total=N_BATCHES)):
        if i >= N_BATCHES:
            break
        item_inters = batch['item_inters'].to(device)
        inter_lens  = batch['inter_lens'].to(device)
        code_seq    = model.index[item_inters].reshape(
            item_inters.shape[0] * item_inters.shape[1], -1)
        model.forward(item_inters, inter_lens, code_seq)

        if 'assign' not in captured:
            continue

        asgn    = captured['assign']   # [B, n_long, K]
        msk     = captured['mask']     # [B, n_long] or None
        item_np = item_inters.cpu().numpy()
        lens_np = inter_lens.cpu().numpy()

        for b in range(asgn.shape[0]):
            L = int(lens_np[b])
            n_long = asgn.shape[1]
            if msk is not None:
                valid = msk[b].astype(bool)   # [n_long]
                a_valid = asgn[b][valid]       # [n_valid, K]
            else:
                a_valid = asgn[b]
            if len(a_valid) == 0:
                continue
            proto_profile = a_valid.mean(0)    # [K]
            all_proto_profiles.append(proto_profile)
            all_item_seqs.append(item_np[b, :L])

proto_profiles = np.array(all_proto_profiles)   # [N, K]
print(f'Users collected: {proto_profiles.shape}')

# ── Position = Σ_k (profile[k] * prototype_emb[k]) ────────────────────────────
user_pos = proto_profiles @ proto_emb   # [N, d]
print(f'User position vectors: {user_pos.shape}')

# ── Color = argmax(proto_profile) ─────────────────────────────────────────────
dom_proto = np.argmax(proto_profiles, axis=1)   # [N]

# ── t-SNE ─────────────────────────────────────────────────────────────────────
pca50 = PCA(n_components=min(50, user_pos.shape[1]), random_state=42)
user_pca = pca50.fit_transform(user_pos)
print(f'Running t-SNE on {len(user_pos)} users...')
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42,
            learning_rate='auto', init='pca')
xy = tsne.fit_transform(user_pca)
print('t-SNE done.')

# ── Active prototypes only ────────────────────────────────────────────────────
proto_counts = Counter(dom_proto.tolist())
print('Proto counts:', dict(sorted(proto_counts.items())))
active_protos = [k for k, c in proto_counts.items() if c > 0]
active_protos = sorted(active_protos)
print(f'Active prototypes: {active_protos}')

distinct_colors = ['#E07820', '#3A7EC8', '#2CA02C', '#D62728',
                   '#9467BD', '#8C564B', '#E377C2', '#7F7F7F']
proto_color = {k: distinct_colors[i % len(distinct_colors)]
               for i, k in enumerate(active_protos)}

# Balance: cap large protos so all are visible
cap = min(proto_counts[active_protos[-1]] * 4,
          max(proto_counts.values()))  # max 4x smallest active
cap = max(cap, 500)
rng = np.random.default_rng(42)
idx_vis = []
for k in active_protos:
    idx_k = np.where(dom_proto == k)[0]
    if len(idx_k) > cap:
        idx_k = rng.choice(idx_k, cap, replace=False)
    idx_vis.extend(idx_k.tolist())
idx_vis = np.array(idx_vis)
print(f'Visualizing {len(idx_vis)} points (cap={cap} per proto)')

# ── User-Prototype heatmap ─────────────────────────────────────────────────────
N_PER_PROTO = 1
rng2 = np.random.default_rng(0)
heat_rows  = []
heat_labels = []

for k in active_protos:
    idx_k = np.where(dom_proto == k)[0]
    if len(idx_k) == 0:
        continue
    # pick user near 40th percentile of dominant-prototype weight (not too extreme)
    weights_k = proto_profiles[idx_k, k]
    p40 = np.percentile(weights_k, 40)
    near_p40 = idx_k[np.argsort(np.abs(weights_k - p40))]
    chosen = near_p40[:N_PER_PROTO]
    heat_rows.append(proto_profiles[chosen])
    heat_labels.append((k, len(chosen)))

heat_data = np.concatenate(heat_rows, axis=0)   # [N_users, K]
heat_data = heat_data[:, active_protos]          # only active cols
# 打乱行顺序，消除人为对角线
shuffle_idx = rng2.permutation(len(heat_data))
heat_data = heat_data[shuffle_idx]
heat_labels_shuffled = None   # y轴不再按组标注

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.8))
gs  = gridspec.GridSpec(1, 2, wspace=0.32, width_ratios=[1.05, 1])

# LEFT: t-SNE — plot subsampled idx_vis
ax = fig.add_subplot(gs[0])
xy_vis = xy[idx_vis]
dom_vis = dom_proto[idx_vis]
for k in active_protos:
    mask = dom_vis == k
    if mask.sum() == 0:
        continue
    ax.scatter(xy_vis[mask, 0], xy_vis[mask, 1],
               s=10, color=proto_color[k], alpha=0.60, zorder=3,
               label=f'P{k} (n={proto_counts[k]})')
for k in active_protos:
    mask = dom_proto == k
    if mask.sum() < 5:
        continue
    cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
    ax.text(cx, cy, f'P{k}', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.28', facecolor=proto_color[k],
                      edgecolor='white', linewidth=1.0, alpha=0.92), zorder=5)

ax.set_title('User Preference Space (t-SNE)', fontsize=9.5, pad=6)
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# RIGHT: User-Prototype heatmap  (Blues, square-ish cells)
ax2 = fig.add_subplot(gs[1])
im = ax2.imshow(heat_data, cmap='Blues', aspect='auto',
                vmin=0, vmax=heat_data.max())

# x-axis: prototype labels, colored
ax2.set_xticks(range(len(active_protos)))
ax2.set_xticklabels([f'P{k}' for k in active_protos], fontsize=8.5, rotation=45, ha='right')
for tick, k in zip(ax2.get_xticklabels(), active_protos):
    tick.set_color(proto_color[k])
    tick.set_fontweight('bold')

# y-axis: user index
ax2.set_yticks(range(len(heat_data)))
ax2.set_yticklabels([f'U{i+1}' for i in range(len(heat_data))], fontsize=8)

# draw vertical grid lines between prototypes
for x in range(1, len(active_protos)):
    ax2.axvline(x - 0.5, color='white', lw=0.8, alpha=0.6)

# annotate cell values
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        val = heat_data[i, j]
        color = 'white' if val > heat_data.max() * 0.6 else '#333333'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=7, color=color)

cb = plt.colorbar(im, ax=ax2, shrink=0.75, pad=0.02, aspect=20)
cb.set_label('Assign Weight', fontsize=9)
cb.ax.tick_params(labelsize=8)
ax2.set_xlabel('Prototype Index', fontsize=9.5)
ax2.set_ylabel('Sampled Users', fontsize=9)
ax2.set_title('User–Prototype Assignment Pattern\n(1 user per prototype, shuffled order)', fontsize=9.5, pad=6)
ax2.spines[:].set_visible(False)

fig.suptitle('Case Study 3: SMC Learns Semantically Diverse Prototype Memory Slots',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig('./figures/case3_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case3_final.png', bbox_inches='tight', dpi=300)
print('Saved case3_final.png/.pdf')
