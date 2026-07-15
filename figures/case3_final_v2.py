"""
Case 3 (final v2): SMC Semantic Prototype Diversity — ML-1M_TedRec
Left:  t-SNE / PCA of 16 prototype embeddings, sized by user count
       Position = prototype embedding space (d-dim), showing they are spread apart
Right: User-prototype retrieval heatmap (48 sampled users × 16 prototypes)
       Rows grouped and sorted by dominant prototype → block-diagonal pattern
"""
import numpy as np
import pickle, json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.linewidth': 1.0})

# ── Load ──────────────────────────────────────────────────────────────────────
data      = np.load('./figures/case_data_ML1M.npz')
beta      = data['beta']            # [N, 36]
proto_emb = data['prototypes']      # [K, d]
proto_w   = beta[:, 20:]            # [N, 16]  prototype portion of β
dom       = np.argmax(proto_w, axis=1)  # [N] dominant prototype per user

K = 16
counts   = Counter(dom.tolist())
top8     = [k for k, _ in counts.most_common(8)]
base_colors = plt.get_cmap('tab10').colors
proto_color = {k: base_colors[i % 10] for i, k in enumerate(top8)}
gray = '#AAAAAA'

# ── PCA of prototype embeddings (K=16 points) ─────────────────────────────────
# 16 points is too few for t-SNE; use PCA 2D directly
pca2 = PCA(n_components=2, random_state=42)
proto_xy = pca2.fit_transform(proto_emb)   # [16, 2]
var_ratio = pca2.explained_variance_ratio_

# Normalize user count → bubble size (area)
user_counts_all = np.array([counts.get(k, 0) for k in range(K)])
size_max = 600
size_min = 40
nc_min, nc_max = user_counts_all.min(), user_counts_all.max()
bubble_size = size_min + (user_counts_all - nc_min) / (nc_max - nc_min + 1) * (size_max - size_min)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 5.4))
gs  = gridspec.GridSpec(1, 2, wspace=0.36, width_ratios=[1.05, 1])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: Prototype embedding space (PCA)
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])

for k in range(K):
    color = proto_color.get(k, gray)
    alpha = 0.90 if k in top8 else 0.50
    ax.scatter(proto_xy[k, 0], proto_xy[k, 1],
               s=bubble_size[k], color=color, alpha=alpha,
               edgecolors='white', linewidth=1.0, zorder=3)
    # Label
    ax.text(proto_xy[k, 0], proto_xy[k, 1], f'P{k}',
            ha='center', va='center', fontsize=7.5,
            fontweight='bold' if k in top8 else 'normal',
            color='white' if k in top8 else '#555',
            zorder=5)

# Connection lines to nearest neighbors (optional: skip, too cluttered)

# Colorbar legend: user count → bubble size
for n_show, lbl in [(100, '100 users'), (500, '500'), (2000, '2000')]:
    s = size_min + (n_show - nc_min) / (nc_max - nc_min + 1) * (size_max - size_min)
    ax.scatter([], [], s=s, c='#888888', alpha=0.7,
               edgecolors='white', linewidth=0.8, label=lbl)
ax.legend(title='User count', fontsize=8, title_fontsize=8,
          loc='lower right', framealpha=0.85,
          handletextpad=0.3, labelspacing=0.6)

ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var)', fontsize=9)
ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var)', fontsize=9)
ax.set_title('(A) Prototype Embedding Space (PCA)\n'
             'Each bubble = one prototype, size ∝ #users dominated by it',
             fontsize=9.5, pad=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.18)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: User × Prototype heatmap (block-diagonal pattern)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])

# Sample users: 6 per top-8 prototype → 48 rows
np.random.seed(42)
sel_users, sel_group = [], []
for k in top8:
    idxs = np.where(dom == k)[0]
    chosen = idxs[np.random.choice(len(idxs), min(6, len(idxs)), replace=False)]
    # Sort within group by their dominant proto weight (descending)
    order = np.argsort(proto_w[chosen, k])[::-1]
    sel_users.extend(chosen[order].tolist())
    sel_group.extend([k] * len(chosen))

heat = proto_w[sel_users]   # [48, 16]  full 16 columns
n_sel = len(sel_users)

im = ax2.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=heat.max())

# White dividers between groups
group_size = 6
for gi in range(len(top8) - 1):
    ax2.axhline((gi + 1) * group_size - 0.5, color='white', lw=1.8)

# Row tick labels
group_mids = [(gi * group_size + (group_size - 1) / 2) for gi in range(len(top8))]
ax2.set_yticks(group_mids)
ax2.set_yticklabels([f'P{k} users' for k in top8], fontsize=8.5)
for tick, k in zip(ax2.get_yticklabels(), top8):
    tick.set_color(proto_color[k])
    tick.set_fontweight('bold')

# Column tick labels (all 16 prototypes)
ax2.set_xticks(range(K))
ax2.set_xticklabels([f'P{k}' for k in range(K)], fontsize=7.5, rotation=60, ha='right')
# Highlight top-8 columns
for tick in ax2.get_xticklabels():
    k = int(tick.get_text()[1:])
    if k in top8:
        tick.set_color(proto_color[k])
        tick.set_fontweight('bold')

# Highlight diagonal blocks
for gi, k in enumerate(top8):
    col = k  # column index = prototype index
    ax2.add_patch(plt.Rectangle((col - 0.5, gi * group_size - 0.5),
                                  1, group_size,
                                  fill=False, edgecolor=proto_color[k],
                                  linewidth=1.8, zorder=4))

cb = plt.colorbar(im, ax=ax2, shrink=0.65, pad=0.03)
cb.set_label('Retrieval weight β', fontsize=8.5)
cb.ax.tick_params(labelsize=8)

ax2.set_xlabel('Prototype Index', fontsize=9)
ax2.set_title('(B) User–Prototype Retrieval Pattern\n'
              '(6 sampled users per dominant-prototype group)',
              fontsize=9.5, pad=6)

fig.suptitle('Case Study 3: SMC Captures Diverse User Long-Term Preference Profiles',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig('./figures/case3_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case3_final.png', bbox_inches='tight', dpi=300)
print('Saved case3_final.png/.pdf')
print(f'Proto PCA var: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%')
print(f'Top-8 prototype user counts: {[(k, counts[k]) for k in top8]}')
