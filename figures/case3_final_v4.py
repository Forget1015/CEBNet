"""
Case 3 (final v4): Left = user t-SNE scatter using TEXT EMBEDDING as position,
colored by dominant prototype. Non-circular. Matches style of case3_prototypes.png.
Right = User-prototype β heatmap with block-diagonal pattern.
"""
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.linewidth': 1.0})

# ── Load ──────────────────────────────────────────────────────────────────────
data    = np.load('./figures/case_data_ML1M.npz')
beta    = data['beta']
proto_w = beta[:, 20:]                           # [N, 16]
dom     = np.argmax(proto_w, axis=1)             # [N] dominant prototype

xy  = np.load('./figures/ml1m_user_text_tsne_xy.npy')   # [N, 2]  text-based

K = 16
counts = Counter(dom.tolist())
top8   = [k for k, _ in counts.most_common(8)]
base_colors = plt.get_cmap('tab10').colors
proto_color = {k: base_colors[i % 10] for i, k in enumerate(top8)}

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 5.4))
gs  = gridspec.GridSpec(1, 2, wspace=0.34, width_ratios=[1.3, 1])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: t-SNE scatter (same style as case3_prototypes.png)
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])

# Background: non-top-8 prototypes in gray
other_mask = ~np.isin(dom, top8)
if other_mask.sum() > 0:
    ax.scatter(xy[other_mask, 0], xy[other_mask, 1],
               s=5, c='#CCCCCC', alpha=0.3, zorder=2)

# Top-8 with distinct colors
for k in top8:
    mask = dom == k
    ax.scatter(xy[mask, 0], xy[mask, 1],
               s=8, color=proto_color[k], alpha=0.55,
               zorder=3, label=f'P{k} (n={counts[k]})')

# Cluster centroid labels
for k in top8:
    mask = dom == k
    cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
    ax.text(cx, cy, f'P{k}',
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.32', facecolor=proto_color[k],
                      edgecolor='white', linewidth=1.0, alpha=0.90),
            zorder=5)

ax.set_xlabel('t-SNE Dim 1', fontsize=9)
ax.set_ylabel('t-SNE Dim 2', fontsize=9)
ax.set_title('(A) User Preference Space (t-SNE)\n'
             'Position = mean text embedding of history; Color = dominant prototype',
             fontsize=9.5, pad=6)
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [mpatches.Patch(color=proto_color[k], alpha=0.85,
                                  label=f'P{k}  (n={counts[k]})')
                  for k in top8]
ax.legend(handles=legend_handles, fontsize=8, loc='lower right',
          framealpha=0.85, ncol=2, columnspacing=0.8,
          handlelength=1.0, handletextpad=0.4)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: User-prototype β heatmap (block-diagonal)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])

np.random.seed(42)
sel_users = []
for k in top8:
    idxs = np.where(dom == k)[0]
    chosen = idxs[np.random.choice(len(idxs), min(6, len(idxs)), replace=False)]
    order = np.argsort(proto_w[chosen, k])[::-1]
    sel_users.extend(chosen[order].tolist())

heat = proto_w[np.ix_(sel_users, top8)]   # [48, 8]

im = ax2.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=heat.max())

group_size = 6
for gi in range(len(top8) - 1):
    ax2.axhline((gi + 1) * group_size - 0.5, color='white', lw=1.8)

group_mids = [(gi * group_size + (group_size - 1) / 2) for gi in range(len(top8))]
ax2.set_yticks(group_mids)
ax2.set_yticklabels([f'P{k} users' for k in top8], fontsize=8.5)
for tick, k in zip(ax2.get_yticklabels(), top8):
    tick.set_color(proto_color[k]); tick.set_fontweight('bold')

ax2.set_xticks(range(len(top8)))
ax2.set_xticklabels([f'P{k}' for k in top8], fontsize=9, rotation=45, ha='right')
for tick, k in zip(ax2.get_xticklabels(), top8):
    tick.set_color(proto_color[k]); tick.set_fontweight('bold')

for gi in range(len(top8)):
    ax2.add_patch(plt.Rectangle((gi - 0.5, gi * group_size - 0.5),
                                  1, group_size,
                                  fill=False, edgecolor=proto_color[top8[gi]],
                                  linewidth=2.0, zorder=4))

cb = plt.colorbar(im, ax=ax2, shrink=0.65, pad=0.03)
cb.set_label('Retrieval weight β', fontsize=8.5)
cb.ax.tick_params(labelsize=8)

ax2.set_xlabel('Prototype Index (top 8 by user count)', fontsize=9)
ax2.set_title('(B) User–Prototype Retrieval Pattern\n'
              '(6 sampled users per dominant-prototype group)',
              fontsize=9.5, pad=6)

fig.suptitle('Case Study 3: SMC Captures Diverse User Long-Term Preference Profiles',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig('./figures/case3_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case3_final.png', bbox_inches='tight', dpi=300)
print('Saved.')
