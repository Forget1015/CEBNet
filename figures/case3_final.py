"""
Case 3 (final v2): SMC Semantic Prototype Diversity
Left:  t-SNE of item TEXT embeddings, colored by dominant prototype assignment
       Position = semantic space (non-circular), Color = model output
Right: Prototype × Genre heatmap showing each prototype's semantic focus
"""
import numpy as np
import pickle, json
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 10,
                     'axes.linewidth': 1.0})

# ── Load ──────────────────────────────────────────────────────────────────────
data    = np.load('./figures/case_data_ML1M.npz')
beta    = data['beta']            # [N, 36]
targets = data['targets']         # [N]
proto_w = beta[:, 20:]            # [N, 16]  prototype portion

with open('./figures/case_data_ML1M_var.pkl', 'rb') as f:
    v = pickle.load(f)
seqs   = v['seqs']    # list of [L] item ids
assign = v['assign']  # list of [n, K]  (n = long-term item count)

with open('./dataset/ML-1M_TedRec/ML-1M_TedRec.meta.json') as f:
    meta = json.load(f)

# Load item text embeddings (title)
title_emb = np.load('./dataset/ML-1M_TedRec/ML-1M_TedRec.t5.title.emb.npy')  # [n_items+1, 1024]

# ── Build item → prototype mapping ────────────────────────────────────────────
# For each (user, item) pair, record assign[item_pos] → K weights
# Item index in seqs is 1-based (0 = padding), remap to emb index
K = 16
item_proto_accum = defaultdict(lambda: np.zeros(K))
item_count = defaultdict(int)

for i, (seq, asgn) in enumerate(zip(seqs, assign)):
    # seq: [L] item ids (1-indexed), asgn: [n, K] where n <= L (long-term part)
    n = asgn.shape[0]
    # asgn corresponds to the LAST n items in seq (long-term history fed to SMC)
    # seq[-n:] are the long-term items
    if n > len(seq):
        n = len(seq)
    items_long = seq[-n:]
    for j, iid in enumerate(items_long):
        item_proto_accum[int(iid)] += asgn[j]
        item_count[int(iid)] += 1

# Normalize: average proto weight per item
item_ids = np.array(sorted(item_proto_accum.keys()))
item_proto_mat = np.array([item_proto_accum[iid] / max(item_count[iid], 1)
                            for iid in item_ids])  # [n_unique_items, 16]
item_dom_proto = np.argmax(item_proto_mat, axis=1)  # dominant prototype per item

# Filter to items that appear ≥ 3 times for stability
stable_mask = np.array([item_count[iid] >= 3 for iid in item_ids])
item_ids_stable = item_ids[stable_mask]
item_dom_stable = item_dom_proto[stable_mask]
print(f'Unique items: {len(item_ids)}, stable (≥3): {stable_mask.sum()}')

# ── t-SNE on item text embeddings ─────────────────────────────────────────────
# Get text embeddings for stable items (item_id is 1-indexed)
embs_stable = title_emb[item_ids_stable]  # [n_stable, 1024]

# PCA first for speed
pca = PCA(n_components=50, random_state=42)
embs_pca = pca.fit_transform(embs_stable)

print(f'Running t-SNE on {len(embs_stable)} items...')
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42,
            learning_rate='auto', init='pca')
xy = tsne.fit_transform(embs_pca)
print('t-SNE done.')

# ── Pick top-8 prototypes by item count ───────────────────────────────────────
proto_item_counts = Counter(item_dom_stable.tolist())
top8 = [k for k, _ in proto_item_counts.most_common(8)]

base_colors = plt.get_cmap('tab10').colors
proto_color = {k: base_colors[i % 10] for i, k in enumerate(top8)}

# ── Build prototype × genre heatmap ───────────────────────────────────────────
# Genres in ML-1M
all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']

# Accumulate genre counts per prototype (weighted by assign)
proto_genre_mat = np.zeros((K, len(all_genres)))
for i, (seq, asgn) in enumerate(zip(seqs, assign)):
    n = asgn.shape[0]
    if n > len(seq): n = len(seq)
    items_long = seq[-n:]
    for j, iid in enumerate(items_long):
        g = meta.get(str(int(iid)), {}).get('genres', '')
        for genre in g.split():
            if genre in all_genres:
                gi = all_genres.index(genre)
                proto_genre_mat[:, gi] += asgn[j]  # [K] += softmax weights

# Normalize per prototype (row)
row_sums = proto_genre_mat.sum(axis=1, keepdims=True).clip(min=1)
proto_genre_norm = proto_genre_mat / row_sums  # [K, n_genres]

# Select top-8 protos and top-12 most discriminative genres
# Discriminativeness: variance across prototypes (top8 only)
genre_var = proto_genre_norm[top8].var(axis=0)
top_genres_idx = np.argsort(genre_var)[-12:][::-1]
top_genres = [all_genres[gi] for gi in top_genres_idx]
heat_data = proto_genre_norm[np.ix_(top8, top_genres_idx)]  # [8, 12]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 2, wspace=0.38, width_ratios=[1.3, 1])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: t-SNE scatter (position=text semantic, color=prototype)
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])

# Background: non-top prototypes
other_mask = ~np.isin(item_dom_stable, top8)
if other_mask.sum() > 0:
    ax.scatter(xy[other_mask, 0], xy[other_mask, 1],
               s=4, c='#CCCCCC', alpha=0.25, zorder=1)

for k in top8:
    mask = item_dom_stable == k
    ax.scatter(xy[mask, 0], xy[mask, 1],
               s=7, color=proto_color[k], alpha=0.55, zorder=3,
               label=f'P{k} (n={proto_item_counts[k]})')

# Cluster centroid labels
for k in top8:
    mask = item_dom_stable == k
    cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
    ax.text(cx, cy, f'P{k}', ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.30', facecolor=proto_color[k],
                      edgecolor='white', linewidth=1.0, alpha=0.90), zorder=5)

ax.set_xlabel('', fontsize=9)
ax.set_ylabel('', fontsize=9)
ax.set_title('Item Semantic Space (t-SNE)',
             fontsize=9.5, pad=6)
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Prototype × Genre heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])

im = ax2.imshow(heat_data, cmap='YlOrRd', aspect='auto',
                vmin=0, vmax=heat_data.max())

ax2.set_xticks(range(len(top_genres)))
ax2.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=8.5)

ax2.set_yticks(range(len(top8)))
ax2.set_yticklabels([f'P{k}' for k in top8], fontsize=9)
for tick, k in zip(ax2.get_yticklabels(), top8):
    tick.set_color(proto_color[k])
    tick.set_fontweight('bold')

cb = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.03)
cb.set_label('Avg. assign weight', fontsize=8.5)
cb.ax.tick_params(labelsize=8)

ax2.set_xlabel('Movie Genre', fontsize=9)
ax2.set_title('Prototype–Genre Distribution',
              fontsize=9.5, pad=6)

fig.suptitle('Case Study 3: SMC Learns Semantically Diverse Prototype Memory Slots',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig('./figures/case3_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case3_final.png', bbox_inches='tight', dpi=300)
print('Saved case3_final.png/.pdf')
