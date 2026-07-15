"""
Case 1: DEBR retrieval weight β visualization.
Three user panels: semantic-dominant, typical, working-memory-anchored.

Run from: /data0/yejinxuan/workspace/CEBNet
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.0,
})

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load('./figures/case_data.npz')
beta = data['beta']   # [N, 26]

wm_mass    = beta[:, :10].sum(1)
proto_mass = beta[:, 10:].sum(1)

# Three representative users
iA = np.argmax(proto_mass)
iC = np.argmin(proto_mass)
iB = np.argmin(np.abs(proto_mass - proto_mass.mean()))

idx_list = [iA, iB, iC]
user_labels = ['User A', 'User B', 'User C']
subtitles = [
    f'Semantic-Dominant\n(proto={proto_mass[iA]:.2f}, wm={wm_mass[iA]:.2f})',
    f'Typical\n(proto={proto_mass[iB]:.2f}, wm={wm_mass[iB]:.2f})',
    f'Working-Memory-Anchored\n(proto={proto_mass[iC]:.2f}, wm={wm_mass[iC]:.2f})',
]

C_WM    = '#3A7EC8'
C_PROTO = '#E07B35'
ALPHA   = 0.88

fig = plt.figure(figsize=(12.5, 3.6))
gs  = gridspec.GridSpec(1, 3, wspace=0.40)

x = np.arange(26)
wm_idx    = x < 10
proto_idx = x >= 10

for col, (idx, ulbl, sub) in enumerate(zip(idx_list, user_labels, subtitles)):
    ax = fig.add_subplot(gs[col])
    bvals = beta[idx]
    ymax = bvals.max()

    ax.bar(x[wm_idx],    bvals[wm_idx],    color=C_WM,    alpha=ALPHA, width=0.75,
           edgecolor='white', linewidth=0.4, zorder=3)
    ax.bar(x[proto_idx], bvals[proto_idx], color=C_PROTO, alpha=ALPHA, width=0.75,
           edgecolor='white', linewidth=0.4, zorder=3)

    ax.axvline(x=9.5, color='#888', lw=1.0, ls='--', alpha=0.6, zorder=2)

    # Region labels above the bars area at top
    ax.text(4.5,  ymax * 1.08, 'Working Memory\n(10 slots)',
            ha='center', va='bottom', fontsize=7.5, color=C_WM, fontweight='bold')
    ax.text(17.5, ymax * 1.08, 'Semantic Prototypes\n(16 slots)',
            ha='center', va='bottom', fontsize=7.5, color=C_PROTO, fontweight='bold')

    ax.set_xlim(-0.8, 25.8)
    ax.set_ylim(0, ymax * 1.45)
    ax.set_xticks([])
    ax.set_xlabel('Memory Position (1–10: WM, 11–26: Prototypes)', fontsize=8.0)
    if col == 0:
        ax.set_ylabel('Retrieval Weight β', fontsize=9)
    ax.set_title(f'{ulbl}\n{sub}', fontsize=9.0, pad=4)
    ax.grid(axis='y', alpha=0.22, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

patches = [
    mpatches.Patch(color=C_WM,    alpha=ALPHA, label='Working Memory slots'),
    mpatches.Patch(color=C_PROTO, alpha=ALPHA, label='Semantic Prototype slots'),
]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.09),
           ncol=2, fontsize=9.5, frameon=False,
           handlelength=1.4, handletextpad=0.5, columnspacing=2.0)

fig.suptitle('Case Study 1: DEBR Retrieval Weights β across User Types',
             fontsize=11, fontweight='bold', y=1.03)

plt.savefig('./figures/case1_retrieval.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case1_retrieval.png', bbox_inches='tight', dpi=300)
print('Case 1 saved.')
