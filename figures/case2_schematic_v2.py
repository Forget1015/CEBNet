"""
Case 2 schematic v2: Time-domain view of WEBD burst preservation.
Shows raw interest-shift magnitude over time, threshold, and what WEBD keeps vs suppresses.
Run from: /data0/yejinxuan/workspace/CEBNet
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.linewidth': 1.0})

np.random.seed(42)

def make_seq(L, bursts, noise_scale=0.08):
    """bursts: list of (position, magnitude)"""
    seq = np.abs(np.random.normal(0.12, noise_scale, L))
    seq = np.clip(seq, 0.02, 0.25)
    for pos, mag in bursts:
        seq[pos] = mag
    return seq

L = 18

# User A: one sudden burst in the middle (gift purchase)
rawA = make_seq(L, [(9, 0.88), (10, 0.62)], noise_scale=0.05)
thrA = 0.32
burstA = {9: 'Gift\npurchase', 10: ''}

# User B: two periodic bursts (monthly repurchase)
rawB = make_seq(L, [(4, 0.79), (13, 0.83)], noise_scale=0.06)
thrB = 0.30
burstB = {4: 'Repurchase\ncycle 1', 13: 'Repurchase\ncycle 2'}

# User C: recent sustained new interest (last 4 steps are all high)
rawC = make_seq(L, [(13, 0.58), (14, 0.72), (15, 0.69), (16, 0.85), (17, 0.77)], noise_scale=0.05)
thrC = 0.33
burstC = {15: 'New interest\nemerging →'}

users = [
    (rawA, thrA, burstA, 'User A', 'Occasional Burst'),
    (rawB, thrB, burstB, 'User B', 'Periodic Bursts'),
    (rawC, thrC, burstC, 'User C', 'Recent Interest Shift'),
]

C_RAW   = '#AAAAAA'
C_BURST = '#E07B35'
C_NOISE = '#3A7EC8'
C_THR   = '#CC2222'

fig = plt.figure(figsize=(13.5, 4.4))
gs  = gridspec.GridSpec(1, 3, wspace=0.40)

for col, (raw, thr, burst_annot, ulbl, subtitle) in enumerate(users):
    ax = fig.add_subplot(gs[col])
    t  = np.arange(L)

    burst_mask = raw >= thr
    noise_mask = ~burst_mask

    # After WEBD: burst kept, noise flattened to near-zero
    after = raw.copy()
    after[noise_mask] = raw[noise_mask] * 0.05   # suppressed to ~0

    # Draw raw as gray background line + shading
    ax.fill_between(t, 0, raw, color=C_RAW, alpha=0.30, zorder=1)
    ax.plot(t, raw, color='#999999', lw=1.4, alpha=0.7, zorder=2, label='Before WEBD (raw)')

    # Draw after as colored line
    ax.plot(t, after, color='#222222', lw=2.0, alpha=0.9, zorder=4, label='After WEBD (denoised)')

    # Shade burst regions (orange)
    for i in t:
        if burst_mask[i]:
            ax.fill_between([i - 0.5, i + 0.5], 0, raw[i],
                            color=C_BURST, alpha=0.55, zorder=3)
            ax.fill_between([i - 0.5, i + 0.5], 0, after[i],
                            color=C_BURST, alpha=0.88, zorder=4)

    # Shade noise regions (blue, very small)
    for i in t:
        if noise_mask[i]:
            ax.fill_between([i - 0.5, i + 0.5], 0, after[i],
                            color=C_NOISE, alpha=0.55, zorder=3)

    # Threshold line
    ax.axhline(thr, color=C_THR, lw=1.6, ls='--', alpha=0.88, zorder=5)
    ax.text(L - 0.3, thr + 0.025, f'θ = {thr:.2f}',
            color=C_THR, fontsize=8.2, ha='right', fontweight='bold')

    # Burst annotations
    for pos, txt in burst_annot.items():
        if txt == '': continue
        yval = raw[pos]
        ax.annotate(txt,
            xy=(pos, yval), xytext=(pos, yval + 0.14),
            ha='center', va='bottom', fontsize=7.5, color='#7A3800', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#7A3800', lw=1.1))

    # Stats box
    n_burst = burst_mask.sum()
    n_noise = noise_mask.sum()
    ax.text(0.03, 0.97,
            f'Preserved: {n_burst} steps\nSuppressed: {n_noise} steps',
            transform=ax.transAxes, ha='left', va='top', fontsize=8.0,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#F8F8F8',
                      edgecolor='#CCCCCC', linewidth=0.7))

    ax.set_xlim(-0.7, L - 0.3)
    ax.set_ylim(-0.02, 1.25)
    ax.set_xticks(np.arange(0, L, 3))
    ax.set_xticklabels([f't={i+1}' for i in range(0, L, 3)], fontsize=8.0)
    ax.set_xlabel('Interaction Time Step', fontsize=9)
    if col == 0:
        ax.set_ylabel(r'Interest-shift Magnitude  $\|\mathbf{e}_{t+1}-\mathbf{e}_t\|_2$', fontsize=8.5)
    ax.set_title(f'{ulbl} — {subtitle}', fontsize=10, pad=5)
    ax.grid(axis='y', alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend
raw_line  = plt.Line2D([0],[0], color='#999999', lw=1.4, alpha=0.8, label='Before WEBD (raw sequence)')
aft_line  = plt.Line2D([0],[0], color='#222222', lw=2.0, label='After WEBD (denoised)')
bst_patch = mpatches.Patch(color=C_BURST, alpha=0.75, label='Burst interest (preserved, |Δe| ≥ θ)')
noi_patch = mpatches.Patch(color=C_NOISE, alpha=0.55, label='Noise (suppressed, |Δe| < θ → ≈0)')
thr_line  = plt.Line2D([0],[0], color=C_THR, lw=1.6, ls='--', label='Adaptive threshold θ')

fig.legend(handles=[raw_line, aft_line, bst_patch, noi_patch, thr_line],
           loc='lower center', bbox_to_anchor=(0.5, -0.10),
           ncol=5, fontsize=8.8, frameon=False,
           handlelength=1.5, handletextpad=0.5, columnspacing=1.5)

fig.suptitle(
    'Case Study 2: WEBD Selectively Preserves Burst Interests While Suppressing Routine Noise',
    fontsize=11, fontweight='bold', y=1.03)

plt.savefig('./figures/case2_schematic_v2.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./figures/case2_schematic_v2.png', bbox_inches='tight', dpi=300)
print('Saved ./figures/case2_schematic_v2.png')
