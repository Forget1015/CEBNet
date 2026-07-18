"""
生成小波分解的低频（平缓）和高频（锯齿）波形
输出矢量图（PDF + SVG）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 设置样式
plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'font.size': 11,
    'axes.linewidth': 1.5,
})

def generate_smooth_wave(x, amplitude=0.8, frequency=1.5):
    """生成平缓的低频波（类似余弦波）"""
    return amplitude * np.cos(2 * np.pi * frequency * x)

def generate_zigzag_wave(x, amplitude=0.5, frequency=6):
    """生成锯齿高频波（快速振荡）"""
    # 使用正弦波 + 高频成分
    base = amplitude * np.sin(2 * np.pi * frequency * x)
    noise = 0.15 * np.sin(2 * np.pi * frequency * 3 * x)
    return base + noise

def plot_waveforms():
    """绘制波形对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # 生成 x 轴
    x = np.linspace(0, 1, 200)

    # 1. 原始信号（低频 + 高频）
    low_freq = generate_smooth_wave(x, amplitude=0.8, frequency=1.5)
    high_freq = generate_zigzag_wave(x, amplitude=0.3, frequency=8)
    original = low_freq + high_freq

    ax = axes[0]
    ax.plot(x, original, 'k-', linewidth=2, label='Original Signal')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_title('(a) Original Behavioral Sequence', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

    # 2. 低频分量（cA - Approximation）
    ax = axes[1]
    ax.plot(x, low_freq, color='#2E86AB', linewidth=2.5, label='Low Frequency (cA)')
    ax.fill_between(x, low_freq, alpha=0.2, color='#2E86AB')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_title('(b) Low Frequency (cA) - Long-term Trend', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

    # 添加标注
    ax.annotate('Smooth trend\n(stable interest)',
                xy=(0.3, 0.6), fontsize=10, color='#2E86AB',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', alpha=0.8))

    # 3. 高频分量（cD - Detail）
    ax = axes[2]
    ax.plot(x, high_freq, color='#E63946', linewidth=2, label='High Frequency (cD)')
    ax.fill_between(x, high_freq, alpha=0.2, color='#E63946')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_xlabel('Time / Position', fontweight='bold')
    ax.set_title('(c) High Frequency (cD) - Noisy Fluctuation', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

    # 添加标注
    ax.annotate('Fast oscillation\n(noise + burst)',
                xy=(0.7, 0.4), fontsize=10, color='#E63946',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#E63946', alpha=0.8))

    plt.tight_layout()

    # 保存
    plt.savefig('./figures/wavelet_waveforms_decomposition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./figures/wavelet_waveforms_decomposition.svg', format='svg', bbox_inches='tight')
    plt.savefig('./figures/wavelet_waveforms_decomposition.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: wavelet_waveforms_decomposition.pdf/svg/png')

    plt.close()

def plot_standalone_waves():
    """生成独立的波形（可直接插入论文）"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))

    x = np.linspace(0, 1, 200)

    # 低频波
    ax = axes[0]
    low_freq = generate_smooth_wave(x, amplitude=1.0, frequency=1.5)
    ax.plot(x, low_freq, color='#2E86AB', linewidth=3)
    ax.set_ylabel('cA (Low-freq)', fontweight='bold', fontsize=12)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis='y')

    # 高频波
    ax = axes[1]
    high_freq = generate_zigzag_wave(x, amplitude=0.8, frequency=8)
    ax.plot(x, high_freq, color='#E63946', linewidth=2.5)
    ax.set_ylabel('cD (High-freq)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time', fontweight='bold', fontsize=12)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()

    # 保存独立版本
    plt.savefig('./figures/wavelet_waves_standalone.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./figures/wavelet_waves_standalone.svg', format='svg', bbox_inches='tight')
    plt.savefig('./figures/wavelet_waves_standalone.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: wavelet_waves_standalone.pdf/svg/png')

    plt.close()

def plot_comparison_side_by_side():
    """并排对比（适合放在论文中）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    x = np.linspace(0, 1, 200)

    # 低频
    low_freq = generate_smooth_wave(x, amplitude=1.0, frequency=1.5)
    ax1.plot(x, low_freq, color='#2E86AB', linewidth=3)
    ax1.fill_between(x, low_freq, alpha=0.15, color='#2E86AB')
    ax1.set_title('Low Frequency (cA)\nLong-term Trend', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontweight='bold')
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.set_ylim(-1.3, 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.3)

    # 高频
    high_freq = generate_zigzag_wave(x, amplitude=0.8, frequency=8)
    ax2.plot(x, high_freq, color='#E63946', linewidth=2.5)
    ax2.fill_between(x, high_freq, alpha=0.15, color='#E63946')
    ax2.set_title('High Frequency (cD)\nNoisy Fluctuation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontweight='bold')
    ax2.set_xlabel('Time', fontweight='bold')
    ax2.set_ylim(-1.3, 1.3)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig('./figures/wavelet_waves_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./figures/wavelet_waves_comparison.svg', format='svg', bbox_inches='tight')
    plt.savefig('./figures/wavelet_waves_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: wavelet_waves_comparison.pdf/svg/png')

    plt.close()

if __name__ == '__main__':
    print('Generating wavelet waveform visualizations...\n')

    # 生成三种版本
    plot_waveforms()              # 完整分解图（三子图）
    plot_standalone_waves()       # 独立波形（两子图）
    plot_comparison_side_by_side() # 并排对比

    print('\nDone! Generated 3 versions × 3 formats = 9 files')
    print('  1. wavelet_waveforms_decomposition.* - 完整分解（原始+低频+高频）')
    print('  2. wavelet_waves_standalone.*        - 独立波形（低频+高频）')
    print('  3. wavelet_waves_comparison.*        - 并排对比（最适合论文）')
