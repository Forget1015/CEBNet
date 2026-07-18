"""
Figure for Challenge 2: 长短期兴趣冲突统计
统计用户的长期主导兴趣与近期行为的一致性
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load_ml1m_data():
    """加载 ML-1M 数据（使用 genres 信息）"""
    data_path = './dataset/ML-1M_TedRec/'

    # 读取 meta.json 获取 genres
    with open(f'{data_path}ML-1M_TedRec.meta.json', 'r') as f:
        meta = json.load(f)

    # 构建 item_id -> genres 映射（item_id 是 0-based，meta key 是 1-based）
    item2genres = {}
    emb_map_path = f'{data_path}ML-1M_TedRec.emb_map.json'
    with open(emb_map_path, 'r') as f:
        emb_map = json.load(f)

    for item_str, cebnet_id in emb_map.items():
        meta_entry = meta.get(str(cebnet_id), {})
        genres_str = meta_entry.get('genres', '')
        genres_list = genres_str.split() if genres_str else []
        item2genres[item_str] = genres_list

    # 读取训练集交互记录
    with open(f'{data_path}ML-1M_TedRec.train.jsonl', 'r') as f:
        lines = f.readlines()

    # 对每个用户，取最长的历史序列
    user_sequences = {}
    for line in lines:
        data = json.loads(line)
        user_id = data['user_id']
        item_seq = data['inter_history'] + [data['target_id']]

        if user_id not in user_sequences or len(item_seq) > len(user_sequences[user_id]):
            user_sequences[user_id] = item_seq

    train_data = []
    for user_id, items in user_sequences.items():
        train_data.append({
            'user_id': user_id,
            'item_seq': items
        })

    return item2genres, train_data


def get_dominant_genre(item_ids, item2genres):
    """获取一组物品的主导 genre"""
    genre_counts = Counter()
    for iid in item_ids:
        genres = item2genres.get(str(iid), [])
        for g in genres:
            genre_counts[g] += 1

    if len(genre_counts) == 0:
        return None, []
    top_genres = [g for g, _ in genre_counts.most_common(3)]
    return top_genres[0], top_genres


def compute_conflict_statistics(item2genres, train_data, recent_ratio=0.2):
    """
    计算长短期兴趣冲突统计（基于 genres）

    Args:
        item2genres: 物品到 genres 的映射
        train_data: 训练数据
        recent_ratio: 最近行为的比例

    Returns:
        dict: 各类冲突的用户数量
    """
    conflict_stats = {
        'consistent': 0,        # 完全一致
        'partial_conflict': 0,  # 部分冲突
        'strong_conflict': 0,   # 强冲突
        'insufficient_data': 0, # 数据不足
    }

    user_examples = {
        'consistent': [],
        'partial_conflict': [],
        'strong_conflict': []
    }

    for user_data in train_data:
        item_seq = user_data['item_seq']

        # 需要足够长的序列
        if len(item_seq) < 15:
            conflict_stats['insufficient_data'] += 1
            continue

        # 划分长期和短期
        split_point = int(len(item_seq) * (1 - recent_ratio))
        long_term_items = item_seq[:split_point]
        recent_items = item_seq[split_point:]

        # 获取长期主导 genre（top 3）
        long_dominant, long_top3 = get_dominant_genre(long_term_items, item2genres)
        if long_dominant is None:
            conflict_stats['insufficient_data'] += 1
            continue

        # 获取短期主导 genre
        recent_dominant, recent_top3 = get_dominant_genre(recent_items, item2genres)
        if recent_dominant is None:
            conflict_stats['insufficient_data'] += 1
            continue

        # 判断冲突类型
        if recent_dominant == long_dominant:
            conflict_stats['consistent'] += 1
            if len(user_examples['consistent']) < 5:
                user_examples['consistent'].append({
                    'user_id': user_data.get('user_id', '?'),
                    'long_top': long_top3,
                    'recent_top': recent_top3,
                    'match': recent_dominant
                })
        elif recent_dominant in long_top3:
            conflict_stats['partial_conflict'] += 1
            if len(user_examples['partial_conflict']) < 5:
                user_examples['partial_conflict'].append({
                    'user_id': user_data.get('user_id', '?'),
                    'long_top': long_top3,
                    'recent_top': recent_top3,
                    'long_dominant': long_dominant,
                    'recent_dominant': recent_dominant
                })
        else:
            conflict_stats['strong_conflict'] += 1
            if len(user_examples['strong_conflict']) < 5:
                user_examples['strong_conflict'].append({
                    'user_id': user_data.get('user_id', '?'),
                    'long_top': long_top3,
                    'recent_top': recent_top3,
                    'long_dominant': long_dominant,
                    'recent_dominant': recent_dominant
                })

    return conflict_stats, user_examples


def plot_conflict_distribution(conflict_stats):
    """绘制冲突分布图"""
    plt.rcParams.update({
        'font.family': 'STIXGeneral',
        'font.size': 13,
        'axes.linewidth': 1.2,
    })

    # 计算有效用户数（排除数据不足）
    total_valid = sum([
        conflict_stats['consistent'],
        conflict_stats['partial_conflict'],
        conflict_stats['strong_conflict']
    ])

    # 计算百分比
    categories = ['Consistent\n(Same Interest)',
                  'Partial Shift\n(Related Interest)',
                  'Strong Shift\n(New Interest)']
    counts = [
        conflict_stats['consistent'],
        conflict_stats['partial_conflict'],
        conflict_stats['strong_conflict']
    ]
    percentages = [c / total_valid * 100 for c in counts]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：柱状图
    colors = ['#4472C4', '#FFC000', '#E07820']
    bars = ax1.bar(range(len(categories)), percentages, color=colors,
                   edgecolor='white', linewidth=1.5, alpha=0.85)

    # 添加数值标签
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n(n={cnt})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Percentage of Users (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_ylim(0, max(percentages) * 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Long-term vs. Short-term Interest Consistency',
                  fontsize=12, fontweight='bold', pad=10)

    # 右图：饼图
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax2.pie(
        counts, labels=categories, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 11}, wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )

    # 美化百分比文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax2.set_title('(b) Distribution of Interest Shift Patterns',
                  fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    # 保存图片（只保存 PNG，节省空间）
    plt.savefig('./figures/fig_long_short_conflict.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fig_long_short_conflict.png')

    return fig


def main():
    print('Loading ML-1M data...')
    item2genres, train_data = load_ml1m_data()
    print(f'  Loaded {len(item2genres)} items, {len(train_data)} users')

    print('\nComputing conflict statistics...')
    conflict_stats, examples = compute_conflict_statistics(item2genres, train_data)

    print('\n' + '='*60)
    print('CONFLICT STATISTICS')
    print('='*60)
    total_valid = sum([
        conflict_stats['consistent'],
        conflict_stats['partial_conflict'],
        conflict_stats['strong_conflict']
    ])

    print(f"Consistent (长短期一致):       {conflict_stats['consistent']:>5d} "
          f"({conflict_stats['consistent']/total_valid*100:>5.1f}%)")
    print(f"Partial Shift (部分转移):      {conflict_stats['partial_conflict']:>5d} "
          f"({conflict_stats['partial_conflict']/total_valid*100:>5.1f}%)")
    print(f"Strong Shift (强烈转移):       {conflict_stats['strong_conflict']:>5d} "
          f"({conflict_stats['strong_conflict']/total_valid*100:>5.1f}%)")
    print(f"Insufficient Data (数据不足):  {conflict_stats['insufficient_data']:>5d}")
    print(f"\nTotal Valid Users: {total_valid}")

    # 输出冲突比例
    conflict_ratio = (conflict_stats['partial_conflict'] + conflict_stats['strong_conflict']) / total_valid * 100
    print(f"\n** {conflict_ratio:.1f}% of users show interest shift between long-term and short-term **")

    print('\n' + '='*60)
    print('EXAMPLE CASES')
    print('='*60)

    print('\n[Consistent Users]')
    for ex in examples['consistent'][:3]:
        print(f"  User {ex['user_id']}: Long={ex['long_top']}, Recent={ex['recent_top']}, Match={ex.get('match')}")

    print('\n[Partial Shift Users]')
    for ex in examples['partial_conflict'][:3]:
        print(f"  User {ex['user_id']}: Long={ex['long_top']}, Recent={ex['recent_top']}")
        print(f"    Long dominant: {ex.get('long_dominant')}, Recent dominant: {ex.get('recent_dominant')}")

    print('\n[Strong Shift Users]')
    for ex in examples['strong_conflict'][:3]:
        print(f"  User {ex['user_id']}: Long={ex['long_top']}, Recent={ex['recent_top']}")
        print(f"    Long dominant: {ex.get('long_dominant')}, Recent dominant: {ex.get('recent_dominant')}")

    print('\nPlotting figure...')
    plot_conflict_distribution(conflict_stats)

    print('\nDone!')


if __name__ == '__main__':
    main()
