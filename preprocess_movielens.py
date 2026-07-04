"""
MovieLens-1M preprocessing for CEB-Net.
Downloads ML-1M, converts to CCFRec-compatible format.

Usage:
    python preprocess_movielens.py --dataset ML-1M --his_len 50
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def load_movielens_1m(data_dir):
    """Load ML-1M ratings and movies."""
    # Ratings: UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(
        os.path.join(data_dir, 'ratings.dat'), sep='::', engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    # Movies: MovieID::Title::Genres
    movies = pd.read_csv(
        os.path.join(data_dir, 'movies.dat'), sep='::', engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    return ratings, movies


def split_sequences(ratings, his_len=50):
    """
    Split user interactions into train/valid/test.
    For each user: last item = test, second-to-last = valid, rest = train.
    """
    # Sort by user and timestamp
    ratings = ratings.sort_values(['user_id', 'timestamp'])

    # Filter users with >= 5 interactions
    user_counts = ratings.groupby('user_id').size()
    valid_users = user_counts[user_counts >= 5].index
    ratings = ratings[ratings['user_id'].isin(valid_users)]

    train_data, valid_data, test_data = [], [], []

    for user_id, group in ratings.groupby('user_id'):
        items = group['movie_id'].tolist()

        if len(items) < 5:
            continue

        # Test: predict last item, history = all except last
        test_history = items[:-1][-his_len:]
        test_data.append({
            'user_id': str(user_id),
            'target_id': str(items[-1]),
            'inter_history': [str(x) for x in test_history]
        })

        # Valid: predict second-to-last, history = all except last two
        valid_history = items[:-2][-his_len:]
        valid_data.append({
            'user_id': str(user_id),
            'target_id': str(items[-2]),
            'inter_history': [str(x) for x in valid_history]
        })

        # Train: for each position from 4th to (n-2)th
        for idx in range(3, len(items) - 2):
            train_history = items[:idx][-his_len:]
            train_data.append({
                'user_id': str(user_id),
                'target_id': str(items[idx]),
                'inter_history': [str(x) for x in train_history]
            })

    return train_data, valid_data, test_data


def build_meta(movies):
    """Build meta info dict: {item_id: {"title": ..., "genres": ..., "meta": ...}}"""
    meta = {}
    for _, row in movies.iterrows():
        movie_id = str(row['movie_id'])
        title = str(row['title']).strip()
        genres = str(row['genres']).replace('|', ' ').strip()
        meta_text = title + ' ' + genres
        meta[movie_id] = {
            'title': title,
            'genres': genres,
            'meta': meta_text
        }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ML-1M')
    parser.add_argument('--his_len', type=int, default=50)
    parser.add_argument('--ml_dir', type=str, default='./dataset/ml-1m',
                        help='Path to extracted ml-1m directory containing ratings.dat and movies.dat')
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = f'./dataset/{dataset}'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading MovieLens from {args.ml_dir}...")
    ratings, movies = load_movielens_1m(args.ml_dir)
    print(f"  Ratings: {len(ratings)}, Movies: {len(movies)}, Users: {ratings['user_id'].nunique()}")

    # Split
    print("Splitting sequences...")
    train, valid, test = split_sequences(ratings, his_len=args.his_len)
    print(f"  Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Collect all items
    all_items = set()
    for split in [train, valid, test]:
        for entry in split:
            all_items.add(entry['target_id'])
            all_items.update(entry['inter_history'])
    all_items = sorted(all_items, key=lambda x: int(x))
    item2id = {item: idx + 1 for idx, item in enumerate(all_items)}
    print(f"  Total items: {len(all_items)}")

    # Write jsonl files
    for name, data in [('train', train), ('valid', valid), ('test', test)]:
        path = os.path.join(output_dir, f'{dataset}.{name}.jsonl')
        with open(path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"  Written {path} ({len(data)} lines)")

    # Write item.json
    with open(os.path.join(output_dir, f'{dataset}.item.json'), 'w') as f:
        json.dump(all_items, f)

    # Write emb_map.json
    with open(os.path.join(output_dir, f'{dataset}.emb_map.json'), 'w') as f:
        json.dump(item2id, f)

    # Build and write meta.json
    raw_meta = build_meta(movies)
    # Reindex by item2id
    meta_indexed = {}
    for item_str, idx in item2id.items():
        if item_str in raw_meta:
            meta_indexed[str(idx)] = raw_meta[item_str]
        else:
            meta_indexed[str(idx)] = {'title': '', 'genres': '', 'meta': ''}
    with open(os.path.join(output_dir, f'{dataset}.meta.json'), 'w') as f:
        json.dump(meta_indexed, f)

    # Print stats
    hist_lens = [len(e['inter_history']) for e in train]
    print(f"\n  Train history length: mean={np.mean(hist_lens):.1f}, "
          f"median={np.median(hist_lens):.1f}, max={np.max(hist_lens)}")

    print(f"\nDone! Output in {output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Encode text embeddings:")
    print(f"     python encode_emb.py --dataset {dataset} --text_types title genres --gpu_id 0")
    print(f"  2. Generate VQ codes:")
    print(f"     cd vq && python generate_faiss_multi_emb.py --config {dataset}.yaml")
    print(f"  3. Train CEB-Net:")
    print(f"     python main.py --dataset {dataset} --text_types title genres --max_his_len {args.his_len}")


if __name__ == '__main__':
    main()
