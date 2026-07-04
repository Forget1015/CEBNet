"""
Convert TedRec ML-1M data format to CEB-Net format.

TedRec format:
  - ml-1m.train.inter: TSV with header "user_id:token  item_id_list:token_seq  item_id:token"
    Each row: user_id \t space-separated-history \t target_item
  - ml-1m.item2index: TSV "original_id \t new_id" (0-based)
  - ml-1m.text: TSV "item_id:token \t text:token_seq" (item_id is original movie ID)

CEB-Net format:
  - ML-1M.train.jsonl: {"user_id": str, "target_id": str, "inter_history": [str, ...]}
  - ML-1M.item.json: list of all item string IDs
  - ML-1M.emb_map.json: {item_str: int_id} (1-based, 0=padding)
  - ML-1M.meta.json: {int_id_str: {"title": ..., "genres": ..., "meta": ...}}

Key difference: TedRec uses 0-based item IDs; CEB-Net uses 1-based (0=padding).
So we do: cebnet_id = tedrec_id + 1

Usage:
    python convert_tedrec_to_cebnet.py
"""
import os
import json
import re


def parse_text_file(text_path):
    """Parse ml-1m.text → {original_movie_id: text_string}"""
    texts = {}
    with open(text_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                item_id = parts[0].strip()
                text = parts[1].strip()
                texts[item_id] = text
    return texts


def parse_item2index(item2index_path):
    """Parse ml-1m.item2index → {original_id: tedrec_index}"""
    mapping = {}
    with open(item2index_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                orig_id = parts[0].strip()
                tedrec_idx = int(parts[1].strip())
                mapping[orig_id] = tedrec_idx
    return mapping


def parse_inter_file(inter_path):
    """Parse .inter file → list of (user_id, history_list, target_id)
    All IDs are TedRec 0-based indices (strings).
    """
    records = []
    with open(inter_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            user_id = parts[0].strip()
            history = parts[1].strip().split()
            target = parts[2].strip()
            records.append((user_id, history, target))
    return records


def extract_title_genres(text):
    """Extract title and genres from TedRec text like 'Toy Story 1995 Animation Children's Comedy.'
    Format: Title Year Genre1 Genre2 ...
    The year is 4 digits, genres follow after year.
    """
    # Find year pattern (4 digits)
    match = re.search(r'\b(\d{4})\b', text)
    if match:
        year_pos = match.start()
        year_end = match.end()
        title_part = text[:year_pos].strip()
        year = match.group(1)
        genres_part = text[year_end:].strip().rstrip('.')
        # Include year in title
        title = f"{title_part} ({year})"
    else:
        title = text.strip().rstrip('.')
        genres_part = ''

    return title, genres_part


def main():
    tedrec_dir = '/data0/yejinxuan/workspace/TedRec/dataset/ml-1m'
    output_dir = '/data0/yejinxuan/workspace/CEBNet/dataset/ML-1M_TedRec'
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'ML-1M_TedRec'

    # 1. Load item2index: original_movie_id → tedrec_0based_index
    print("Loading item2index...")
    orig2tedrec = parse_item2index(os.path.join(tedrec_dir, 'ml-1m.item2index'))
    # Reverse: tedrec_index → original_id
    tedrec2orig = {v: k for k, v in orig2tedrec.items()}
    n_items_tedrec = len(orig2tedrec)
    print(f"  Total items in TedRec: {n_items_tedrec}")

    # 2. Load text
    print("Loading text...")
    orig_texts = parse_text_file(os.path.join(tedrec_dir, 'ml-1m.text'))
    print(f"  Text entries: {len(orig_texts)}")

    # 3. CEB-Net uses 1-based IDs (0=padding)
    # cebnet_id = tedrec_id + 1
    # item string in jsonl = str(tedrec_id)  (we keep tedrec IDs as string keys)

    # 4. Convert inter files
    for split in ['train', 'valid', 'test']:
        print(f"Converting {split}...")
        inter_path = os.path.join(tedrec_dir, f'ml-1m.{split}.inter')
        records = parse_inter_file(inter_path)

        out_path = os.path.join(output_dir, f'{dataset}.{split}.jsonl')
        with open(out_path, 'w') as f:
            for user_id, history, target in records:
                entry = {
                    "user_id": user_id,
                    "target_id": target,
                    "inter_history": history
                }
                f.write(json.dumps(entry) + '\n')
        print(f"  Written {out_path} ({len(records)} lines)")

    # 5. Build item.json: sorted list of all item string IDs
    all_items = [str(i) for i in range(n_items_tedrec)]
    with open(os.path.join(output_dir, f'{dataset}.item.json'), 'w') as f:
        json.dump(all_items, f)
    print(f"  item.json: {len(all_items)} items")

    # 6. Build emb_map.json: {item_str: cebnet_1based_id}
    emb_map = {}
    for tedrec_idx in range(n_items_tedrec):
        emb_map[str(tedrec_idx)] = tedrec_idx + 1  # 1-based
    with open(os.path.join(output_dir, f'{dataset}.emb_map.json'), 'w') as f:
        json.dump(emb_map, f)
    print(f"  emb_map.json: {len(emb_map)} entries")

    # 7. Build meta.json: {cebnet_1based_id_str: {"title": ..., "genres": ..., "meta": ...}}
    meta = {}
    missing_text = 0
    for tedrec_idx in range(n_items_tedrec):
        cebnet_id = tedrec_idx + 1
        orig_id = tedrec2orig.get(tedrec_idx)
        if orig_id and orig_id in orig_texts:
            text = orig_texts[orig_id]
            title, genres = extract_title_genres(text)
            meta[str(cebnet_id)] = {
                "title": title,
                "genres": genres,
                "meta": f"{title} {genres}".strip()
            }
        else:
            missing_text += 1
            meta[str(cebnet_id)] = {"title": "", "genres": "", "meta": ""}

    with open(os.path.join(output_dir, f'{dataset}.meta.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"  meta.json: {len(meta)} entries, {missing_text} missing text")

    # 8. Print sample
    print("\n--- Sample data ---")
    sample_path = os.path.join(output_dir, f'{dataset}.train.jsonl')
    with open(sample_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  train[{i}]: {line.strip()}")

    sample_keys = list(meta.keys())[:5]
    for k in sample_keys:
        print(f"  meta[{k}]: {meta[k]}")

    print(f"\nDone! Output in {output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Generate text embeddings:")
    print(f"     python encode_emb.py --dataset {dataset} --text_types title genres --gpu_id 0 --data_path ./dataset")
    print(f"  2. Generate VQ codes (update vq/ML-1M_TedRec.yaml first):")
    print(f"     cd vq && python generate_faiss_multi_emb.py --config ML-1M_TedRec.yaml")
    print(f"  3. Train CEB-Net:")
    print(f"     python main.py --dataset {dataset} --text_types title genres \\")
    print(f"       --text_index_path .code.pq.8_64.pca128.title_genres.json \\")
    print(f"       --code_level 8 --n_codes_per_lel 64 --max_his_len 200 \\")
    print(f"       --neg_num 3000 --batch_size 600 --device cuda:7")


if __name__ == '__main__':
    main()
