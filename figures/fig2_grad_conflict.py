"""
Figure 2: Gradient conflict between rec_loss and cl_loss on shared item embeddings.

Run from: /data0/yejinxuan/workspace/CEBNet
"""
import sys, os
os.chdir('/data0/yejinxuan/workspace/CEBNet')
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from data import load_split_data, CEBNetDataset, Collator
from torch.utils.data import DataLoader
from model import CEBNet
from utils import load_json

CKPT = ('./myckpt/Industrial_and_Scientific/'
        'Jun-29-2026_09-21-9e07b1_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_'
        'dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05/best_model.pth')
DEVICE    = 'cuda:0'
N_BATCHES = 150


def main():
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = state['args']
    args.device    = DEVICE
    args.neg_num   = 0       # full-sort avoids needing neg sampling
    args.batch_size = 64
    args.num_workers = 2
    if not hasattr(args, 'no_webd_rehearsal'):
        setattr(args, 'no_webd_rehearsal', False)

    device = torch.device(DEVICE)
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(f'./dataset/{args.dataset}/{args.dataset}{args.text_index_path}')
    train_dataset = CEBNetDataset(args, n_items, train, index, 'train')
    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=Collator(args), num_workers=2)

    text_embs = []
    for ttype in args.text_types:
        emb = np.load(f'./dataset/{args.dataset}/{args.dataset}.t5.{ttype}.emb.npy')
        emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(emb)
        text_embs.append(emb)
    args.text_embedding_size = text_embs[0].shape[-1]

    model = CEBNet(args, train_dataset, index, device).to(device)
    for i, emb in enumerate(text_embs):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(
            emb, dtype=torch.float32, device=device)
    model.load_state_dict(state['state_dict'], strict=False)
    model.train()

    # Use query_code_embedding — trainable, shared by both rec and CL paths
    watch_param = model.query_code_embedding.weight  # [n_codes, d], requires_grad=True

    angle_list = []
    for batch_idx, batch in enumerate(tqdm(loader, total=N_BATCHES)):
        if batch_idx >= N_BATCHES:
            break

        item_inters  = batch['item_inters'].to(device)
        inter_lens   = batch['inter_lens'].to(device)
        targets      = batch['targets'].to(device)
        code_inters  = batch['code_inters'].to(device)
        mask_targets = batch['mask_targets'].to(device)

        # Pass 1: rec_loss only
        model.zero_grad()
        loss_dict = model.calculate_loss(item_inters, inter_lens, targets, code_inters, mask_targets)
        loss_dict['rec_loss'].backward()
        if watch_param.grad is None:
            continue
        grad_rec = watch_param.grad.detach().clone()

        # Pass 2: cl_loss only
        model.zero_grad()
        loss_dict2 = model.calculate_loss(item_inters, inter_lens, targets, code_inters, mask_targets)
        cl = loss_dict2.get('cl_loss', None)
        if cl is None or cl.item() == 0:
            continue
        cl.backward()
        if watch_param.grad is None:
            continue
        grad_cl = watch_param.grad.detach().clone()

        # Per-row cosine similarity (active rows only)
        active = ((grad_rec.abs().sum(-1) > 1e-9) &
                  (grad_cl.abs().sum(-1)  > 1e-9))
        if active.sum() < 2:
            continue
        gr = grad_rec[active].float()
        gc = grad_cl[active].float()
        cos = torch.nn.functional.cosine_similarity(gr, gc, dim=1).cpu().numpy()
        angle_list.extend(cos.tolist())

    angle_arr = np.array([a for a in angle_list if not np.isnan(a)])
    np.save('./figures/grad_angles.npy', angle_arr)

    conflict_pct   = (angle_arr < 0).mean() * 100
    consistent_pct = (angle_arr >= 0).mean() * 100
    print(f'\nPairs collected: {len(angle_arr)}')
    print(f'Conflicting:     {conflict_pct:.1f}%')
    print(f'Consistent:      {consistent_pct:.1f}%')

    _plot(angle_arr, conflict_pct, consistent_pct)


def _plot(angle_arr, conflict_pct, consistent_pct):
    plt.rcParams.update({
        'font.family': 'STIXGeneral',
        'font.size': 13,
        'axes.linewidth': 1.2,
    })
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    weights = np.full(len(angle_arr), 1.0 / len(angle_arr))

    ax.axvline(x=0, color='#7F7F7F', lw=1.2, zorder=2)

    n, bins, patches = ax.hist(angle_arr, bins=40, weights=weights,
                               range=(-1, 1), edgecolor='white',
                               linewidth=0.3, zorder=3)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor('#E07820' if left < 0 else '#4472C4')

    ylim = ax.get_ylim()[1]
    ax.text(-0.68, ylim * 0.88, 'Conflicting',
            fontsize=12, color='#7F3000', fontweight='bold', ha='center')
    ax.text(-0.68, ylim * 0.74, f'{conflict_pct:.1f}%',
            fontsize=12, color='#7F3000', ha='center')
    ax.text( 0.65, ylim * 0.88, 'Consistent',
            fontsize=12, color='#1A3A6B', fontweight='bold', ha='center')
    ax.text( 0.65, ylim * 0.74, f'{consistent_pct:.1f}%',
            fontsize=12, color='#1A3A6B', ha='center')

    ax.set_xlabel('Cosine Similarity of Gradients', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(alpha=0.25, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('./figures/fig2_grad_conflict.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('./figures/fig2_grad_conflict.png', bbox_inches='tight', dpi=300)
    print('Figure 2 saved.')


if __name__ == '__main__':
    main()
