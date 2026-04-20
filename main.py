"""
CEB-Net: Cognitive Episodic Buffer Network for Sequential Recommendation
"""
import os
import argparse
import warnings
import torch
import numpy as np
from data import load_split_data, CEBNetDataset, Collator
from torch.utils.data import DataLoader
from model import CEBNet
from trainer import CEBNetTrainer
from utils import init_logger, init_seed, get_local_time, log, get_file_name, load_json
from logging import getLogger
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description="CEB-Net")

    # Base
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--cl_weight', type=float, default=0.4)
    parser.add_argument('--mlm_weight', type=float, default=0.6)
    parser.add_argument('--neg_num', type=int, default=24000)
    parser.add_argument('--text_types', nargs='+', type=str,
                        default=['title', 'brand', 'features', 'categories', 'description'])

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=10)

    # Data
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument('--map_path', type=str, default=".emb_map.json")
    parser.add_argument('--text_index_path', type=str,
                        default=".code.pq.20_256.pca128.title_brand_features_categories_description.json")
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--n_codes_per_lel', type=int, default=256)
    parser.add_argument('--code_level', type=int, default=20)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    # Model (shared with CCFRec)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_layers_cross', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)

    # CEB-Net specific
    parser.add_argument('--wm_length', type=int, default=5)
    parser.add_argument('--n_prototypes', type=int, default=16)
    parser.add_argument('--wavelet', type=str, default='haar',
                        choices=['haar', 'db4', 'sym4', 'coif2'])
    parser.add_argument('--ortho_weight', type=float, default=0.1)
    parser.add_argument('--freq_weight', type=float, default=0.01)
    parser.add_argument('--attn_size', type=int, default=None)
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    parser.add_argument('--n_layers_webd', type=int, default=2)
    parser.add_argument('--n_layers_smc', type=int, default=2)
    parser.add_argument('--wavelet_levels', type=int, default=2,
                        help='Number of wavelet decomposition levels')
    parser.add_argument('--decouple_weight', type=float, default=0.01,
                        help='Weight for decoupling regularization loss')

    # Output
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--metrics', type=str, default="recall@5,ndcg@5,recall@10,ndcg@10")
    parser.add_argument('--valid_metric', type=str, default="ndcg@10")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--ckpt_dir", type=str, default="./myckpt/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    args.run_local_time = get_local_time()
    args_dict = vars(args)

    args.save_file_name = (
        get_file_name(args_dict)
        + f"_wm{args.wm_length}_K{args.n_prototypes}_wav{args.wavelet}"
        + f"_mlm{args.mlm_weight}_cl{args.cl_weight}"
        + f"_drop{args.dropout_prob}_dpcross{args.dropout_prob_cross}"
    )

    init_seed(args.seed, True)
    init_logger(args_dict, args.save_file_name + '.log')
    logger = getLogger()
    log(args_dict, logger)

    device = torch.device(args.device)
    dataset_path = os.path.join(args.data_path, args.dataset)

    # Load data
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(dataset_path, args.dataset + args.text_index_path))

    train_dataset = CEBNetDataset(args, n_items, train, index, 'train')
    val_dataset = CEBNetDataset(args, n_items, val, index, 'val')
    test_dataset = CEBNetDataset(args, n_items, test, index, 'test')
    collator = Collator(args)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              collate_fn=collator, batch_size=args.batch_size,
                              shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers,
                            collate_fn=collator, batch_size=args.batch_size,
                            shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                             collate_fn=collator, batch_size=args.batch_size,
                             shuffle=False, pin_memory=False)

    # Load text embeddings
    text_embs = []
    for ttype in args.text_types:
        if ttype not in ['meta', 'title', 'brand', 'features', 'categories', 'description']:
            raise ValueError(f"{ttype} not in valid text types")
        text_emb_file = f".t5.{ttype}.emb.npy"
        text_emb = np.load(os.path.join(args.data_path, args.dataset,
                                        args.dataset + text_emb_file))
        text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
        text_embs.append(text_emb)
    args.text_embedding_size = text_embs[0].shape[-1]

    # Build model
    model = CEBNet(args, train_dataset, index, device).to(device)

    # Load frozen text embeddings
    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(
            text_embs[i], dtype=torch.float32, device=device)

    trainer = CEBNetTrainer(args, model, train_loader, val_loader, test_loader, device)
    log(model, logger)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)

    best_score, best_results = trainer.fit(start_epoch=start_epoch)
    test_results = trainer.test()

    log(f"Best Validation Score: {best_score}", logger)
    log(f"Test Results: {test_results}", logger)
