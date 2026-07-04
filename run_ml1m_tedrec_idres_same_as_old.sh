#!/bin/bash
# CEB-Net on ML-1M_TedRec with old training config plus trainable ID residual gate.
# Reference log: Apr-23-2026_14-34-77aea7, which used neg_num=3000 and batch_size=1000.

DEVICE=${1:-cuda:4}

/data0/yejinxuan/miniconda3/envs/CCF/bin/python main.py \
    --dataset=ML-1M_TedRec \
    --lr=0.001 \
    --neg_num=3000 \
    --text_types title genres \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.6 \
    --data_path=./dataset \
    --text_index_path=.code.pq.8_64.pca128.title_genres.json \
    --code_level=8 \
    --n_codes_per_lel=64 \
    --max_his_len=50 \
    --batch_size=1000 \
    --dropout_prob=0.2 \
    --dropout_prob_cross=0.2 \
    --n_layers=2 \
    --n_layers_cross=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512 \
    --wm_length=20 \
    --n_prototypes=16 \
    --wavelet=haar \
    --ortho_weight=0.1 \
    --freq_weight=0.01 \
    --n_layers_webd=2 \
    --n_layers_smc=1 \
    --use_id_residual \
    --eval_test_each_epoch \
    --device=$DEVICE
