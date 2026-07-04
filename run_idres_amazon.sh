#!/bin/bash
# CEB-Net with trainable ID residual gate for Amazon-style CCFRec datasets.

DATASET=${DATASET:-Musical_Instruments}
DEVICE=${DEVICE:-cuda:0}
MAX_HIS_LEN=${MAX_HIS_LEN:-20}
BATCH_SIZE=${BATCH_SIZE:-400}

/data0/yejinxuan/miniconda3/envs/CCF/bin/python main.py \
    --dataset=$DATASET \
    --lr=0.001 \
    --neg_num=24000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.6 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=$MAX_HIS_LEN \
    --batch_size=$BATCH_SIZE \
    --dropout_prob=0.3 \
    --dropout_prob_cross=0.3 \
    --n_layers=2 \
    --n_layers_cross=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512 \
    --wm_length=5 \
    --n_prototypes=16 \
    --wavelet=haar \
    --ortho_weight=0.1 \
    --freq_weight=0.01 \
    --n_layers_webd=2 \
    --n_layers_smc=2 \
    --use_id_residual \
    --device=$DEVICE
