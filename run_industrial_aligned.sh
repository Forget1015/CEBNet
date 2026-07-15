#!/bin/bash
# Industrial_and_Scientific 完整超参数配置（符合大纲架构）
# 对应 log: Jul-14-2026_06-33-fc8180_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log
#
# 关键架构特性：
# 1. WEBD 去噪：no_webd_rehearsal=True → 只有 DWT + 动态阈值 + IDWT，没有 Transformer
# 2. Trace ID：只在查询阶段使用 (use_decoupled_trace_id=True, use_id_residual=False)
# 3. SMC：n_layers_smc=2 用于长期记忆编码

DATASET=Industrial_and_Scientific
DEVICE=cuda:0

python main.py \
    --seed=2020 \
    --dataset=$DATASET \
    --device=$DEVICE \
    --data_path=./dataset/ \
    --text_types title brand features categories description \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --mask_ratio=0.5 \
    --embedding_size=128 \
    --hidden_size=512 \
    --n_heads=2 \
    --n_layers=2 \
    --n_layers_cross=2 \
    --dropout_prob=0.2 \
    --dropout_prob_cross=0.1 \
    --wm_length=10 \
    --n_prototypes=16 \
    --wavelet=haar \
    --n_layers_webd=2 \
    --n_layers_smc=2 \
    --no_webd_rehearsal \
    --ortho_weight=0.1 \
    --burst_loss_weight=0.05 \
    --proto_temperature=1.0 \
    --use_decoupled_trace_id \
    --trace_id_gate_bias_init=-2.0 \
    --use_seq_branch \
    --n_layers_seq=2 \
    --seq_fusion_mode=trace_residual_debr \
    --seq_gate_bias_init=-2.0 \
    --seq_add_weight=0.2 \
    --trace_memory_gate_bias_init=-2.0 \
    --history_neg_weight=0.05 \
    --history_neg_num=10 \
    --lr=0.0005 \
    --tau=0.05 \
    --cl_weight=0.2 \
    --mlm_weight=0.3 \
    --neg_num=24000 \
    --batch_size=300 \
    --num_workers=8 \
    --learner=AdamW \
    --lr_scheduler_type=constant \
    --warmup_steps=500 \
    --weight_decay=0.0001 \
    --epochs=500 \
    --early_stop=10 \
    --eval_step=1 \
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" \
    --valid_metric=ndcg@10
