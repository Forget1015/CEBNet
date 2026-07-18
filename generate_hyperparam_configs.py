#!/usr/bin/env python3
"""
生成超参数分析实验配置文件
输出: hyperparameter_analysis_full.md
"""

# 基础配置模板
CONFIGS = {
    "Industrial_and_Scientific": {
        "base": """python main.py \\
    --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \\
    --data_path=./dataset/ \\
    --text_types title brand features categories description \\
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \\
    --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \\
    --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \\
    --dropout_prob=0.2 --dropout_prob_cross=0.1 \\
    --wm_length={wm} --n_prototypes={K} --wavelet=haar \\
    --n_layers_webd=2 --n_layers_smc=2 --no_webd_rehearsal \\
    --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \\
    --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \\
    --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \\
    --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \\
    --history_neg_weight=0.05 --history_neg_num=10 \\
    --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \\
    --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \\
    --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \\
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" --valid_metric=ndcg@10""",
        "wm_default": 0.2,
        "K_default": 16,
    },
    "Video_Games": {
        "base": """python main.py \\
    --seed=2020 --dataset=Video_Games --device=cuda:0 \\
    --data_path=./dataset \\
    --text_types title brand features categories description \\
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \\
    --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \\
    --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \\
    --dropout_prob=0.2 --dropout_prob_cross=0.1 \\
    --wm_length={wm} --n_prototypes={K} --wavelet=haar \\
    --n_layers_webd=2 --n_layers_smc=2 --no_webd_rehearsal \\
    --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \\
    --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \\
    --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \\
    --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \\
    --history_neg_weight=0.05 --history_neg_num=10 \\
    --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \\
    --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \\
    --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \\
    --eval_step=1 --eval_test_each_epoch \\
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" --valid_metric=ndcg@10""",
        "wm_default": 0.2,
        "K_default": 16,
    },
    "ML-1M_TedRec": {
        "base": """python main.py \\
    --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \\
    --data_path=./dataset \\
    --text_types title genres \\
    --text_index_path=.code.pq.8_64.pca128.title_genres.json \\
    --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \\
    --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \\
    --dropout_prob=0.2 --dropout_prob_cross=0.2 \\
    --wm_length={wm} --n_prototypes={K} --wavelet=haar \\
    --n_layers_webd=2 --n_layers_smc=1 --no_webd_rehearsal \\
    --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \\
    --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \\
    --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \\
    --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \\
    --history_neg_weight=0.05 --history_neg_num=10 \\
    --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \\
    --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \\
    --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \\
    --eval_step=1 --eval_test_each_epoch \\
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" --valid_metric=ndcg@10""",
        "wm_default": 0.4,
        "K_default": 16,
    }
}

# 超参数取值
WM_VALUES = [0.1, 0.2, 0.3, 0.5]
K_VALUES = [4, 8, 16, 32]

# 生成文件
output = []
output.append("# 超参数分析实验配置\n")
output.append("**实验设计**：针对 wm_length 和 n_prototypes 两个关键超参数，在三个数据集上进行系统分析。\n\n")
output.append("**架构确认**（符合大纲.md）：\n")
output.append("- ✅ `--no_webd_rehearsal`：WEBD 不使用 Transformer，只用 DWT + 动态阈值\n")
output.append("- ✅ `--use_decoupled_trace_id`：Trace ID 在检索阶段引入\n")
output.append("- ✅ `--use_seq_branch`：DMR 使用 CausalTransformer 构建查询\n\n")
output.append("**使用方法**：直接复制对应的命令运行即可。\n\n")
output.append("---\n\n")

for dataset, config in CONFIGS.items():
    # wm_length 分析
    output.append(f"## {dataset} - wm_length 分析\n\n")
    output.append(f"**固定参数**：n_prototypes={config['K_default']}\n")
    output.append(f"**变化参数**：wm_length ∈ {WM_VALUES}\n\n")

    for wm in WM_VALUES:
        output.append(f"### wm_length={wm}\n```bash\n")
        output.append(config['base'].format(wm=wm, K=config['K_default']))
        output.append("\n```\n\n")

    # n_prototypes 分析
    output.append(f"## {dataset} - n_prototypes 分析\n\n")
    output.append(f"**固定参数**：wm_length={config['wm_default']}\n")
    output.append(f"**变化参数**：n_prototypes ∈ {K_VALUES}\n\n")

    for K in K_VALUES:
        output.append(f"### n_prototypes={K}\n```bash\n")
        output.append(config['base'].format(wm=config['wm_default'], K=K))
        output.append("\n```\n\n")

    output.append("---\n\n")

# 写入文件
with open('hyperparameter_analysis_full.md', 'w') as f:
    f.write(''.join(output))

print(f"✓ 已生成 hyperparameter_analysis_full.md")
print(f"  包含 3 个数据集 × 2 个超参数 × 4 个取值 = 24 个实验配置")
