# 超参数分析实验配置

**实验设计**：针对 wm_length 和 n_prototypes 两个关键超参数，在三个数据集上进行系统分析。

**架构确认**（符合大纲.md）：
- ✅ `--no_webd_rehearsal`：WEBD 不使用 Transformer，只用 DWT + 动态阈值
- ✅ `--use_decoupled_trace_id`：Trace ID 在检索阶段引入
- ✅ `--use_seq_branch`：DMR 使用 CausalTransformer 构建查询

**参考配置来源**：
- Industrial_and_Scientific: 基于 Jul-15-2026 log，去掉消融标记，添加架构对齐
- Video_Games: 基于 Jun-29-2026_14-37 log + 添加 `--no_webd_rehearsal`
- ML-1M_TedRec: 基于 Jun-29-2026_14-42 log + 添加 `--no_webd_rehearsal`

---

## 1. Industrial_and_Scientific - wm_length 分析

**固定参数**：n_prototypes=16（默认值）  
**变化参数**：wm_length ∈ {0.1, 0.2, 0.3, 0.5}

### wm_length=0.1
```bash
python main.py \
    --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \
    --data_path=./dataset/ \
    --text_types title brand features categories description \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
    --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
    --dropout_prob=0.2 --dropout_prob_cross=0.1 \
    --wm_length=0.1 --n_prototypes=16 --wavelet=haar \
    --n_layers_webd=2 --n_layers_smc=2 --no_webd_rehearsal \
    --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \
    --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
    --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
    --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
    --history_neg_weight=0.05 --history_neg_num=10 \
    --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
    --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
    --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" --valid_metric=ndcg@10
```

### wm_length=0.2
```bash
python main.py \
    --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \
    --data_path=./dataset/ \
    --text_types title brand features categories description \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
    --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
    --dropout_prob=0.2 --dropout_prob_cross=0.1 \
    --wm_length=0.2 --n_prototypes=16 --wavelet=haar \
    --n_layers_webd=2 --n_layers_smc=2 --no_webd_rehearsal \
    --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \
    --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
    --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
    --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
    --history_neg_weight=0.05 --history_neg_num=10 \
    --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
    --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
    --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \
    --metrics="recall@5,ndcg@5,recall@10,ndcg@10" --valid_metric=ndcg@10
```
