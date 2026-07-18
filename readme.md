python main.py \
  --dataset=Musical_Instruments \
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
  --max_his_len=20 \
  --batch_size=100 \
  --dropout_prob=0.3 \
  --dropout_prob_cross=0.3 \
  --n_layers=2 \
  --n_heads=4 \
  --embedding_size=128 \
  --hidden_size=512 \
  --early_stop=100 \
  --log_dir="./logs" \
  --device=cuda:7 \
  --wm_length=5 \
  --n_prototypes=32 \
  --wavelet=haar \
  --ortho_weight=0.1 \
  --freq_weight=0.01 \
  --proto_temperature=1.0


python main.py \
    --dataset=Musical_Instruments \
    --lr=5e-4 \
    --neg_num=24000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.6 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=20 \
    --batch_size=100 \
    --dropout_prob=0.3 \
    --dropout_prob_cross=0.3 \
    --n_layers=2 \
    --n_heads=4 \
    --embedding_size=128 \
    --hidden_size=512 \
    --early_stop=100 \
    --log_dir="./logs" \
    --device=cuda:6 \
    --wm_length=10 \
    --n_prototypes=32 \
    --wavelet=haar \
    --ortho_weight=0.01 \
    --freq_weight=0.001 \
    --proto_temperature=1.0

python main.py \
    --dataset=Musical_Instruments \
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
    --max_his_len=50 \
    --batch_size=450 \
    --dropout_prob=0.3 \
    --dropout_prob_cross=0.3 \
    --n_layers=2 \
    --n_heads=4 \
    --embedding_size=128 \
    --hidden_size=512 \
    --early_stop=100 \
    --log_dir="./logs/长度为50" \
    --device=cuda:6 \
    --wm_length=5 \
    --n_prototypes=32 \
    --wavelet=haar \
    --ortho_weight=0.1 \
    --freq_weight=0.01 \
    --proto_temperature=1.0\
    --n_layers_webd=2\
    --n_layers_smc=2


python main.py \
    --dataset Video_Games \
    --lr=0.001 \
    --neg_num=24000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.5 \
    --mlm_weight=0.3 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=50 \
    --batch_size=300 \
    --dropout_prob=0.2 \
    --dropout_prob_cross=0.1 \
    --n_layers=2 \
    --n_layers_cross=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512 \
    --wm_length=10 \
    --n_prototypes=16 \
    --wavelet=haar \
    --ortho_weight=0.1 \
    --freq_weight=0.01 \
    --n_layers_webd=2 \
    --n_layers_smc=2 \
    --proto_temperature=1.0 \
    --eval_test_each_epoch \
    --use_id_residual \
    --device=cuda:4


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
    --device=cuda:4


  python main.py \
      --seed=2020 \
      --dataset=Industrial_and_Scientific \
      --device=cuda:0 \
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
      --disable_webd \
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
      --valid_metric=ndcg@10 \
      --device=cuda:5

python main.py \
      --seed=2020 \
      --dataset=Industrial_and_Scientific \
      --device=cuda:0 \
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
      --wm_length=50 \
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
      --valid_metric=ndcg@10 \
      --device=cuda:6


  消融一：w/o wavelet

  python main.py \
      --seed=2020 \
      --dataset=Industrial_and_Scientific \
      --device=cuda:0 \
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
      --disable_webd \
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

  消融二：w/o memory split

  python main.py \
      --seed=2020 \
      --dataset=Industrial_and_Scientific \
      --device=cuda:0 \
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
      --wm_length=50 \
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

消融三：fixed threshold（固定阈值）

  python main.py \
      --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \
      --data_path=./dataset/ \
      --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --fixed_threshold --threshold_value=0.5 \
      --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \
      --device=cuda:7 

  消融四：w/o SMP（去掉原型化）

  python main.py \
      --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \
      --data_path=./dataset/ \
      --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --no_smc \
      --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \
      --device=cuda:7 


  消融五：w/o decoupling（不解耦检索）

  python main.py \
      --seed=2020 --dataset=Industrial_and_Scientific --device=cuda:0 \
      --data_path=./dataset/ \
      --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --no_debr_decoupling \
      --ortho_weight=0.1 --burst_loss_weight=0.05 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.0005 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 --eval_step=1 \
      --device=cuda:6


 ML-1M 消融一：w/o wavelet

  python main.py \
      --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \
      --data_path=./dataset --text_types title genres \
      --text_index_path=.code.pq.8_64.pca128.title_genres.json \
      --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.2 \
      --wm_length=20 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=1 \
      --disable_webd --no_webd_rehearsal \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \
      --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1 \
      --device=cuda:5

  ML-1M 消融二：w/o memory split

  python main.py \
      --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \
      --data_path=./dataset --text_types title genres \
      --text_index_path=.code.pq.8_64.pca128.title_genres.json \
      --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.2 \
      --wm_length=1.0 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=1 \
      --no_webd_rehearsal \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \
      --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1 \
      --device=cuda:4

  ML-1M 消融三：fixed threshold

  python main.py \
      --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \
      --data_path=./dataset --text_types title genres \
      --text_index_path=.code.pq.8_64.pca128.title_genres.json \
      --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.2 \
      --wm_length=20 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=1 \
      --no_webd_rehearsal --fixed_threshold --threshold_value=0.5 \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \
      --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1 \
      --device=cuda:0

  ML-1M 消融四：w/o SMP

  python main.py \
      --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \
      --data_path=./dataset --text_types title genres \
      --text_index_path=.code.pq.8_64.pca128.title_genres.json \
      --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.2 \
      --wm_length=20 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=1 \
      --no_webd_rehearsal --no_smc \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \
      --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1\
      --device=cuda:1

  ML-1M 消融五：w/o decoupling

  python main.py \
      --seed=2020 --dataset=ML-1M_TedRec --device=cuda:0 \
      --data_path=./dataset --text_types title genres \
      --text_index_path=.code.pq.8_64.pca128.title_genres.json \
      --code_level=8 --n_codes_per_lel=64 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.2 \
      --wm_length=20 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=1 \
      --no_webd_rehearsal --no_debr_decoupling \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=3000 \
      --batch_size=1000 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1\
      --device=cuda:2


 Video_Games 消融一：w/o wavelet

  python main.py \
      --seed=2020 --dataset=Video_Games --device=cuda:0 \
      --data_path=./dataset --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --disable_webd --no_webd_rehearsal \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1 \
      --device=cuda:6

  Video_Games 消融二：w/o memory split

  python main.py \
      --seed=2020 --dataset=Video_Games --device=cuda:0 \
      --data_path=./dataset --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=1.0 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1 \
      --device=cuda:7

  Video_Games 消融三：fixed threshold

  python main.py \
      --seed=2020 --dataset=Video_Games --device=cuda:0 \
      --data_path=./dataset --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --fixed_threshold --threshold_value=0.5 \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1\
      --device=cuda:6

  Video_Games 消融四：w/o SMP

  python main.py \
      --seed=2020 --dataset=Video_Games --device=cuda:0 \
      --data_path=./dataset --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --no_smc \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1\
      --device=cuda:7

  Video_Games 消融五：w/o decoupling

  python main.py \
      --seed=2020 --dataset=Video_Games --device=cuda:0 \
      --data_path=./dataset --text_types title brand features categories description \
      --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
      --code_level=20 --n_codes_per_lel=256 --max_his_len=50 --mask_ratio=0.5 \
      --embedding_size=128 --hidden_size=512 --n_heads=2 --n_layers=2 --n_layers_cross=2 \
      --dropout_prob=0.2 --dropout_prob_cross=0.1 \
      --wm_length=10 --n_prototypes=16 --wavelet=haar --n_layers_webd=2 --n_layers_smc=2 \
      --no_webd_rehearsal --no_debr_decoupling \
      --ortho_weight=0.1 --freq_weight=0.01 --proto_temperature=1.0 \
      --use_decoupled_trace_id --trace_id_gate_bias_init=-2.0 \
      --use_seq_branch --n_layers_seq=2 --seq_fusion_mode=trace_residual_debr \
      --seq_gate_bias_init=-2.0 --seq_add_weight=0.2 --trace_memory_gate_bias_init=-2.0 \
      --history_neg_weight=0.05 --history_neg_num=10 \
      --lr=0.001 --tau=0.05 --cl_weight=0.2 --mlm_weight=0.3 --neg_num=24000 \
      --batch_size=300 --num_workers=8 --learner=AdamW --lr_scheduler_type=constant \
      --warmup_steps=500 --weight_decay=0.0001 --epochs=500 --early_stop=10 \
      --eval_step=1\
      --device=cuda:6