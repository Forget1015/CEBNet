# 超参数分析日志整理

**实验日期**: 主要在 Jul-16-2026 和 Jul-17-2026

**超参数范围**:
- wm_length: {0.1, 0.2, 0.3, 0.5}
- n_prototypes: {4, 8, 16, 32}

---

## Industrial_and_Scientific

### wm_length 分析（固定 n_prototypes=16）

| wm_length | 日志文件（推荐使用最新的） |
|-----------|--------------------------|
| **0.1** | `Jul-16-2026_03-02-ad2425_wm0.1_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.2** | `Jul-16-2026_03-07-dd0264_wm0.5_K16_...norehearsal.log` 或其他 wm10 的 Jul-16 日志 |
| **0.3** | `Jul-16-2026_03-03-e72d7c_wm0.3_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.5** | `Jul-16-2026_03-06-d57210_wm0.5_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

### n_prototypes 分析（固定 wm_length=0.2）

| n_prototypes | 日志文件（推荐使用最新的） |
|--------------|--------------------------|
| **4** | `Jul-16-2026_03-08-591b70_wm0.2_K4_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **8** | `Jul-16-2026_03-08-0ff159_wm0.2_K8_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **16** | `Jul-16-2026_03-07-dd0264_wm0.5_K16_...` 或其他 Jul-16 的 wm10+K16 日志 |
| **32** | `Jul-16-2026_03-09-d0d1cf_wm0.2_K32_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

---

## Video_Games

### wm_length 分析（固定 n_prototypes=16）

| wm_length | 日志文件（推荐使用最新的） |
|-----------|--------------------------|
| **0.1** | `Jul-16-2026_16-28-81bce8_wm0.1_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.2** | `Jul-16-2026_02-47-fcc1af_wm10.0_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.3** | `Jul-16-2026_16-28-7a9b44_wm0.3_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.5** | `Jul-17-2026_02-17-d79676_wm0.5_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

### n_prototypes 分析（固定 wm_length=0.2）

| n_prototypes | 日志文件（推荐使用最新的） |
|--------------|--------------------------|
| **4** | `Jul-17-2026_02-18-bf6e47_wm0.2_K4_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **8** | `Jul-17-2026_02-20-4de346_wm0.2_K8_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **16** | `Jul-16-2026_02-47-fcc1af_wm10.0_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **32** | `Jul-17-2026_02-23-f8b414_wm0.2_K32_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

---

## ML-1M_TedRec

**注意**: ML-1M 的 wm_default=0.4（即 wm_length=20，因为 max_his_len=50）

### wm_length 分析（固定 n_prototypes=16）

| wm_length | 日志文件（推荐使用最新的） |
|-----------|--------------------------|
| **0.1** | `Jul-16-2026_06-21-af3f11_wm0.1_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.2** | `Jul-16-2026_06-22-634502_wm0.2_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.3** | `Jul-16-2026_06-29-c46689_wm0.3_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **0.5** | `Jul-16-2026_06-30-a204a0_wm0.5_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

### n_prototypes 分析（固定 wm_length=0.4）

| n_prototypes | 日志文件（推荐使用最新的） |
|--------------|--------------------------|
| **4** | `Jul-16-2026_06-32-e33de0_wm0.4_K4_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **8** | `Jul-16-2026_08-12-4b4210_wm0.4_K8_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **16** | `Jul-16-2026_08-12-689d3d_wm0.4_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |
| **32** | `Jul-16-2026_09-00-027442_wm0.4_K32_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05_norehearsal.log` |

---

## 快速索引命令

### 提取所有超参数日志到单独目录

```bash
# Industrial_and_Scientific
mkdir -p hyperparam_logs/Industrial_and_Scientific
cp logs/Industrial_and_Scientific/Jul-16-2026_03-02-ad2425_*.log hyperparam_logs/Industrial_and_Scientific/  # wm=0.1
cp logs/Industrial_and_Scientific/Jul-16-2026_03-03-e72d7c_*.log hyperparam_logs/Industrial_and_Scientific/  # wm=0.3
cp logs/Industrial_and_Scientific/Jul-16-2026_03-06-d57210_*.log hyperparam_logs/Industrial_and_Scientific/  # wm=0.5
cp logs/Industrial_and_Scientific/Jul-16-2026_03-08-591b70_*.log hyperparam_logs/Industrial_and_Scientific/  # K=4
cp logs/Industrial_and_Scientific/Jul-16-2026_03-08-0ff159_*.log hyperparam_logs/Industrial_and_Scientific/  # K=8
cp logs/Industrial_and_Scientific/Jul-16-2026_03-09-d0d1cf_*.log hyperparam_logs/Industrial_and_Scientific/  # K=32

# Video_Games  
mkdir -p hyperparam_logs/Video_Games
cp logs/Video_Games/Jul-16-2026_16-28-81bce8_*.log hyperparam_logs/Video_Games/  # wm=0.1
cp logs/Video_Games/Jul-16-2026_16-28-7a9b44_*.log hyperparam_logs/Video_Games/  # wm=0.3
cp logs/Video_Games/Jul-17-2026_02-17-d79676_*.log hyperparam_logs/Video_Games/  # wm=0.5
cp logs/Video_Games/Jul-17-2026_02-18-bf6e47_*.log hyperparam_logs/Video_Games/  # K=4
cp logs/Video_Games/Jul-17-2026_02-20-4de346_*.log hyperparam_logs/Video_Games/  # K=8
cp logs/Video_Games/Jul-17-2026_02-23-f8b414_*.log hyperparam_logs/Video_Games/  # K=32

# ML-1M_TedRec
mkdir -p hyperparam_logs/ML-1M_TedRec
cp logs/ML-1M_TedRec/Jul-16-2026_06-21-af3f11_*.log hyperparam_logs/ML-1M_TedRec/  # wm=0.1
cp logs/ML-1M_TedRec/Jul-16-2026_06-22-634502_*.log hyperparam_logs/ML-1M_TedRec/  # wm=0.2
cp logs/ML-1M_TedRec/Jul-16-2026_06-29-c46689_*.log hyperparam_logs/ML-1M_TedRec/  # wm=0.3
cp logs/ML-1M_TedRec/Jul-16-2026_06-30-a204a0_*.log hyperparam_logs/ML-1M_TedRec/  # wm=0.5
cp logs/ML-1M_TedRec/Jul-16-2026_06-32-e33de0_*.log hyperparam_logs/ML-1M_TedRec/  # K=4
cp logs/ML-1M_TedRec/Jul-16-2026_08-12-4b4210_*.log hyperparam_logs/ML-1M_TedRec/  # K=8
cp logs/ML-1M_TedRec/Jul-16-2026_08-12-689d3d_*.log hyperparam_logs/ML-1M_TedRec/  # K=16
cp logs/ML-1M_TedRec/Jul-16-2026_09-00-027442_*.log hyperparam_logs/ML-1M_TedRec/  # K=32
```

---

## 说明

1. **推荐使用的日志**: 都是 Jul-16 或 Jul-17 开头的，这些是最新跑的超参数分析实验
2. **参数对应**:
   - `wm10` 或 `wm10.0` = wm_length=0.2 (10/50)
   - `wm0.1` = wm_length=0.1
   - `wm20` 或 `wm20.0` = wm_length=0.4 (20/50，ML-1M 默认)
3. **所有日志都包含** `norehearsal` 标记，符合大纲架构要求
4. **避免使用**: 包含 `nowebd`、`wm1.0`、`wm50` 等消融标记的日志，那些是消融实验
