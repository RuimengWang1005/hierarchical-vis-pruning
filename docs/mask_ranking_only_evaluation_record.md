# Mask-ranking-only 评测结果留档

记录时间：2026-10-15。该模型经训练方式核查后应称为 **Mask-ranking-only 消融模型**，不是只将 BBox 换成 Mask 的完整 Mask-GAZE。原始评测 JSON、逐样本预测、日志和 scorer 权重将在本记录完成后删除；下表是删除前从结果 JSON 直接读取的数值。

## 模型与训练记录

- 初始 scorer SHA256：`e711b7dcb1958343a2e3cd74c14622ee7f3e77d1e03a55f3e4475af9663e009c`
- epoch 1 scorer SHA256：`c09e76fb2f36323ebc407fbb3896d7c1cb782f6f391a273298c002fa9b7c009d`
- 有效训练样本：19,668；epoch：1；optimizer updates：615。
- 每图排序 loss：均值 `0.4126879788`，中位数 `0.3113210946`；跳过样本：0。
- 正式训练耗时：`3415.4563 s`；峰值 CUDA 显存：`2,564,855,808 bytes`。
- 评测使用原 GAZE Focal-Spatial 推理逻辑，预算 10%/20%/30%，spatial ratio 0.1，cosine penalty weight 1.0、threshold 0.7。

## Pilot 200

样本数 200。

| 指标 | 原始 GAZE 10% | Mask 10% | 差值 | 原始 GAZE 20% | Mask 20% | 差值 | 原始 GAZE 30% | Mask 30% | 差值 |
| -- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 主体 Token recall | 0.408583 | 0.475491 | +0.066907 | 0.586996 | 0.636474 | +0.049478 | 0.689411 | 0.736435 | +0.047024 |
| 至少保留一个主体 | 0.875 | 0.900 | +0.025 | 0.955 | 0.955 | 0.000 | 0.980 | 0.970 | -0.010 |
| 可靠背景保留率 | 0.082306 | 0.080600 | -0.001705 | 0.171591 | 0.169125 | -0.002466 | 0.265625 | 0.262841 | -0.002784 |

主体—背景得分差：GAZE `2.344618`，Mask `2.406587`，差值 `+0.061969`。

## FGVC 零样本评测

数值为 accuracy；所有运行的 generation failures 均为 0。

| 数据集 | 样本数 | 10% | 20% | 30% |
| -- | --: | --: | --: | --: |
| CUB-200 / bird-200 | 1,127 | 0.610470 | 0.629104 | 0.644188 |
| FGVC Aircraft | 600 | 0.525000 | 0.593333 | 0.603333 |
| Flowers102 | 6,149 | 0.788096 | 0.816230 | 0.828753 |
| Oxford-IIIT Pet | 3,669 | 0.840283 | 0.875715 | 0.884710 |
| Stanford Cars | 727 | 0.800550 | 0.825309 | 0.836314 |

## 通用数据集评测

| 数据集 | 样本数 | 指标 | 10% | 20% | 30% |
| -- | --: | -- | --: | --: | --: |
| AI2D no mask | 3,088 | exact match | 0.841321 | 0.909974 | 0.916775 |
| ChartQA | 2,500 | relaxed overall | 0.616000 | 0.711200 | 0.760000 |
| ChartQA | 2,500 | human / augmented | 0.4344 / 0.7976 | 0.5400 / 0.8824 | 0.6072 / 0.9128 |
| DocVQA val local | 5,349 | ANLS | 0.815044 | 0.904161 | 0.928960 |
| MME | 2,374 | perception | 1578.4995 | 1660.0950 | 1674.3254 |
| MME | 2,374 | cognition | 613.5714 | 632.1429 | 635.3571 |
| MME | 2,374 | perception+cognition | 2192.0709 | 2292.2379 | 2309.6826 |
| MMMU val | 900 | accuracy | 0.471110 | 0.487780 | 0.494440 |
| OCRBench | 1,000 | accuracy | 0.699000 | 0.789000 | 0.822000 |
| POPE | 9,000 | accuracy | 0.836889 | 0.866111 | 0.866222 |
| POPE | 9,000 | F1 | 0.814553 | 0.851656 | 0.851870 |
| ScienceQA image | 2,017 | exact match | 0.824492 | 未完成 | 未完成 |

ScienceQA 20% 在 Hugging Face 连接中断后没有产生结果 JSON，30% 未运行。早期 `general_eval/mask/budget_0.1`、`0.2`、`0.3` 目录也没有结果 JSON，均不计为完成结果。当前输出中没有对应的原始 GAZE 通用任务重跑 JSON，因此本表不推算或补写 GAZE 差值。

## 清理范围

清理已完成；删除前上述目标合计约 853 MB。本记录保留；评测程序、任务配置、Hugging Face 数据缓存、20K Mask 标注数据及 pilot 200 Mask/标注数据保留。已删除范围为：

- `outputs/mask_vis_20k_online/` 下全部 `fgvc_eval`、`general_eval*` 和 `pilot_200_eval`；
- 正式训练的 `initial_scorer.safetensors`、`scorer_epoch_1.safetensors`、`checkpoint_last.pt`；
- `outputs/mask_vis_20k_dry_run/` 与 `outputs/mask_vis_20k_benchmark_200/` 测试产物及其中 scorer/checkpoint；
- 不删除 `outputs/pilot_200/` 中的 Mask、manifest、Token 审计和可视化数据；
- 不删除较早的 `outputs/mask_gaze_1k/preexperiment/`，因其不属于本轮 20K Mask-ranking-only 正式训练/评测。
