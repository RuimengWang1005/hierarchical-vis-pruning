# Mask-VIS 与原始 GAZE 训练方式只读核查

核查对象是当前服务器上的真实执行代码与已有运行产物。除本报告外未修改代码、未启动训练、未生成数据。结论以执行路径为准；论文见 [`GAZE_ACMMM26.pdf`](/home/ruimeng/gaze/GAZE_ACMMM26.pdf)，其中式 (2) 和式 (11)–(13) 仅用于和代码对照。

## 一、原始 GAZE 实际如何训练 VIS

真实入口是 [`sft_7b.sh` L38–61](/home/ruimeng/gaze/qwen-vl-finetune/scripts/sft_7b.sh:38)，调用 `train_qwen_selector.py`，加载完整 `Qwen2_5_VLForConditionalGeneration` 并替换视觉塔和整模型的 forward（[`train_qwen_selector.py` L249–330](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/train_qwen_selector.py:249)）。训练数据会把问题和答案编码进 `input_ids/labels`，用户内容被设为 ignore、助手答案保留为 CE 标签（[`data_qwen.py` L120–128](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/data_qwen.py:120)；[`data_qwen.py` L575–624](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/data_qwen.py:575)）。

代码实际总损失为：

\[
L_{GAZE}=L_{CE}+\lambda_t L_{cons}+1.0\,L_{bbox},
\]

其中：

- `L_CE` 是完整冻结语言模型输出上的自回归交叉熵（[`selector_model.py` L1151–1182](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1151)）。
- `L_cons = BCE(soft_foc, hard_foc) + BCE(soft_spa, hard_spa)`（[`selector_model.py` L1188–1195](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1188)）。`λ_t` 从 `0.1` 到 `2.0` 随 `global_step/max_steps` 线性增长（[`train_qwen_selector.py` L90–105](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/train_qwen_selector.py:90)；入口参数见 [`sft_7b.sh` L58–61](/home/ruimeng/gaze/qwen-vl-finetune/scripts/sft_7b.sh:58)）。代码中定义的 `gumbel_start_tau/end_tau` 没有任何调用，因此不存在温度调度（[`argument.py` L30–31](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/argument.py:30)）。
- `L_bbox` 的**真实代码**不是注释所写的 BCE+Dice，也不是论文式 (2) 的“正均值对负均值”单次排序，而是全部正负 Token 的笛卡尔积：`mean softplus(-(s_pos-s_neg-0.2))`（[`bbox_supervision.py` L211–238](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/bbox_supervision.py:211)），再按图平均（[`selector_model.py` L1011–1031](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1011)）。`bbox_loss_weight=1`（[`selector_model.py` L1197–1211](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1197)）。`BCE + Dice` 只存在于未被这条 forward 调用的旧函数/错误注释中。

BBox 不是按像素覆盖率阈值映射；它被直接投影成 spatial-merge 后网格上的并集二值矩形（[`data_qwen.py` L184–264](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/data_qwen.py:184)）。代码将 `mask>0.5` 作为正 Token、其余全部作为负 Token（[`bbox_supervision.py` L228–237](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/bbox_supervision.py:228)），没有 ignore 类。

训练前向确实执行 Focal-Spatial：总预算拆成 focal/spatial，hard 分支执行余弦冗余惩罚，soft 分支执行可微 Top-K（[`selector_model.py` L515–579](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:515)；网格 Spatial 见 [`selector_model.py` L588–636](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:588)）。动态预算取 BBox 二值 Token mask 的前景比例，夹在 `[0.1, 0.6]`，再令 `k=ceil(N×ratio)`（[`selector_model.py` L759–814](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:759)）。需注意：训练代码没有真的缩短序列，而是把 soft mask 乘到所有视觉特征后送入 LLM（[`selector_model.py` L920–922](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:920)）。

视觉编码器、视觉 merger/投影层、语言模型和 `lm_head` 均冻结，只有 `visual.importance_scorer` 可训练（[`train_qwen_selector.py` L176–211](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/train_qwen_selector.py:176)）；优化器只收集 `requires_grad=True` 的参数（[`trainer.py` L370–388](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/trainer.py:370)）。但“冻结 LLM”不等于“不经过 LLM”：`L_CE` 仍经 soft Focal-Spatial mask、冻结 LLM 和 `lm_head` 反传到 VIS。hard Top-K/冗余贪心本身不可微，CE 梯度走 soft mask；hard mask通过 `L_cons` 提供目标（可微 Top-K backward 见 [`selector_model.py` L108–165](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:108)）。

实际配置为 1 epoch、batch 1、梯度累积 32、AdamW、学习率 `5e-5`、warmup 0.03、cosine scheduler（[`sft_7b.sh` L28–83](/home/ruimeng/gaze/qwen-vl-finetune/scripts/sft_7b.sh:28)）；保存的训练状态为 625 updates（[`trainer_state.json` L5–7](/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B/trainer_state.json:5)）。

**论文与代码差异：**论文明确写三项总损失、线性 consistency 权重和训练/推理均使用 Focal-Spatial；这些大方向与代码一致。但论文 `L_bbox` 是“前景均分 − 背景均分”的一次 softplus，代码是全部正负 Token pair 的均值；论文描述保留子序列，训练代码实际为等长 soft-mask 乘法。代码中的 “BCE+Dice” 注释也与实际调用不符。

## 二、当前 Mask-VIS 实际如何训练

入口 [`train_mask_vis_online.py` L9–13](/home/ruimeng/hierarchical-vis-pruning/scripts/train_mask_vis_online.py:9) 只转调 `src.mask_vis_online.main_train`。有效清单仅保留图片、网格和 Mask 三值标签，不保留问题或答案（[`mask_vis_online.py` L153–164](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:153)）。

实际总损失只有：

\[
L_{current}=L_{mask}=\frac{1}{B}\sum_{i=1}^{B}\frac{1}{|P_i||N_i|}
\sum_{p\in P_i,n\in N_i}\operatorname{softplus}(-(s_p-s_n-0.2)).
\]

`label=1` 进入正集合，`label=0` 进入可靠负集合，`label=-1` 完全不进入损失；实现按正样本分块，但累加全部正负 pair，数学上仍是完整笛卡尔积（[`mask_vis_online.py` L286–305](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:286)）。先逐图对全部 pair 求均值，再对 batch 内图片求均值（[`mask_vis_online.py` L484–511](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:484)），不是把全 batch 的 Token pair 混在一起平均。

Mask 标签来自 90% 累计 mask mass 的自适应正 Token，以及 `mask_coverage<=0.01 且 bbox_coverage<=0.01` 的共享可靠背景；其他 Token 为 ignore（[`generate_mask_gaze_20k.py` L394–410](/home/ruimeng/hierarchical-vis-pruning/src/generate_mask_gaze_20k.py:394)；标签赋值见 [`generate_mask_gaze_1k.py` L106–130](/home/ruimeng/hierarchical-vis-pruning/src/generate_mask_gaze_1k.py:106)）。这同时改变了前景几何和负样本语义。

训练只加载 processor、冻结的 Qwen 视觉塔（含 merger）和 VIS scorer；从完整 checkpoint 中只取 `visual.*` 且排除已有 scorer 权重，视觉前向在 `torch.inference_mode()` 下运行（[`mask_vis_online.py` L222–283](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:222)）。不加载语言模型，不输入问题/答案，不计算 `L_CE` 或 `L_cons`，也不在训练循环执行动态预算、Focal-Spatial、冗余惩罚或可微选择。配置中的 `selection` 字段（[`train_mask_vis_20k.yaml` L46–51](/home/ruimeng/hierarchical-vis-pruning/configs/train_mask_vis_20k.yaml:46)）没有被训练循环读取；训练中视觉塔预算被固定为 `1.0`，目的是取出全部 merged features（[`mask_vis_online.py` L254–260](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:254)）。

因此当前“只训练 VIS”同时意味着两件事：**只有 VIS 参数更新，而且只计算 VIS 的 Mask 排序损失**。这不同于原始 GAZE 的“只有 VIS 参数更新，但梯度来自 CE、consistency、BBox ranking 三项”。当前优化器直接接收 `scorer.parameters()`（[`mask_vis_online.py` L420–432](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:420)）。

初始化使用同一 `TransformerScorer` 类和构造参数（[`mask_vis_online.py` L33–46](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:33)；原类见 [`selector_scorer.py` L5–31](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_scorer.py:5)），但服务器没有经过核验的 pre-BBox scorer，实际运行是 `original_random`，哈希 `e711b7dc...e009c`（[`config_snapshot.yaml` L69–74](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/config_snapshot.yaml:69)）。同为 seed 42 只能证明当前构造可复现，不能证明与原始 GAZE 在“加载完整模型之后才新建 scorer”的逐参数初值相同。

实际训练清单为 19,668 张：20,000 中按并集排除 30 个 fallback 和 303 个零可靠负样本，零正样本为 0（[`valid_training_manifest.summary.json` L1–19](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k/training_ready/valid_training_manifest.summary.json:1)）。它也不是原 GAZE 的同一 20K：为去重并排除 pilot，原始清单只有 17,406 张可用，生成阶段从同源 25K pool 补齐且来源比例发生漂移（[`generation_report.md` L18–25](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k/generation_report.md:18)）。已有训练实际为 1 epoch、615 updates（[`run_summary.json` L1–8](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/run_summary.json:1)）。

## 三、逐项差异表

| 项目 | 原始 GAZE 实际实现 | 当前 Mask-VIS 实际实现 | 是否一致 | 证据位置 | 对下游性能的潜在影响 |
| -- | -- | -- | -- | -- | -- |
| 初始化权重 | seed 42 后加载完整 Qwen，再随机新建 scorer | 同构造器、seed 42 直接新建；无已核验 pre-BBox 权重 | 未证实一致 | [`GAZE L242–325`](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/train_qwen_selector.py:242)；[`Mask L394–417`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:394) | 初值差异会混入最终差值 |
| 输入数据 | 原 20K 问答+BBox 清单 | 去重、排 pilot、补池后的 20K，再过滤为 19,668 | 否 | [`GAZE 数据路径`](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/__init__.py:20)；[`Mask 生成报告`](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k/generation_report.md:18) | 数据组成和难度变化是混杂变量 |
| 问题/答案 | 输入并产生答案 labels | 未进入有效清单或训练 | 否 | [`GAZE data L575–624`](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/data_qwen.py:575)；[`Mask L153–164`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:153) | scorer 不再受任务答案约束 |
| 视觉特征 | 完整模型内在线视觉 forward，冻结视觉塔 | 视觉塔在线 `inference_mode`，取相同 merged feature；不缓存 | 特征路径一致，梯度图不同 | [`GAZE L1062–1082`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1062)；[`Mask L268–283`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:268)；[`一致性结果`](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_dry_run/online_consistency.json:15) | 基础特征不是主要差异；后续训练信号是 |
| CE 损失 | 有，答案自回归 CE | 无 | 否 | [`GAZE L1168–1182`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1168)；[`Mask train loop`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:484) | 缺少直接下游任务优化，最可能影响生成/VQA表现 |
| consistency 损失 | soft/hard focal、spatial 两项 BCE；权重 0.1→2.0 | 无 | 否 | [`GAZE L1188–1195`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1188)；[`调度 L90–105`](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/train_qwen_selector.py:90) | 训练分数与离散推理选择更可能失配 |
| 排序损失 | BBox 二值矩形；正×所有其余负；完整 pairwise、逐图平均 | Mask 正×可靠负；ignore 排除；完整 pairwise、逐图/批平均 | 公式骨架一致，集合定义不同 | [`GAZE loss`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/bbox_supervision.py:211)；[`Mask loss`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:286) | 不只是前景更精确，也减少/改变了负监督 |
| 动态预算 | BBox Token 前景比，限制 10%–60% | 训练中无；始终提取 100% feature | 否 | [`GAZE L783–841`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:783)；[`Mask L254–255`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:254) | 未训练跨预算鲁棒性 |
| Focal-Spatial | 训练 forward 实际执行 | 仅推理/一致性检查使用，训练不执行 | 否 | [`GAZE L515–638`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:515)；[`Mask L484–511`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:484) | 训练目标与最终选法脱节 |
| 冗余惩罚 | hard Focal/Spatial 选择实际执行，并通过 consistency 间接监督 | 训练中无 | 否 | [`GAZE L527–579`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:527) | scorer 未适配去冗余后的联合选择 |
| 可微选择 | 自定义可微 Top-K 生成 soft mask | 无选择算子，直接对所有分数排序监督 | 否 | [`GAZE TopK`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:108)；[`Mask loss`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:286) | CE 无法经选择过程约束 scorer |
| 语言模型前向 | 完整冻结 LLM + lm_head 前向 | 不加载 LLM | 否 | [`GAZE L1151–1165`](/home/ruimeng/gaze/qwen-vl-finetune/compression_method/selector_model.py:1151)；[`Mask snapshot L89–92`](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/config_snapshot.yaml:89) | 无法计算任务 CE，也失去语言任务反馈 |
| 优化器 | HF/DeepSpeed AdamW；只含 `requires_grad` 的 VIS | PyTorch AdamW；显式只含 scorer | 更新对象一致，执行栈不同 | [`GAZE trainer`](/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/train/trainer.py:370)；[`Mask L420–426`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:420) | 不是主要差异，但数值轨迹不保证相同 |
| scheduler | HF cosine，warmup ratio 0.03 | 自写 LambdaLR cosine，warmup ratio 0.03 | 名义一致，具体边界未保证一致 | [`GAZE 配置`](/home/ruimeng/gaze/qwen-vl-finetune/scripts/sft_7b.sh:71)；[`Mask L427–432`](/home/ruimeng/hierarchical-vis-pruning/src/mask_vis_online.py:427) | 次要但仍会改变优化轨迹 |
| epoch / 有效 updates | 1 epoch / 625 updates | 1 epoch / 615 updates | 否 | [`GAZE state`](/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B/trainer_state.json:5)；[`Mask summary`](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/run_summary.json:1) | 训练量略少，且每步梯度来源完全不同 |

## 四、为什么会出现这些差异

1. **CE 的删除来自任务约束。**实现该训练器时的要求明确提出“不加载语言模型（除非结构确实无法拆开）”，因此不可能再计算答案 CE。当前文档也明确记录“不加载语言模型”（[`mask_vis_online.md` L3–11](/home/ruimeng/hierarchical-vis-pruning/docs/mask_vis_online.md:3)）。
2. **consistency 的删除并非需求明确要求。**需求指定了 Mask 三值标签和完整 pairwise 公式，但没有明确要求删除 `L_cons`。实现选择了“冻结视觉特征 → scorer → pairwise loss”的独立训练循环；配置虽保留 Focal-Spatial 参数，训练代码没有消费它们。这是实现层面的目标简化。
3. **“只优化 VIS 参数”被落实成了“只优化 VIS，且只算 VIS 排序损失”。**原 GAZE 同样只有 VIS 参数更新，却仍加载完整冻结模型、计算三项损失。因此若把“只训练 VIS”理解为只允许 VIS 参数进入 optimizer，它不要求删除 CE/consistency；当前实现比这一含义更窄。
4. **此前“核心 GAZE 训练逻辑一致”不准确。**准确说法只能是：scorer 网络结构、视觉预处理/merged feature、pairwise 数学形式、主要 optimizer 超参数和推理选择逻辑得到复用；完整训练目标和训练前向并不一致。已有一致性检查只证明同一样本的 feature、score 和 10/20/30% 推理索引相同（[`online_consistency.json` L15–35](/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_dry_run/online_consistency.json:15)），不能证明训练方法一致。
5. 当前已训练 scorer 应命名为 **`Mask-ranking-only` 消融模型**，不应称为“只把 BBox 换成 Mask 的正式 Mask-GAZE”。

## 五、明确结论

**当前 Mask-VIS 并非只把 `BBox监督` 替换成了 `Mask监督`。**除监督区域外，它还删除了问题/答案、完整 LLM 前向、`L_CE`、`L_cons`、训练期动态预算、Focal-Spatial、冗余惩罚和可微选择；并改变了训练样本集合、负样本定义、初始化可比性和 update 数量。

因此，已有 Mask-VIS 低于 GAZE 的下游结果**不能用于判断“Mask监督不如 BBox监督”**；它只能说明当前 `Mask-ranking-only` 训练器得到的 scorer 在这些评测上不如完整 GAZE，因果变量并未隔离。

最小修正范围不是改 VIS 结构，而是复用原 GAZE 的完整训练入口、数据问答编码和模型 forward，保留冻结策略、`L_CE + λ_t L_cons + ranking`、动态预算和 Focal-Spatial，仅将 ranking 的参考标签改为 Mask 的 `1/0/-1`（`-1` 忽略）。三值 rank label 与动态预算用的二值 foreground mask 应分开，避免 `-1` 被错误计入前景比例。若目标是严格判断 Mask 与 BBox 监督孰优，还必须让两组使用同一批样本、同一份已保存的初始 scorer、同一动态预算或预先固定的每图 `k`、同一训练步数与随机顺序；否则仍无法把差异归因给监督区域。
