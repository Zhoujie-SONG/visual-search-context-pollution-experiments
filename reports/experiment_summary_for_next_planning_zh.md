# 视觉上下文污染与多轮视觉搜索实验总汇

更新时间：2026-09-03

## 一、研究目标

项目研究视觉语言模型在局部视觉证据不断进入上下文时的行为变化，包含两个互补问题：

1. **受控上下文污染（E1）**：在固定 GT crop 的前提下，人为加入 1、4、8 个无关 crop，测量无关视觉证据对 VQA 准确率的影响。
2. **真实多轮搜索（E2-M）**：允许视觉搜索 agent 自主选择 crop，分析真实轨迹中的目标命中、冗余搜索、误导性观察、终止失败与答案准确率。

主要模型：

- 受控实验：`Qwen/Qwen2.5-VL-7B-Instruct`
- 多轮搜索：`Mini-o3/Mini-o3-7B-v1`
- Crop/答案复核判定器：本地 `Qwen2.5-VL-7B-Instruct`，关键答案经过人工复核

## 二、应作为最终依据的结果

### E1-v2.1p：受控 Context Pollution 最终版

实验设计：

- 数据集：V*Bench，共 238 个样本
- 类别：GPT4V-hard 17、OCR 30、direct_attributes 115、relative_position 76
- 条件：`original_only`、`gt_crop_only`、`gt_plus_1_irrelevant`、`gt_plus_4_irrelevant`、`gt_plus_8_irrelevant`
- 随机种子：42、43、44
- 每个种子的五个条件均独立运行，没有复制 deterministic baseline 行
- 总数据量：238 × 5 × 3 = 3570 行
- 输入方式：原图和 evidence crop 作为独立图像输入，避免 contact-sheet 的布局和缩放混淆
- 解码：`do_sample=False`，`max_new_tokens=64`
- Crop：GT crop 跨种子固定；种子只影响无关 crop 采样及 evidence 顺序
- 极细 crop：短边不足 28 像素时使用边缘复制 padding，不拉伸内容

完整性检查：

- 3570 行全部唯一、完整
- OOM：0
- invalid：0
- preprocessing error：0

总体结果（三个种子先分别计算，再报告 seed-level mean ± std）：

| 条件 | 准确率 | Seed std | 相对 GT-only 下降 |
|---|---:|---:|---:|
| original_only | 74.79% | 0.00 pp | 9.66 pp |
| gt_crop_only | 84.45% | 0.00 pp | 0.00 pp |
| gt_plus_1_irrelevant | 82.91% | 0.24 pp | 1.54 pp |
| gt_plus_4_irrelevant | 81.37% | 0.49 pp | 3.08 pp |
| gt_plus_8_irrelevant | 80.25% | 0.73 pp | 4.20 pp |

以每个样本跨种子的平均表现为 bootstrap 单位：

| 比较 | GT-only 准确率 | 污染条件准确率 | 差值 | 95% CI | 双侧 p |
|---|---:|---:|---:|---:|---:|
| GT-only vs +1 | 84.45% | 82.91% | 1.54 pp | [-0.28, 3.64] pp | 0.1116 |
| GT-only vs +4 | 84.45% | 81.37% | 3.08 pp | [0.70, 5.74] pp | 0.0114 |
| GT-only vs +8 | 84.45% | 80.25% | 4.20 pp | [0.98, 7.56] pp | 0.0136 |

分类别的 GT-only 到 +8 变化：

| 类别 | GT-only | +8 | GT-only 减 +8 |
|---|---:|---:|---:|
| GPT4V-hard | 94.12% | 88.24% | 5.88 pp |
| OCR | 100.00% | 100.00% | 0.00 pp |
| direct_attributes | 86.96% | 78.26% | 8.70 pp |
| relative_position | 72.37% | 73.68% | -1.32 pp |

**可信结论：**正确局部证据能显著提高表现；增加无关视觉证据后，收益逐步被侵蚀。+4 和 +8 的总体下降显著，+1 尚不显著。修复工程问题后，`direct_attributes` 是 +8 下最敏感的类别，`relative_position` 不再表现为最敏感。

### E2-M：VisualNeedle 100 条真实多轮搜索

实验设计：

- 数据集：VisualNeedle 300 EN 中平衡抽取 100 条
- 五类各 20 条：Color、Entity、OCR、Occluded Object、Spatial Relationship
- 模型：Mini-o3-7B-v1
- 最多 12 轮、最多 12 张输入图像
- `do_sample=False`，每轮最多 2048 个新 token，总输出预算 8192 token
- 保留每轮 raw output、bbox、crop、输入/输出 token 和轨迹叠框
- 原始模型输入只使用 `images/`，不使用带 GT 框的 `images_bbox/`

运行结果：

| 指标 | 结果 |
|---|---:|
| 样本数 | 100 |
| 总 crop 数 | 536 |
| 平均 crop 数 | 5.36 |
| 中位 crop 数 | 4 |
| GT 中心命中率 | 62% |
| 达到轮次上限比例 | 47% |
| 最终输出 invalid 比例 | 49% |
| OOM 比例 | 0% |

答案经过宽松语义评分与全部 100 条人工复核：允许自然语言句式、同义词和不冲突的具体化表达，不要求严格字符串匹配。

| 类别 | 正确 | 错误 | 未形成答案 | 语义准确率 |
|---|---:|---:|---:|---:|
| Color Recognition | 2 | 11 | 7 | 10% |
| Entity Recognition | 3 | 9 | 8 | 15% |
| OCR Recognition | 1 | 9 | 10 | 5% |
| Occluded Object Recognition | 0 | 9 | 11 | 0% |
| Spatial Relationship | 1 | 6 | 13 | 5% |
| **总体** | **7** | **44** | **49** | **7%** |

只在 51 条明确给出答案的样本中计算，语义准确率为 7/51 = 13.73%。Plain-VQA anchor 的自动语义准确率为 1%，但未进行与 agent 输出相同强度的全量人工复核，因此只能作辅助参考。

## 三、真实轨迹中的无关 Crop

定义：crop 与最终 GT bbox 的交集面积除以 GT bbox 面积小于 10%，即视为“几何无关 crop”。该定义不是 IoU，也不等价于该 crop 在推理上一定无用。

- 几何无关 crop：347/536 = 64.74%
- 至少包含一个无关 crop 的轨迹：79/100
- 每条轨迹平均无关 crop 数：3.47

对 347 个无关 crop 统一进行内容判定：

| 类型 | 数量 | 占无关 crop |
|---|---:|---:|
| MISLEADING | 47 | 13.54% |
| EXPLORATION | 18 | 5.19% |
| REDUNDANT | 223 | 64.27% |
| NEAR_MISS | 59 | 17.00% |
| 解析失败 | 0 | 0.00% |

定义摘要：

- `MISLEADING`：包含同类错误实例或部分符合约束、可能诱导错误答案的内容。
- `EXPLORATION`：合理的候选区域排查或参照物定位。
- `REDUNDANT`：重复覆盖历史区域，或几乎没有有效信息。
- `NEAR_MISS`：靠近目标或只包含目标局部，但缺失回答所需属性。

与最终结果的观察性关联：

| 轨迹切片 | 样本数 | 语义准确率 |
|---|---:|---:|
| 至少一个几何无关 crop | 79 | 2.53% |
| 没有几何无关 crop | 21 | 23.81% |
| 至少一个 MISLEADING crop | 30 | 0% |
| 至少一个 REDUNDANT crop | 63 | 1.59% |

按最终结果划分的搜索负担：

| 最终结果 | 样本数 | 平均总 crop | 平均无关 crop |
|---|---:|---:|---:|
| 正确 | 7 | 3.29 | 0.43 |
| 错误 | 44 | 3.66 | 2.16 |
| 未形成答案 | 49 | 7.18 | 5.08 |

**可信结论：**Mini-o3 的失败主要表现为搜索不收敛、重复/近失 crop 增多，以及到达限制后仍没有最终答案，而不是 OOM。无关 crop 与低准确率关联强，但当前数据是模型自选择轨迹，困难样本可能同时导致更多 crop 和更低准确率，因此不能直接宣称因果。

## 四、实验演进与不应混用的中间结果

### E1 初始版与 E1.5

初始 E1 使用 contact sheet 和 `max_new_tokens=8`。E1.5 在不重新推理的情况下分析 valid/invalid、答案 flip 和 bootstrap，得到 GT-only 到 +8 约 6.30 pp 的下降，并一度判断 relative_position 最敏感。

这些结果能说明研究方向，但**不应作为最终效应量**，原因由 E1.6 和 E1-v2.1p 后续确认：

- 8-token 生成预算会截断拒答式或说明式输出，抬高 invalid 率。
- contact-sheet 的画布布局和各面板缩放随 crop 数改变，可能形成视觉分辨率/排版混淆。
- 修复后效应仍存在，但最终 +8 降幅为 4.20 pp，类别排序也发生变化。

### E1.6 技术检查

- 初始推理为 greedy decoding，`do_sample=False`。
- 所有 invalid 输出都接近 8-token 上限。
- 输入长度远低于上下文上限，context overflow 或图像 token 截断不太可能。
- Prompt、选项 shuffle 和消息格式跨条件基本一致。
- 结论：初始方向有效，但 invalid/abstention 结论受到生成预算影响。

### E1-v2 smoke

20 条单种子 separate-image smoke 中，GT-only 为 65%，+8 为 45%，下降 20 pp。样本太小且当时 invalid 较多，只用于确认修正协议可运行，不作为正式效应量。

### E2-M V*Bench smoke

- Plain-VQA anchor：15/20 = 75%。
- Agent 仅完成 7 条，7 条均正确，均使用 1–2 次 crop。
- 该结果不完整，且检查到部分 V*Bench 样本存在问题目标在原图中不可见的情况，因此停止扩展并转向 VisualNeedle。
- 不能把 7/7 当作 Mini-o3 的总体准确率。

### VisualNeedle 20 条 smoke

- 五类各 4 条，共 20 条。
- 107 次 crop，GT 命中 14/20，语义正确 2/20，8 条未形成答案。
- 后续 100 条实验包含这 20 条，并采用统一判定器重新分类；最终规划应优先使用 100 条统一结果。
- 早期曾得到 62 个无关 crop 中 MISLEADING 24、EXPLORATION 6、REDUNDANT 27、NEAR_MISS 5。该批分类与后来的统一重判不属于同一判定批次，不能直接比较比例变化。

## 五、工程修复与复现信息

1. E1 环境与 E2-M 环境分离，避免 Transformers 升级破坏 E1 可复现性。
2. Qwen2.5-VL 极端长宽比限制曾导致 VisualNeedle 两条 episode 中断；加入不拉伸内容的边缘复制 padding 后，只恢复缺失样本并完成 100/100 合并。
3. 同样的 padding 后来应用于 crop 分类判定器输入，但没有修改原始 crop 文件。
4. 多 GPU 运行使用 shard-specific 文件，完成后统一合并，避免多个进程同时写一个 CSV。
5. E1-v2.1p 的 bootstrap 使用每个样本跨 seed 的平均值，避免把三个 seed 当成三个独立样本。

## 六、面向下一步规划的核心问题

下一步实验应优先区分以下机制，而不是简单扩大样本数：

1. **因果污染效应**：对同一真实搜索状态，固定此前轨迹，只注入或删除特定类型的 crop，比较答案变化。
2. **搜索策略效应**：加入已访问区域记忆、重复 bbox 抑制、最小尺度限制和覆盖式搜索规划，检验 REDUNDANT 比例是否下降。
3. **终止策略效应**：在固定搜索预算下比较原始终止、置信度终止、强制最终答案和“最后一个有效候选”回退机制。
4. **错误类型效应**：分别注入 MISLEADING、REDUNDANT、NEAR_MISS 和 EXPLORATION crop，测量哪一类最容易引起答案 flip。
5. **难度混淆**：按 GT bbox 面积、目标遮挡、OCR 字体大小、关系链长度和所需搜索轮数进行匹配或分层。
6. **模型泛化**：在至少一个非 Qwen 系模型或不同 agent policy 上复现 E1 与真实搜索结论。
7. **统计设计**：预先定义 primary endpoint；同一 sample 使用 paired design；多 seed 先计算 seed-level 指标；bootstrap 以 sample 为单位。

建议的最小下一步是一个 **E2-Causal Replay**：从 100 条轨迹中选取匹配的正确、错误、未作答样本，冻结原图、问题、历史消息和最终决策点；分别移除无关 crop、只保留 GT 相关 crop、重新加入同数量的四类 crop，然后测量语义准确率、答案 flip 和置信度变化。该设计能把“困难样本更爱搜索”的混淆与 crop 本身的因果影响拆开。

## 七、关键文件

### E1 最终结果

- `outputs/results_context_pollution_full_v21_all_seeds.csv`
- `outputs/summary_context_pollution_full_v21p.csv`
- `outputs/summary_context_pollution_full_v21p_by_category.csv`
- `outputs/bootstrap_context_pollution_full_v21p.csv`
- `outputs/report_context_pollution_full_v21p.md`
- `outputs/accuracy_oom_as_wrong_full_v21p.png`

### E1 技术审计

- `outputs/e15_report.md`
- `outputs/e16_technical_sanity_report.md`
- `outputs/e16_invalid_output_diagnosis.csv`
- `outputs/e16_targeted_rerun_outputs.csv`

### E2-M VisualNeedle

- `/mnt/data2/szj/experiments/e2m_needle_smoke/trajectories.jsonl`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/episode_summary.csv`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/crop_records.csv`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/crop_check_100_per_turn.csv`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/crop_check_100_per_episode.csv`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/crop_and_accuracy_summary_100.csv`
- `/mnt/data2/szj/experiments/e2m_needle_smoke/semantic_rescore_100/semantic_rescore_reviewed.csv`

## 八、给下一位规划模型的口径要求

- E1 的正式效应量只使用 E1-v2.1p，不使用初始 E1/E1.5 的 6.30 pp 作为最终数值。
- E2-M 的主准确率使用人工复核后的 7%，并将 49 条未形成答案单独报告。
- “无关 crop”使用 `intersection(crop, GT) / area(GT) < 0.10`，不要误写成 IoU < 0.10。
- Crop 四分类比例使用 100 条实验统一重判的 347 个结果。
- 真实轨迹中的无关 crop 与错误是相关关系，不应直接写成因果关系。
- Plain-VQA anchor 未进行同等强度的全量人工复核，不宜与 agent 的 7% 作严格模型性能对照。

## 九、可直接交给 ChatGPT 的规划任务

请基于本文档设计下一阶段实验，不要重新解释或覆盖已完成的 E1-v2.1p。规划应至少回答：

1. 如何通过 paired causal replay 区分“crop 本身造成污染”与“困难样本导致更多搜索”的混淆？
2. 如何分别测量 MISLEADING、REDUNDANT、NEAR_MISS、EXPLORATION 四类 crop 的因果影响？
3. 如何设计搜索策略与终止策略的消融，使每个改动只有一个主要变量？
4. 样本如何按类别、目标尺寸、遮挡、OCR 难度、关系链长度和原轨迹结果进行分层或匹配？
5. Primary endpoint、secondary endpoints、配对检验、bootstrap 单位、随机种子和停止规则应如何预注册？
6. 最小 smoke、正式样本量及 GPU 预算分别是多少？
7. 哪些结果足以支持“视觉上下文污染具有因果作用”，哪些只能支持相关性？

请输出：实验假设、实验矩阵、样本构造、干预方法、指标定义、统计方案、工程实现步骤、完整性检查、预期结果解释，以及失败时的最小诊断路径。优先提出一个成本可控但能够改变论文结论可信度的实验，而不是同时铺开很多模型和数据集。
