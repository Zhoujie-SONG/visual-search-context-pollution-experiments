# Round-6 Visual Memory Surgery Devcheck + Smoke Pilot 最终报告

## 第一部分：Devcheck

- N = 20，仅运行 `A_full`。
- `next_output_exact_match` = 100%（仅诊断，不作为门槛）。
- `next_action_match` = 100%。
- `next_bbox_match` = 100%。
- `final_answer_match` = 100%。
- `later_crop_count_match` = 100%。
- `suffix_behavior_match` = 100%。
- 结论：**PASS**。

## 第二部分：Smoke cohort

10 条全部由预注册 seed 自动选出，全部属于 `primary_prefix_hit`、baseline no-answer、至少有 6 个成功 crop，且第 6 个 crop 前 GT coverage 至少一次达到 10%。未人工替换样本。

样本 ID：

```text
TPROMPT08c724f096a1
TPROMPT294532df0b70
TPROMPT2d3a530d79de
TPROMPT312e2d4924ab
TPROMPT4e84c67324b4
TPROMPT59892e5d6946
TPROMPT642beb76b8d9
TPROMPTbbf9cdc186ad
TPROMPTe2ae8879903b
TPROMPTf9789aa0279c
```

类别分布：Entity Recognition 3，Spatial Relationship 3，OCR Recognition 2，Color Recognition 1，Occluded Object Recognition 1。

最大 prefix GT coverage：9/10 为 100%，1/10 为 72.58%；10/10 均至少为 50%。

## 第三部分：五臂结果

| Arm | N | Answer rate | Final accuracy | Acc. among answered | Natural stop | Mean/median additional crops | Mean stop round | Max-round | Max-image | First visual tokens | Mean runtime | OOM | Invalid | Unavailable calls/episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_full | 10 | 0% | 0% | NA | 0% | 5.0 / 5.0 | 12.0 | 100% | 100% | 4098.0 | 39.42s | 0 | 0 | 0 / 0 |
| B_oracle_top2 | 10 | 0% | 0% | NA | 0% | 3.3 / 2.5 | 12.0 | 100% | 40% | 3086.8 | 37.22s | 0 | 0 | 17 / 6 |
| C_recent_top2 | 10 | 0% | 0% | NA | 0% | 3.8 / 5.0 | 12.0 | 100% | 60% | 2661.1 | 32.20s | 0 | 0 | 12 / 4 |
| D_force_answer_r6 | 10 | 0% | 0% | NA | 0% | 0.0 / 0.0 | 7.0 | 0% | 0% | 4098.0 | 6.64s | 0 | 10 | 0 / 0 |
| E_random_top2 | 10 | 0% | 0% | NA | 0% | 3.1 / 3.0 | 12.0 | 100% | 40% | 2762.4 | 32.32s | 0 | 0 | 19 / 6 |

`D_force_answer_r6` 的 10 条输出均继续产生 grounding，因此按预注册优先级全部属于 force-answer 协议违反，未被误判为合法答案。

## 第四部分：B vs A paired rescue

| sample_id | A答 | B答 | A对 | B对 | A自然停 | B自然停 | A新增crop | B新增crop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TPROMPT08c724f096a1 | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 2 |
| TPROMPT294532df0b70 | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 5 |
| TPROMPT2d3a530d79de | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 2 |
| TPROMPT312e2d4924ab | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 5 |
| TPROMPT4e84c67324b4 | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 2 |
| TPROMPT59892e5d6946 | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 5 |
| TPROMPT642beb76b8d9 | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 5 |
| TPROMPTbbf9cdc186ad | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 2 |
| TPROMPTe2ae8879903b | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 3 |
| TPROMPTf9789aa0279c | 否 | 否 | 否 | 否 | 否 | 否 | 5 | 2 |

汇总：answer rescue 0/10，accuracy rescue 0/10，negative answer/accuracy transition 0/10。B 将平均新增 crop 从 5.0 降到 3.3，但没有促成自然停止或答案。

## 第五部分：B vs C / E / D

- B vs C：answer/correct transition 均为 0；B 平均新增 crop 3.3，C 为 3.8。
- B vs E：answer/correct transition 均为 0；B 平均新增 crop 3.3，E 为 3.1。
- B vs D：两者均无合法答案；D 的 10 条全部继续 grounding 并触发协议违反，因此 D 不能作为有效的强制回答 accuracy 对照。
- N=10，仅作工程和描述性观察，不做显著性结论。

## 第六部分：Eviction audit

- B/C/E 第一轮 post-intervention 输入均严格为 3 张图：original image + 2 个 retained prefix crops。
- B/C/E 每条均保留 2 张、驱逐 4 张 prefix crop；active source registry 中不存在被驱逐 source。
- unavailable-source calls：B 17 次、涉及 6/10 条；C 12 次、涉及 4/10 条；E 19 次、涉及 6/10 条。
- 被驱逐 source 的调用均未执行 crop，只返回中性 `source_unavailable`，模型可继续生成。
- 全部 arm 的 intervention 累计 acquired images 均为 7，eviction 未恢复 image budget。
- 全部 arm 的下一 observation index 均从 7 开始，retained source ID 未重编号。
- 原图、crop、assistant textual prefix hash 跨 arm 一致。
- GPU 输入不含 GT answer、GT bbox、category 或 coverage metadata，GT leakage 检查通过。

## 第七部分：Oracle audit

- Oracle top-2 crop mutual IoU：均值 0.2344，中位数 0.1964，范围 0.1201 到 0.6019。
- 高重叠 pair：0/10。
- 两张 retained crop 的 GT coverage 均约 100%：5/10。
- 同时满足两张约 100% coverage 且高重叠：0/10。
- 因此本次 B 无提升不能归因于 oracle pair 普遍高度重复；但 coverage-only oracle 仍不保证保留了回答所需的语义细节。

## 第八部分：Judge

- Model：`gemini-3.8-flash`。
- Required：0。
- YES：0。
- NO：0。
- Pending：0。
- 原因：50 个 episode 均没有 final answer，因此没有“有答案但 normalized mismatch”的样本需要 image-aware judge。

## 第九部分：Integrity

**PASS**。Devcheck 20/20、smoke 50/50、eviction semantics、source identity、预算、GT 隔离、baseline fingerprint 与 judge pending 检查均通过；OOM 为 0。D 的 10 个 invalid 是被正确捕获的预注册 force-answer 协议违反，不是推理崩溃。

## 第十部分：结论

**Ready for formal: NO**

工程实现本身通过了重建与完整性检查，但 10 条样本的五个 arm 都没有产生最终答案，B_oracle 没有出现 answer 或 accuracy rescue。D_force_answer 的 10 条也全部继续 grounding，说明当前强制回答对照在该模型协议下失效。Smoke 暂时只能说明 eviction 机制工作且显著改变了 tool-source 访问和 crop 数，不能支持正式因果效果评估。进入 formal 前应先人工审查这些 no-answer 轨迹与 D 的 prompt/协议预期，但不要在本次冻结 smoke 上换样本或改结果。
