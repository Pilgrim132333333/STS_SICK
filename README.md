# 高效句子语义表示：STS 基线与 GenEOL 推理优化

## 实验背景

本项目研究句子语义相似度（Semantic Textual Similarity, STS）：将两个句子编码为向量，通过余弦相似度预测语义相关程度，并与人工评分比较。数据集为 STS-B 与 SICK-R，主指标为 Spearman，辅助指标为 Pearson。

项目已完成五种公开句向量方法在两个测试集上的 checkpoint 评价。复现范围是加载作者公开权重并统一评价，不包含重新训练原模型。已有 notebook 和结果说明保留在本文后半部分。

后续探索涉及词级匹配、少样本适配与回放、样本选择、排序损失、句法短语匹配。根据已有探索记录，这些具体实现尚未形成足够稳健的新增收益；部分信号在控制实验协议后消失。负结果应保留，不能据此断言相关研究领域整体无效。

当前重心是降低 GenEOL 的时间成本。该方法生成句子变体，再编码、聚合，以额外计算换取表示质量。本项目拟研究生成数量 m 的按需分配，以及推理执行优化。背景见 [GenEOL 原论文](https://aclanthology.org/2025.findings-naacl.122/)，实现参照 [官方代码](https://github.com/raghavlite/GenEOL)。

**研究问题：在预先约定的 STS 质量容忍范围内，自适应生成数量与推理优化能否减少真实端到端耗时，并优于调优后的固定 m 基线？**

## 代码与实验状态

| 内容 | 入口 | 状态 |
| --- | --- | --- |
| 五方法 × 两数据集评价 | `notebooks/` | 远程 main 包含 10 个已执行 notebook |
| 基线与词汇重叠汇总 | `results/` | 远程 main 已有 |
| 适配、选样、损失、短语探索 | 服务器脚本、忽略的 `runs/`、对话实验记录 | 待同步、登记与逐项审计 |
| GenEOL 下一阶段 | [实验方案](docs/GENEOL_EXPERIMENT_PLAN.md) | 待执行，没有本项目 GenEOL 成绩 |
| 代码整理约定 | [代码组织说明](docs/CODE_ORGANIZATION.md) | 尚未完成源代码迁移 |

本次整理读取了远程 main（03941fe）的目录与 README。本地终端和 Node 运行时无法启动，文件工具亦未读到原有 README，因此本文保留的是远程 README 快照；未核验本地工作树、服务器最新文件或未提交改动。以下拟建目录不是已经存在的运行接口。

## 当前运行设想

服务器已实查，现有设备对应的第一轮配置与实验顺序见 [服务器基线实验协议](docs/GENEOL_SERVER_BASELINE.md)。完整 BF16 不作为这台 12GiB 设备的主运行方案。

- 生成器：`mistralai/Mistral-7B-Instruct-v0.1`。
- 向量模型：`mistralai/Mistral-7B-v0.1`。
- 12 GiB 设备先做 4-bit 可运行性验证，两个模型顺序加载；显存允许时另建 BF16 对照。精度配置分别报告。
- 第一轮先固定 m、测质量与耗时，不直接开发自适应算法或自定义算子。

原论文 Table 2 中，此模型组合 m=8 / m=32 的 STS-B 分别为 81.67 / 81.56，SICK-R 为 76.44 / 76.83（Spearman ×100）。这是文献参照，不能写成本项目或 4-bit 成绩。复现须对齐变换组合、提示、池化、归一化和评价划分。[原论文](https://aclanthology.org/2025.findings-naacl.122.pdf)

## 既有基线记录

以下保留已读取的远程 README，说明已完成基线的范围与运行方式。

### STS-B and SICK-R baseline evaluation

This repository contains one standalone notebook for each of five published sentence-embedding methods on each of two English sentence-relatedness test sets. These notebooks **evaluate author-released, pretrained checkpoints**. They do not reproduce the original model-training runs or contain a proposed research method.

## Hardware and setup

- Tested on an NVIDIA RTX 3080 Ti with 12 GiB VRAM; the notebooks use settings intended to fit RTX 3080 variants with 10 or 12 GiB.
- Use Python 3.10–3.12 in a Jupyter environment. The first code cell checks the required packages and installs any that are missing while preserving an existing CUDA-enabled PyTorch installation.
- Run each notebook from top to bottom. Downloads require access to Hugging Face. Models and datasets are cached locally and are not included in Git.
- All ten notebooks were executed on the server and their outputs are committed. The run used Python 3.10.8, PyTorch 2.1.2+cu118, Transformers 4.46.3, and Datasets 3.2.0. On this server, `HF_ENDPOINT=https://hf-mirror.com` was needed because direct access to Hugging Face timed out; model and dataset revision hashes are recorded in the notebooks.

## Evaluation protocol

Each notebook encodes every unique sentence once with its method's official checkpoint, computes cosine similarity for each pair, and reports Spearman and Pearson correlation against the human score. Scores for STS-B (0–5) and SICK-R (1–5) are reported separately. A fixed 32-sentence batch size, dynamic padding, 512-token maximum, seed 42, and full-precision inference are used to fit common RTX 3080 variants. The notebook records model and dataset revision hashes and checks the expected test-set row count.

The STS-B test set has 1,379 pairs from [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts). The SICK-R test set has 4,927 pairs; it uses the original English sentence and score columns from [JSICK's source TSV](https://github.com/verypluming/JSICK/blob/b3034994192fae2f41b5937bcf69544e4282fc39/jsick/jsick.tsv), filtered to `data == "test"`. Each SICK-R notebook fetches that fixed Git commit, so Git must be installed. The MTEB `sickr-sts` collection has all 9,927 SICK pairs under one `test` split and is therefore not used here.

For repeated SICK-R runs, set `SICK_TSV_PATH` to a local copy of that exact JSICK TSV. The notebooks verify its SHA-256 before use.

This common direct-cosine protocol may differ from a paper's original reporting protocol or checkpoint-selection process. The raw paper scores should not be treated as directly reproduced unless these settings have been checked. The [original SentEval toolkit](https://github.com/facebookresearch/SentEval) also offers supervised regressors for STS-B and SICK-R; those are not used here.

## Executed baseline results

Spearman and Pearson are correlations on the specified test sets. The displayed scores, checkpoint hashes, dataset revisions, and evaluation settings are in [`results/baseline_scores.csv`](results/baseline_scores.csv).

| Method | STS-B Spearman | STS-B Pearson | SICK-R Spearman | SICK-R Pearson |
| --- | ---: | ---: | ---: | ---: |
| ESimCSE | 0.801480 | 0.804252 | 0.722837 | 0.804269 |
| DiffCSE | 0.805805 | 0.803306 | 0.712216 | 0.797260 |
| PromptBERT | 0.819899 | 0.818141 | 0.706306 | 0.789052 |
| InfoCSE | 0.819948 | 0.818662 | 0.713615 | 0.792600 |
| WhitenedCSE | 0.798291 | 0.798571 | 0.724995 | 0.796916 |

These are evaluations of released checkpoints under one shared direct-cosine protocol. They are not retraining results or exact reproductions of each paper's original evaluation setup.

## Exploratory lexical overlap check

[`results/lexical_overlap_summary.csv`](results/lexical_overlap_summary.csv) records a descriptive word-set Jaccard check on the same test pairs. Text was Unicode NFKC-normalized and case-folded; English words (including contractions) and numbers were extracted, while punctuation was ignored. Stopwords and negation words were retained. Jaccard is the number of shared unique words divided by the number of unique words across both sentences, so it does not capture word order or repetition frequency. The preselected high-overlap threshold was 0.75. Pair-level data and the exploratory script remain outside Git.

[`results/high_overlap_model_diagnostics.csv`](results/high_overlap_model_diagnostics.csv) records a subsequent exploratory check of all five model predictions on the high-overlap pairs. Each exported pair was matched against the original sentence text and gold score, and the full-set Spearman scores were checked against the executed notebooks. `high_overlap_spearman` is calculated only within pairs with Jaccard at least 0.75; its smaller gold-score range means it should not be read as a direct performance drop relative to the full set. A low gold score means at most half of the dataset's 0–5 (STS-B) or 1–5 (SICK-R) scale: at most 2.5 or 3, respectively. For SICK-R only, `low_gold_detection_auc` measures whether lower cosine scores rank the 52 high-overlap low-gold pairs below the other high-overlap pairs. `low_gold_top_quartile_count` counts how many of those 52 nevertheless fall in the top quarter of model cosine scores within the high-overlap subset. The corresponding STS-B statistics are omitted because only one pair meets its low-score threshold. This is diagnostic analysis, not a sixth formal baseline. Pair-level predictions and selected examples remain outside Git.

## Notebooks and original sources

The follow-up train/development diagnostics are in `results/train_dev_overlap_counts.csv` and `results/train_dev_overlap_metrics.csv`. They use the same word-set Jaccard threshold (0.75), gold-score cutoffs (STS-B <=2.5; SICK-R <=3), and five checkpoint encoders. SICK's `trial` split is treated as development data. All 20 prediction files were checked for identical sentence/gold alignment across models and the rank correlations were independently recalculated from the saved scores. AUC is omitted when fewer than 10 low-gold examples are available: SICK trial has only 2, while STS-B train/validation have 4/1. These exploratory statistics do not establish a causal failure mechanism. Published checkpoints may have been selected using STS-B development scores, so these splits are diagnostic evidence rather than an independent benchmark of checkpoint generalization. Pair-level outputs and the diagnostic script remain in the ignored `runs/` directory.

| Method | STS-B | SICK-R | Paper and implementation |
| --- | --- | --- | --- |
| ESimCSE | `notebooks/ESimCSE_STSB.ipynb` | `notebooks/ESimCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.coling-1.342/) · [Code](https://github.com/caskcsg/sentemb/tree/main/ESimCSE) |
| DiffCSE | `notebooks/DiffCSE_STSB.ipynb` | `notebooks/DiffCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.naacl-main.311/) · [Code](https://github.com/voidism/DiffCSE) |
| PromptBERT, unsupervised | `notebooks/PromptBERT_STSB.ipynb` | `notebooks/PromptBERT_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.emnlp-main.603/) · [Code](https://github.com/kongds/Prompt-BERT) |
| InfoCSE | `notebooks/InfoCSE_STSB.ipynb` | `notebooks/InfoCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.findings-emnlp.223/) · [Code](https://github.com/caskcsg/sentemb/tree/main/InfoCSE) |
| WhitenedCSE | `notebooks/WhitenedCSE_STSB.ipynb` | `notebooks/WhitenedCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2023.acl-long.677/) · [Code](https://github.com/SupstarZh/WhitenedCSE) |

STS-B should cite [Cer et al. (2017)](https://aclanthology.org/S17-2001/); SICK-R should cite [Marelli et al. (2014)](https://aclanthology.org/L14-1314/). Cite the respective original method papers and released checkpoint/code resources in any report.

The coursework's final submission has additional requirements, including an independently developed proposed method, its two notebooks, executed outputs, and appendix evidence. This repository currently covers only the five existing-method evaluation pairs.

