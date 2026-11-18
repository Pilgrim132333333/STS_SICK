# STS-B and SICK-R baseline evaluation

This repository contains one standalone notebook for each of five published sentence-embedding methods on each of two English sentence-relatedness test sets. These notebooks **evaluate author-released, pretrained checkpoints**. They do not reproduce the original model-training runs or contain a proposed research method.

## Hardware and setup

- Target: NVIDIA RTX 3080 with CUDA-enabled PyTorch, 10 or 12 GiB VRAM.
- Use Python 3.10–3.12 in a Jupyter environment. The first code cell checks the required packages and installs any that are missing while preserving an existing CUDA-enabled PyTorch installation.
- Run each notebook from top to bottom. Downloads require access to Hugging Face. Models and datasets are cached locally and are not included in Git.
- The notebooks have **not** been executed in this repository: the authoring machine has no NVIDIA GPU. Run them on the RTX 3080 and retain the resulting notebook outputs before using results in a paper.

## Evaluation protocol

Each notebook encodes every unique sentence once with its method's official checkpoint, computes cosine similarity for each pair, and reports Spearman and Pearson correlation against the human score. Scores for STS-B (0–5) and SICK-R (1–5) are reported separately. A fixed 32-sentence batch size, dynamic padding, 512-token maximum, seed 42, and full-precision inference are used to fit common RTX 3080 variants. The notebook records model and dataset revision hashes and checks the expected test-set row count.

The STS-B test set has 1,379 pairs from [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts). The SICK-R test set has 4,927 pairs; it uses the original English sentence and score columns from [JSICK's source TSV](https://github.com/verypluming/JSICK/blob/b3034994192fae2f41b5937bcf69544e4282fc39/jsick/jsick.tsv), filtered to `data == "test"`. Each SICK-R notebook fetches that fixed Git commit, so Git must be installed. The MTEB `sickr-sts` collection has all 9,927 SICK pairs under one `test` split and is therefore not used here.

This common direct-cosine protocol may differ from a paper's original reporting protocol or checkpoint-selection process. The raw paper scores should not be treated as directly reproduced unless these settings have been checked. The [original SentEval toolkit](https://github.com/facebookresearch/SentEval) also offers supervised regressors for STS-B and SICK-R; those are not used here.

## Notebooks and original sources

| Method | STS-B | SICK-R | Paper and implementation |
| --- | --- | --- | --- |
| ESimCSE | `notebooks/ESimCSE_STSB.ipynb` | `notebooks/ESimCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.coling-1.342/) · [Code](https://github.com/caskcsg/sentemb/tree/main/ESimCSE) |
| DiffCSE | `notebooks/DiffCSE_STSB.ipynb` | `notebooks/DiffCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.naacl-main.311/) · [Code](https://github.com/voidism/DiffCSE) |
| PromptBERT, unsupervised | `notebooks/PromptBERT_STSB.ipynb` | `notebooks/PromptBERT_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.emnlp-main.603/) · [Code](https://github.com/kongds/Prompt-BERT) |
| InfoCSE | `notebooks/InfoCSE_STSB.ipynb` | `notebooks/InfoCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2022.findings-emnlp.223/) · [Code](https://github.com/caskcsg/sentemb/tree/main/InfoCSE) |
| WhitenedCSE | `notebooks/WhitenedCSE_STSB.ipynb` | `notebooks/WhitenedCSE_SICKR.ipynb` | [Paper](https://aclanthology.org/2023.acl-long.677/) · [Code](https://github.com/SupstarZh/WhitenedCSE) |

STS-B should cite [Cer et al. (2017)](https://aclanthology.org/S17-2001/); SICK-R should cite [Marelli et al. (2014)](https://aclanthology.org/L14-1314/). Cite the respective original method papers and released checkpoint/code resources in any report.

The coursework's final submission has additional requirements, including an independently developed proposed method, its two notebooks, executed outputs, and appendix evidence. This repository currently covers only the five existing-method evaluation pairs.
