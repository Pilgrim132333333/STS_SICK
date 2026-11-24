# RTX 3080 Ti 12GiB：GenEOL 基线实验协议

日期：2026-11-24。状态：已核实服务器资源；Mistral量化模型尚未加载测试。本文替代此前关于“优先租用更大显存跑BF16”的运行建议，本项目按现有服务器实施。

## 一、实际检查结果

通过SSH读取硬件、cgroup限制、包版本、已有脚本与日志；执行了CUDA BF16 512×512矩阵乘法，结果有限且无异常。没有安装环境、下载Mistral权重、修改服务器脚本或启动完整实验。

| 资源 | 实测 |
| --- | --- |
| GPU | 单卡 NVIDIA GeForce RTX 3080 Ti，12288MiB，计算能力8.6 |
| 检查时GPU状态 | 利用率0%，无计算进程 |
| CPU | Xeon Silver 4214R；宿主48逻辑CPU可见，但cgroup配额为12核等效算力 |
| 内存 | cgroup限制96636764160字节，即90GiB；宿主free不是实例额度 |
| 系统盘 | 30G，剩余约1.4G |
| 数据盘 | 200G，剩余约58G |
| 驱动 | 580.95.05 |
| 现有Python | 3.10.8 |
| PyTorch | 2.1.2+cu118，实际从/root/miniconda3下加载 |
| Transformers / Datasets | 4.46.3 / 3.2.0 |
| accelerate / bitsandbytes / vLLM / flash-attn | 现有STS环境均未安装 |
| CUDA检查 | 可用；BF16矩阵乘法通过 |

7B的完整BF16权重超过12GiB容量。主实验使用4-bit权重、BF16计算；这不等于BF16权重复现。CPU offload不作为主速度基线，以免将卸载代价混入预算研究。

## 二、已有GenEOL命名脚本的真实范围

服务器runs/geneol_generate.py：Qwen2.5-1.5B-Instruct，FP16，单一paraphrase提示，m=1/2/4，最长128个新token。

runs/geneol_eval.py：InfoCSE CLS编码，再做原句与改写聚合。没有原文的Mistral编码器、四类变换与composition。

因此保存为“改写聚合适配探索”，不覆盖，也不称完整GenEOL基线。现有dev日志全量启动因CUDA Error304退出；当前CUDA复查通过，不能断定历史错误原因或承诺不会复现。dev JSON实际上仅4句、3种子，并非3834句完整结果。

现有token统计使用新token张量numel，包含padding，不是实际EOS前输出token数。现有50句计时每档只有一次，未形成完整流水线、预热或重复计时协议，不能据此判断m=4比m=1更快。

## 三、锁定主基线 B8-NF4

| 项目 | 固定设置 |
| --- | --- |
| 生成器 | mistralai/Mistral-7B-Instruct-v0.1，revision在下载时记录并锁定 |
| 编码器 | mistralai/Mistral-7B-v0.1，同样锁定revision |
| 权重 | 两个模型分别4-bit NF4，double quant启用 |
| 运算 | bnb_4bit_compute_dtype=bfloat16；非量化模块dtype=bfloat16 |
| 框架 | Transformers 4.46.3；先用HF实现，不将vLLM作为初始依赖 |
| 模型驻留 | 生成全部变体后结束生成进程，再启动编码进程；同一时刻只驻留一个7B |
| attention | 优先显式SDPA，记录实际后端；不额外安装flash-attn作为前提 |
| 提示/变换 | 官方四类变换，开启官方composition；保存原文提示与配置哈希 |
| 采样 | do_sample=True，temperature=0.7，top_p=1，repetition_penalty=1 |
| m | 主基线8；最终聚合不含中间composition文本，原句另计 |
| 编码 | 官方KEEOL提示、末层及官方suffix pooling mask；FP32池化与聚合 |
| 归一化 | 本项目固定每条向量L2归一化后均值聚合，最终L2归一化；这是显式运行设置，需核对作者命令是否一致才能称精确复现 |
| 训练 | 无训练、无projection、无replay |
| seeds | 42、43、44；固定输入顺序、分配规则及每次运行manifest |

量化方法依据[Transformers官方文档](https://huggingface.co/docs/transformers/v4.46.0/quantization/bitsandbytes)。提示、composition和pooling以[GenEOL官方代码](https://github.com/raghavlite/GenEOL)为准；论文参照见[原论文](https://aclanthology.org/2025.findings-naacl.122.pdf)。

长度语义保持官方eval默认max_length=1024及代码中的倍数：生成总长度上限3072token（含提示），编码上限4096token。不得悄悄改成max_new_tokens=128。记录每条实际输入、输出、截断和EOS；短句通常提前EOS，但不能提前假定耗时。

生成初始仅1个提示×1个返回序列；逐类收集变体，避免m=32同时展开32个返回序列。composition单独排队。编码初始batch=1、use_cache=False、只取所需末层，避免保存全部层状态。生成阶段use_cache=True。

这些是启动设置，不是已实测的Mistral吞吐/峰值显存。若batch=1仍失败，先定位，不自动offload或换模型继续混报。

## 四、环境与空间

新环境、模型、HF缓存、pip缓存、临时目录全部放/root/autodl-tmp，系统盘只剩1.4G，不承载大下载。

保留原STS环境。新环境先沿用torch2.1.2+cu118、transformers4.46.3，候选依赖accelerate0.34.2、bitsandbytes0.44.1；它们尚未在此服务器联合验证，安装后先检查导入、NF4运算、Mistral加载和小样本。实际成功环境freeze为准，不能称现在已可运行。

两套原始Mistral权重下载预计约29GB，另留环境、缓存和输出空间；这是估计，下载前核对具体文件尺寸。仅下载一种权重格式，避免同时缓存bin/safetensors、多个revision或量化副本。开始下载前保留至少40GiB可用空间；不删除其他项目。

## 五、先测可行性，再锁batch

阶段A：64个训练句子，STS-B与SICK各32句，按长度分层，seed42。不使用开发集来调batch或长度。

1. 1提示×1返回、编码batch1，跑通m=4和8、composition与聚合，检查NaN/OOM、文本及pooling边界。
2. 生成提示batch候选1/2/4；编码batch候选1/2/4，num_return_sequences固定1。
3. 使用训练句子的长度压力样例，核对峰值allocated/reserved及nvidia-smi，初始目标是总使用量不超过约10.5GiB。
4. 选择通过压力样例的吞吐较好设置，冻结后对全部m使用相同基础执行策略。若正式数据OOM，运行判为失败并记录；统一修订配置、重跑受影响计时，不悄悄缩batch掩盖差异。
5. 另审阅固定数量的输出，检查否定、数量、角色和composition漂移；不按人工相似度标签筛掉难例。

不把10.5GiB写为预测峰值或适用于任意输入的保证。

## 六、实验矩阵与规模

| 配置 | m | 用途 |
| --- | ---: | --- |
| B0 | 0 | 同一Mistral编码器/提示/pooling，只编码原句 |
| B4 | 4 | 四种变换各1条 |
| B8 | 8 | 四种变换各2条，主要复现基线 |
| B16 | 16 | 各4条，观察边际收益 |
| B32 | 32 | 各8条，高预算参照 |

开发集已实际读取核对：STS-B validation为1500句对、2910唯一句子；SICK trial为500句对、924唯一句子；合计3834。两个数据集分别评价，不使用当前test选择配置。

一份种子下，最终待编码文本槽位（尚未做重复文本去重）：
- B0：3834。
- B4：19170。
- B8：34506。
- B16：65178。
- B32：126522。

m=32、3种子的最终变体数为3834×32×3=368064；加原句的编码槽位为379566。若官方composition每个变体再生成一次，中间与最终生成请求槽位合计736128，不能忽略一半生成工作。具体请求数须与官方composition实现核对，重试另计。

第一轮先完成B0、B4、B8、seed42。确认质量和成本合理后，扩展B16/B32；最后补seed43/44。若单种子B8已经超出可接受时间，先报告瓶颈和预算，不预先启动全部大规模组合。

## 七、质量与计时分开实施

质量曲线可使用变换平衡的m32缓存前缀获得嵌套m，但只用于质量比较。独立执行小m的实际输出可能因RNG消耗和batch不同而改变；不要宣称缓存前缀就是其真实生成结果。

实际成本必须独立执行每档，从任务生成/向量缓存为空开始；模型权重已在磁盘，不计网络下载。保存真实完成的预算与重试。

为控制成本，先在训练集选定128个长度分层句子作为固定计时集：
- 每个m独立执行生成、composition、模型切换、编码、聚合。
- 使用正式冻结的batch。
- 做不计入测量的独立预热，进程加载/首调用额外单列。
- 至少3次计时，轮换m的顺序，固定负载、随机种子与线程数；CPU线程初始4、数据加载workers=0。
- 用perf_counter和必要CUDA同步计时；不能只测异步kernel发射。
- 冷启动任务总时长和预热后阶段吞吐分开。两个模型不同时驻留，因此跨阶段切换始终计入任务总时长。
- 完整开发集额外保存至少seed42实际B8总耗时，不能将128句线性外推称为实测全量时间。

输出token按每序列首个EOS前的有效长度统计，输入token排除padding；可另记计算槽位数，不能混用。

## 八、输出与结论规则

每数据集报告Spearman/Pearson及3种子均值、标准差；保存逐句对预测，以配对句对bootstrap比较B4/B8/B16与B32，注明共享句子的依赖和对生成种子的条件性。不把“不显著”当等价。

成本报告：加载、生成、composition、卸载/切换、编码、聚合评价及端到端；实际token、文本数、重试、截断、失败、显存峰值。

自适应阶段的预设目标保持：相对同精度B32，两个开发集各自Spearman单侧95%差异下界高于-0.005；同时真实端到端至少快20%。这是项目验收目标，不是首轮必须得到的结果。还必须比较满足同质量门槛的最便宜固定m，否则只比B32可能夸大贡献。

先回答：
1. 4-bit完整GenEOL是否可在现有设备稳定运行？
2. 相比B0，生成确有质量收益吗？
3. 额外m的收益是否值得实际时间？
4. 瓶颈在哪里？

本轮不安装vLLM、不开发自适应或算子。固定基线成立后，再用兼容的调优引擎建立强对照，并研究自有改进。Qwen+InfoCSE结果作为另一套适配实验单独保留。

## 九、结果目录约定

独立使用runs/geneol_mistral_nf4/，不覆盖runs/geneol/。每次运行保存config/environment/manifest、metrics、stage_timings、生成中间与最终产物、句对预测、日志及模型revision。生成文本与向量留服务器，轻量报告审阅后入库。

