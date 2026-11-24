# 代码整理与产物约定

日期：2026-11-24。源代码迁移尚未执行。

## 已核验范围

已读取远程 main（03941fe）的文件树与 README，确认包含10个基线 notebook及results中的轻量汇总。本轮本地终端和Node运行时无法启动，文件工具未读到原有README；没有核验原本地工作树、AGENTS.md、未提交改动及服务器最新代码。

README的新背景说明保留了远程README快照。文档编辑不代表已完成源代码重构、同步或运行检查。

## 整理原则

1. 保留10个基线notebook的原路径与执行输出。
2. 先做本地/服务器清单、版本与哈希比较，不移动或删除未审阅文件。
3. 每个探索登记模型版本、划分、配置、命令、原始输出位置、结论适用范围。
4. GenEOL使用独立环境，不覆盖旧STS环境。
5. 模型、完整数据、缓存、逐句产物和凭证不进入公开仓库；汇总经审阅后提交。
6. 配置与结果明确区分“论文参照”“本机复现”“待执行设计”。

## 当前已核实远程结构

```text
README.md
notebooks/                 # 10个已执行的五方法×两数据集评价
results/                   # baseline_scores及词汇重叠诊断汇总
.gitignore
```

## 新增文档与待建结构

```text
docs/
  GENEOL_EXPERIMENT_PLAN.md # 本轮实验协议
  CODE_ORGANIZATION.md     # 本文件
configs/geneol/            # 待建：冻结配置
src/geneol_efficiency/     # 待建：数据、生成、编码、聚合、预算与计时
scripts/geneol/            # 待建：smoke、fixed-m、adaptive、profile、report
experiments/archive/      # 待建：审计后的旧探索代码
runs/                     # 本地/服务器缓存与产物，保持忽略
```

以上待建路径没有可运行入口，不能按其名称假设脚本已经存在。

## 旧实验登记清单

下列名称来自已有实验记录，需从服务器确认实际路径与最新版本后再迁移：

| 主题 | 脚本或记录 | 需要保留的协议 |
| --- | --- | --- |
| 适配与回放 | cross_dataset_transfer.py、fewshot_rigorous.py | 归一化标签、单步合并损失、固定回放池 |
| 学习曲线 | data_efficiency_diagnostic.py、learning_curve_earlystop.py | 嵌套样本、停止规则、真实计算量 |
| 样本选择 | sample_selection.py及修正版 | 回归分歧、对称表示、配对随机对照 |
| 排序损失 | loss_control.py及CSV/曲线 | 唯一约束数量、归约、final/best分开 |
| 短语诊断 | phrase_diagnostic.py、审计快照 | 随机跨度边界、多句解析、覆盖与对齐问题 |

短语审计问题未解决前，不把当前结果写成“句法信息没有价值”。控制实验改变多项设置时，也不能仅依据不显著结果断言已确定单一伪影来源。

## 恢复代码访问后的验收

- 读取适用AGENTS.md、检查Git状态，保留既有修改。
- 比较本地与服务器脚本版本，登记来源与哈希。
- 对读写路径和导入关系确认后才迁移，统一配置和日志接口。
- 旧代码保留原始版本；新的共享数据/评价逻辑必须用保存预测复算。
- 新入口执行最小帮助/导入与smoke检查，再扩大实验。
- 检查README链接、Git差异及忽略规则。
- 公开提交前确认无数据、模型、课程PDF或凭证；本轮未提交或推送。

