# 算法回归检查与评估参数

本目录维护统一回归检查和评估参数两个入口，不保存已结项的迁移对照、M0 补证、profiler 或性能探针。
运行主机、容器与 Python 环境见 [工具 README](../quad_gradient/README.md)。

## 1. 统一算法回归检查

```bash
python tools/test/check_algorithm.py -v
```

统一入口包含 39 个用例：11 个核心算法、16 个批次搜索、5 个运行时及 7 个解析复用/无哈希用例。
核心算法检查覆盖：

- NLDM 值、右侧 knot 导数、边界外推、普通导数的独立有限差分、loss 符号与 epsilon。
- 手算小图的 setup slew 符号、own delay / downstream slew / upstream load；缓存/冷启动结果一致，
  原生点改变后重新评分；固定 Library 不检测源文件修改，重新加载才读取修改；真实 Library 条件表绑定与带引号/注释的解析。
- `TCSetupObjective` 的并列端点/状态、零/负 slack、ps/ns 换算和 setup slew 种子，缺失/非唯一检查仍报错。
- 显式 `build_timing_model()` 不保存 objective/reference 或传播结果；不同种子的 `backward()` 不污染模型/种子且可重复，原来不需要的 off-union 报告仍不加载。
- AveSmooth 的正/零/负 arrival 活跃集合、阈值与等权传播；它不是普通前向导数，不用 gradcheck 冒充证明。
- 已加载 `DesignView` / `ActionCatalog` 上的无 I/O 候选枚举，合法邻居与缺失 view/保护实例过滤、确定性排序；拒绝保留完整检查点、接受后刷新、无接受不导出；
  工具流程禁止调用 SHA-256，manifest 不再记录输入/源文件身份验证或输出哈希；
  默认流程不完整评估 R0；每批 apply＋route 共用一个进程，再启动一次独立 clean STA＋导图，
  模型复用该目录；严格 A→B 和 B→C post_route 检查均在 clean 之前，保留独立重载网表比较，
  checkpoint 五文件存在性检查仍保留；clean/导图失败或超时不生成反馈，末批模型失败也不提交；
  原生参考拒绝重复/缺失场景与异常 ERC 报告。共享 backend 启动逻辑仍显式设置各操作参数，
  不继承父环境中的旧 action/route/graph 开关；批次直接使用评估结果，不重读 feedback.json，
  最终最佳指标导出仍读取保存的反馈记录。
- 评估参数确定性、无哈希输出、单位列约定、非法值/缺场景/零分母拒绝与禁止覆盖输出。

解析复用检查验证每份 DEF/cells/pins 和库表只解析一次、上下文不可变且不修改公共 checker、
相同路径下修改文件后重新检查。合并事务显式复用中间 B 快照和 A 的库表；不要求 B/C 重复导出库表，
但缺少 A 库表仍必须失败，不会跳过严格修改阶段或路由阶段检查。与原公共规则作差分比较，并继续拒绝连线、受保护实例、
实例增删、pin 语义、重复记录、组件属性、floorplan 与非路由阶段 GCELLGRID 的非法修改。

批次搜索检查以可控物理响应覆盖全局 Top-K 后去重不补位、严格正收益、同分排序、整批推进、
安全退化、完整回滚、最佳状态、K 整数下取整/计数复位/独立 K10 patience、时间和批数预算、
往返冻结与失败恢复。失败试探不计入更新历史；无改善正常结束且不导出初始设计。

运行时检查启动真实 Python 子进程，验证命令超时后的后代进程清理、外层启动器终止整个进程组，
以及分析 worker 遵守剩余时间预算、跨请求只加载一次 Library 和固定 ActionCatalog 且刷新图/评分、正常关闭和错误/取消/超时后的清理。
持久 worker 用例经过真实优化器/子进程边界，使用小图与仿真 LUT，不执行 E2E；显式 Library API 仅检查目录一致，固定运行 worker 不逐批验证。
这些用例使用 Linux `/proc` 检查存活状态。

数值检查使用 CPU float64、1 thread、临时小图和仿真 Runner；不依赖历史大型快照或 legacy 输出。
批次物理响应仍为模拟；这些回归检查不执行或认证 OpenROAD/formal/MCMM。

本次重构在 `gpu_402` 的 `ispd27-contest` 容器验证，证据位于
`.runs/refactor-method-20261025T193656/`：旧版 37 用例通过；扩展后首次 39 用例中
38 个通过，一个搬迁遗漏的 `os` 导入错误修复后定向通过；最终算法与 worker 的 12 用例通过。
同一 AES 原生图的候选/评分/拒绝集合与分析摘要对照完全一致；独立的一动作物理 smoke 通过
apply+route、必要检查、干净 STA 与后续分析。该 smoke 不属于上述 unittest，也不代表完整公开评测或 QoR 提升。

## 2. 从基线生成评估参数

```bash
python tools/test/make_eval_params.py \
  --baseline /path/to/fresh-r0/summary.csv \
  --output .runs/eval-params-NEW/params.json
```

`--config` 默认为公共 `evaluator/config/scoring_config.json`；省略 `--output` 时打印 JSON。
基线必须是 evaluator 格式、同一设计/配置、BC/TC/WC 三角的未优化 R0，不能用候选结果代替。
列名约定 timing 为 ns；脚本检查列名和数值，但无法识别伪装成 ns 的错误单位或伪装成 R0 的候选。

输出包括 TC 参考、归一化分母、baseline loss、固定算法 epsilon/改善阈值及评分配置，不再输出输入 SHA-256。
不记录时间戳、私人路径，不拟合阈值/容差，不编造 runtime 或物理认证。
算法 epsilon 为 **1e-9 ns**，与公共评分 epsilon **1e-6 ns** 分开保留。
`algorithm.reference_tc_ns` 对应分析器 JSON 请求的 `reference: [TNS, WNS]` 字段；这是离线准备数据，
不覆盖优化器每次运行自行取得的初始原生 STA 参考或接受策略。
真实 E2E 使用公共 `evaluator/run_submission.py`，不在本目录维护实验驱动。

真实物理合法性和优化收益仍使用现有 [evaluator](../../evaluator/README.md)，不在这里重建评估框架。
公共 [results](../../results/README.md) 只保存必要指标与分析，原始运行输出放在新的运行目录。
