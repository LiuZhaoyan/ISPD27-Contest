# ISPD27 Quad-Gradient：分阶段实施规格

目标：在已接受 M0 的基础上完成 M1–M4，把论文 setup/timing 分支迁移为可运行、可验证的 AES / Functional-TC sizing 原型。
本文压缩已确认的 Q1–Q15 决策；论文原式见 [algorithm_spec.md](algorithm_spec.md)，
接口入口见 [integration_map.md](integration_map.md)。

**实施状态以 [M0 results](../../results/quad_gradient/m0_results.md) 为准：**
只读 M0 及 closure 已执行，局部 setup/load 值检查通过，合法 tagged relation 仍未通过。
尚无 M1–M4、梯度正确性、优化收益或成功参赛提交证据。**项目决策：Timing Tag 的完整合法前驱/edge/check 兼容性验证不再阻塞 M1–M4 开发。** 可基于当前已验证的接口继续构建并实测优化原型；缺失的 Tag 证明仍是模型风险，不得宣称已经验证。

**后续执行合同：**
- 实施任务从 M1 开始，按 M1→M2→M3→M4 连续推进。阶段门用于检查实际结果，不是逐阶段等待用户再次授权的停止点；正常实现、验证及范围内修复重试属于后续工作。
- 不要求先重跑、补齐或改绿历史 M0 checker。M0 已完成的项目状态与未证明的 Tag 兼容性风险分别保留；不能伪造历史检查通过。
- “完成后续所有计划步骤”指完成本规格内 M1–M4 及下述最终交付，不包含第 5 节明确排除的扩展。阶段验证失败时先定位、修复和重试，不静默跳过；若证据否定当前方法或必须改内核、换后端、扩大算法范围，再报告依据并形成路线判断。

## 1. 环境与参考状态

- 按 [AGENTS.md](../../AGENTS.md) 在 `gpu_402` / `ispd27-contest` /
  `udxs/ispd27:v1` 执行。保持公共 evaluator、评分、库、SDC、RC、模式和 propagated clocks 原义。
- 复用参考状态适配、合法动作、局部模型/平滑反传、单步更新/回退四项职责；无需通用框架。
- OpenSTA 提供准确前向。每条状态保留 scene/mode、max/min、rise/fall、arc identity、
  sense、条件及必要 clock/path/check context。结构连通、胜出路径或单 pin AT 标量不足以证明合法 tagged graph。
- 原生报告核对单位；TNS 先归约每 endpoint 的合法 transition/path/tag/check 最差 slack，
  每端点只累加一次负 slack；WNS 取最差。重建全部 BC/TC/WC setup/hold 汇总是接口门。

## 2. 合法动作与替换

候选 = 官方 CSV 等价组 ∩ 所需 Liberty/LEF 视图 ∩ 目标 master 政策 ∩ 实例保护/接口功能合同。
已有禁用 master 可原样保留，新换入禁用目标必须拒绝；保留 protected clock cells/sinks、
宏、physical-only、fixed 对象及固定物理资源。

固定另一坐标选最近合法 drive/Vth 邻居，可跨禁用档位；无目标则 invalid，记录拒绝原因。
Vth 用同基名的物理阈值顺序；BUF 与 BUF-f、INV 与 CKINVDC 等特殊结构先分序列，
不凭数字强行排序，不补造 cell 或假设等距物理步长。

M2 先验证已核对同 footprint 的 VT swap，再验证 drive 更换和必要 legalization。
必须检查 STA/OpenDB master、成对 DEF/Verilog 导出及重载、formal、clock snapshot、
placement、位移和完整评测。候选在库中合法不证明当前位置容得下；`replace_cell` 接口存在不证明落盘可靠。

## 3. Loss、反传与单步接受

初版最小化下式，时间统一 ns，R0 分母在一次实验中固定；权重、epsilon 和 provisional
评分政策随运行记录，当前 setup 两项权重均为 1：

\[
L_{TC}=\frac{-TNS_{TC}}{|TNS_{TC,R0}+\epsilon|}
+\frac{\max(0,-WNS^{raw}_{TC})}{|WNS_{TC,R0}+\epsilon|}.
\]

epsilon 在绝对值内先加；正 slack 后 WNS 项为零。这是论文目标适配，不是完整 contest loss。
Hold/power 不进入首版 setup 梯度，但参与最终验收。

- AveSmooth 全局反传包含 own delay、upstream load、downstream slew。
  端点 seed 包含 data slew 对 setup constraint/RAT 的作用，绑定真实 data/clock transition 和表；
  当前参考内固定 clock slew，刷新后重读，constraint 的 data-slew 导数不预设正号。
- 当前与候选在同一当前 input slew/output load 下查询 NLDM，pin cap 用真实候选差值。
  一次评分固定当前 Jacobian，由链式项预测一阶环境变化，禁止重算候选环境后重复叠加。
- 四方向统一 candidate−current，`benefit = -predicted_delta_loss`，仅正收益候选入选。
  不按上/下方向再翻符号，不加未经定义的步长分母；这是解决原文 Eq. (10)–(11) 歧义的工程选择。
- K=1：真实 STA 的 TC loss 改善超过已确定数值噪声，且全部护栏通过才暂收。
  每次接受后刷新参考状态与梯度，不沿用旧 pin cap、driver 模型或 Jacobian。
- 全 corner hold **评分罚分**不高于各自 R0（容忍区间内 raw hold 变化可接受）；
  基线零 ERC 不新增，WC slew 违例总量不高于 R0，max/total overflow 为零。
  等价、保护对象、clock integrity 和 placement 必须通过。
- 当前 wire RC 下的暂收是代理结果；master 几何或合法化位置变化后，旧 RC 不证明物理验收。
  阶段/最终输出前重新路由、寄生及 fresh MCMM 全检查，并刷新参考/梯度。
  失败恢复最近物理验收状态的设计、时序和寄生引用，不能只还原 master。

## 4. 阶段门与可核验条件

| 阶段 | 最小工作与通过条件 |
| --- | --- |
| M0 | 只读基线、全部场景汇总、单位/master、path/arc、setup check、load 工作点；**阶段已人工接受**，原自动 checker 仍为 blocked，Tag 兼容性验证暂缓，不阻塞后续开发 |
| M1 | 最近合法 drive/VT 邻接、保护/禁用过滤，每候选有真实视图与政策依据，invalid 有原因 |
| M2 | VT 和 drive 分别证明替换/落盘/重载/合法化及完整验收 |
| M3 | 在明确当前 Tag/模型近似假设的前提下实现反传与预测；验证局部导数、setup seed、收益符号及独立动作方向/排序。完整 tagged graph 证明为后续补证项，**不是启动条件** |
| M4 | K=1 暂收/撤销、状态刷新、fresh-route 验收及一致回退；报告全部场景指标、得分分解、实际耗时 |

### M0 历史补证协议（可选，不阻塞后续开发）

每项先冻结输入、样本、物理假设与容差，再采证；记录原生参照、重建值、单位、误差、
覆盖、未匹配和失败原因。0.05 ps 只适用于原 M0 时序对齐，cap 独立容差为 0.00001 fF，
不移用于 M3 导数或排序。缺参照、未解释状态、截断、重复/缺失样本均不得标通过。

1. **合法 predecessor/tag/edge/check。** 有界 TC setup 多 fan-in 子图包含胜出与非胜出输入、
   全部边界状态、sense/transition/clock/path/条件/disabled。参照必须来自原生 timer
   传播兼容性与本地 check，不能从同一结构推断规则自证。逐状态核对合法集合及应排除配对，
   用同状态原生 arc delay 重建 max AT，核对终点 RAT/slack；transition/clock 名不同不自动不兼容。
   先求 requireds，再辨认 raw RAT 是否有效/受约束，零或非零都不是单独判据。
   即使有界全覆盖通过，目标全图覆盖仍需单独证明；该验证现已暂缓，**不作为 M3 开发前置条件**。当前缺原生兼容性 oracle，保持未通过。
2. **选中 setup LUT。** 原生实际 check 的 master、pins、scene、transitions、role/条件、
   data/clock slew 唯一绑定 Liberty constraint 表及轴；独立求值对照纯 native Check，
   拆清 derated margin、capture clock、uncertainty/CRPR 后重建 RAT/slack。
   key 集必须恰等于固定 endpoint×transition，无重复代替缺样本。
   已通过固定四样本；未观测条件分支和 M3 data-slew 导数仍未验证。
3. **load/model input。** 保留 driver/arc/scene/transition、input slew、connected pin/wire/total cap、
   report Cap、DMP Pi/Ceff、所选表与 delay/slew 输出；每 driver 两 transition 的 report Cap 必须唯一。
   不能强令 Pi 或 Ceff 等于 connected cap，也不能用 cap 求和替代模型输入身份验证。
   固定两 driver 的 20 样本通过；DMP waveform slew 与 LUT transition slew 有别，
   sink slew 尚未独立建模，网络 Jacobian 与动作预测仍属 M3。

## 5. 验证、证据与未决条件

复用公共 submission/evaluation 接口为主要外部验收；探针、单步 STA 和小型数值自检为支持证据。
测试观察数据身份、单位、合法动作、loss/benefit 符号、接受/拒绝及完整回退，不锁定私有布局。
M3 需先用调试样本测噪声，再冻结独立动作、阈值、预算和物理假设，覆盖可用 drive/VT、
关键程度和相关族；记录预测/真实收益、护栏与不可执行原因。与 delay-only 排序使用相同真实
STA 验证预算，比较正收益命中率、排序相关性及实际收益；几何/新路由效果单列。
阈值尚无实测依据，不能测后移动，也不能以校准后 forward 接近替代梯度验证。

尚未验证：完整合法 tagged graph、raw required-state 约束有效性、网络对 driver slew/
sink cap/driver 模型的 Jacobian、slew 平滑、并列 seed、零/负 arrival 数值规则、
校准刷新/漂移、替换持久化与独立动作预测价值。固定 wire RC 是假设，不是固定 Ceff。
其中局部/网络模型、slew 平滑、并列 seed、数值处理、参考刷新策略及验证阈值是 M3–M4 需要落实的工作项，不是等待用户提供的前置资料。实现者在现有后端和既定范围内选择最小明确方案，记录近似与覆盖，并在独立验收前冻结；不得静默省略三项梯度贡献、setup seed 或放松验收护栏。完整 Tag 证明仍按项目决策暂缓。

正式工具沿用四参数 `<input_dir> <platform_dir> <output_dir> <top_module>`，生成按 benchmark
命名的一致 DEF/Verilog。正常 launcher 实测 tool/flow runtime，五小时上限不免 runtime 扣分；
完整认证和 fresh MCMM 后评分。每次新建 `.runs/` 目录，保留输入/配置/source/tool hash、
命令、原始导出、动作/指标/回退与各门状态。失败就记录失败，不把复制 R0 或仅 `eval.sh`
通过当作成功优化/正式分数。

**最终交付与完成判据：**
- 可由公共 launcher 调用的四参数优化工具，能对 AES 执行 sizing 并导出一致的 DEF/Verilog；真实运行证据不能仅由 M0 探针或占位程序代替。
- M1 候选覆盖与拒绝原因、M2 VT/drive 替换及必要合法化的完整验证、M3 局部导数和独立动作预测对照均有可复核结果。
- M4 实际运行 K=1 更新循环，验证接受、拒绝/撤销、状态刷新及一致回退；对输出候选执行 fresh-route、完整合法性和三场景评测，记录实测 tool/flow runtime 与评分分解。
- 汇总命令、版本/hash、测试、逐阶段结果和剩余近似。分别报告“实现完成”“验证通过”“竞赛净收益成立”；未达冻结标准或未获得有效改进时如实标失败，不把运行结束、仅 TC 改善或复制 R0 当作全部验收通过。

首版不实现独立 timer、hold/min 或 MCMM 联合梯度、GPU、批量/冻结调度、LSESmooth power
recovery、buffer 插删、pin swap、逻辑重构/cloning、主动 placement（必要 legalization 除外）。
若必须改 OpenSTA 内核、换后端或扩算法，先单独决定路线；历史 checker 的 blocked 不再作为停止 M1–M4 的理由。
