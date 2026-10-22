# Quad-gradient 优化器

基于原生 STA 工作点的局部敏感度方法，用于标准单元 sizing / VT 优化。当前预测与优化目标是 **TC setup**；物理反馈覆盖 BC/TC/WC 的 setup/hold。计算使用 PyTorch CPU float64。

## 1. 模块职责

| 文件 | 职责 | 主要入口 |
|---|---|---|
| `actions.py` | **搜索空间**：加载合法单元目录，枚举 drive/VT 四方向的最近合法邻居，记录无效候选及原因；不计算收益 | `ActionCatalog`、`enumerate_actions()` |
| `model.py` | **时序模型**：解析 Liberty、查询 NLDM 表及偏导，建立 path-union 工作点图；统一处理弧筛选和缺失 launch-clock 节点补入，不执行目标反传 | `Library`、`build_timing_model()` |
| `objective.py` | **目标函数**：计算归一化 TC setup TNS/WNS 损失，生成端点 arrival/slew 梯度种子，统一参考值与单位换算 | `TCSetupObjective.value()`、`.seeds()` |
| `gradient.py` | **敏感度传播**：构造 AveSmooth 多路径权重，传播局部扰动并求取 delay、slew、load 敏感度；不生成或选择动作 | `backward()` |
| `scoring.py` | **离散动作评价**：查询候选 master 的电气差分，与敏感度结合得到三项收益及总收益 | `score_actions()` |
| `optimizer.py` | **优化算法**：Top-K 选择、实例去重、批量更新、接受/回滚、振荡冻结、best 管理和停止；协调分析与物理反馈，提供命令行入口 | `select_batch()`、`search()`、`BatchPhysical`、`optimize()` |
| `analyze.py` | **一次分析的串联**：加载工作点、生成候选、建模、生成种子、反传、评分；常驻 worker 复用固定 Library 和 ActionCatalog | `analyze()`、`main()` |
| `physical.py` | **物理执行与反馈**：管理 OpenROAD 和分析 worker，初始化设计、执行批次、读取指标，完成必要检查、超时和进程清理；不定义优化目标 | `Runner`、`PhysicalResult` |
| `backend.tcl` | **OpenROAD 侧操作**：加载设计和场景，应用换门、布线、导出 checkpoint、原生时序统计及模型所需数据 | 由 `Runner` 调用 |
| `data_io.py` | **数据读取**：加载设计快照和路径/弧表，按模型请求读取报告，提供 Liberty 文本解析及 JSON/TSV 辅助函数；不决定图覆盖规则 | `DesignView`、`TimingData`、`load_analysis_inputs()` |
| `run.sh` | **启动入口**：将参数转交给 `optimizer.py`，使用当前 PATH 中的 `python3` | `bash tools/quad_gradient/run.sh ...` |
| `pyproject.toml`、`uv.lock` | Python 依赖声明与锁定版本 | 环境配置文件 |

模型中的 delay、arrival、slew 使用 ps；物理目标统计和 reference 使用 ns。wire delay、clock slew 等在本轮工作点固定，随物理反馈刷新。

## 2. 算法流程

```text
输入网表、布局、平台与约束
    ↓
初始布线 → 独立 STA 重载与导图
    ↓
建立固定目标参考值，初始化 current / best
    ↓
┌─ 读取当前设计与工作点（复用已有分析结果）
│      ↓
│  actions：枚举合法 drive / VT 动作
│      ↓
│  model：构建时序工作点图与电气系数
│      ↓
│  objective：生成目标梯度种子
│      ↓
│  gradient：反向传播得到敏感度
│      ↓
│  scoring：计算离散动作预计收益
│      ↓
│  optimizer：Top-K → 按实例去重 → 处理振荡恢复
│      ↓
│  physical / backend：应用一批动作并布线
│      ↓
│  必要检查 → 独立干净 STA → 物理反馈与新图
│      ↓
│  分析新图，成功后提交 current；严格改善才更新 best
│      ↓
└─ 更新 K、冻结状态和停止计数；继续下一轮或结束
    ↓
输出最佳修改设计
```

### 动作评分

在当前工作点比较候选 master 与当前 master：

```text
预计目标变化 = 自身 delay 贡献 + 下游 slew 贡献 + 上游 input-cap 贡献
benefit      = -预计目标变化
```

`benefit > 0` 表示预测改善，但批次实际效果仍由物理反馈判断，不能把单动作预测收益之和当成实际改善。

### 搜索规则

- 对合法、未冻结、有限且正收益的动作做全局 Top-K，然后按实例去重，不补足空位。
- K 初始为 100；物理失败或连续三轮没有新 best 时减半，最低为 10。同轮最多缩减一次。
- 满足现有接受检查的退化设计可以成为 current；best 只在目标严格下降时替换。
- 实例发生往返换型时，尝试恢复其 best master；只有批次成功提交后才冻结。
- 失败回滚整个 checkpoint，不提交冻结或历史更新。
- 连续五个从 K=10 开始的批次未改善，或无候选、达到批数/时间预算时停止。
- 当前 hold/ERC 退化只记录诊断，不参与 TC 目标；routing overflow 等现有接受检查仍保留。
- 最终输出 best，而不是最后一次探索状态；没有修改后的改善设计时不导出候选。

## 3. 运行方法

### 3.1 进入指定环境

在 `gpu_402` 主机的 `ispd27-contest` 容器内运行，容器镜像为 `udxs/ispd27:v1`：

```bash
ssh gpu_402
docker exec -it ispd27-contest bash
cd /workspace/ISPD27-Contest

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH="/opt/oss-cad-suite/bin:/install/OpenROAD/bin:$PATH"
export OPENROAD_EXE=/install/OpenROAD/bin/openroad
export KEPLER_FORMAL_EXE=/opt/kepler-formal/build/src/bin/kepler-formal
```

Conda base 当前没有 Torch。使用已确认可用的 Python 环境：

```bash
source /workspace/ISPD27-Contest/.runs/qg-torch-uv-20261021/bin/activate
export OMP_NUM_THREADS=1
```


### 3.2 通过赛题启动器运行

在仓库根目录执行，输出目录必须是 `.runs/` 下的新目录：

```bash
python3 evaluator/run_submission.py aes .runs/quad-gradient-aes-NEW -- \
  bash /workspace/ISPD27-Contest/tools/quad_gradient/run.sh \
  --max-batches 3 --search-seconds 1800
```

将 `NEW` 替换为未使用的实验标识；不要预先创建该输出目录。运行 JPEG 时，将 `aes` 改为 `jpeg`，并使用新的输出目录。

启动器自动追加四个位置参数：

```text
<input_dir> <platform_dir> <output_dir> <top_module>
```

也可直接调用工具；此方式只运行优化器，不代替公开评测：

```bash
bash tools/quad_gradient/run.sh \
  "$INPUT_DIR" "$PLATFORM_DIR" "$OUTPUT_DIR" "$TOP_MODULE" \
  --max-batches 3 --search-seconds 1800
```

其中输入只支持仓库内打包的 AES/JPEG，平台使用仓库内 ASAP7，top 必须与对应 `benchmark.json` 一致。输出仍应选择 `.runs/` 下的新路径。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--max-batches`（别名 `--steps`） | 30 | 最多尝试批数，范围 1–30 |
| `--search-seconds` | 10800 | 优化时间预算，范围 `(0, 10800]` 秒，包含初始化与初始分析 |

优化预算不是完整公开评测的总运行预算，需为后续评测留出时间。

### 3.3 输出与测试

工具输出目录下：

- `<benchmark>.def`、`<benchmark>.v`：有改善时导出的最佳候选。
- `quad_gradient/manifest.json`：运行状态、最佳结果及输出信息。
- `quad_gradient/updates.json`：每轮选择、接受/回滚与停止记录。
- `quad_gradient/commands.json`：工具命令、日志位置与耗时。
- 各分析目录的 `actions.json`、`scores.json`、`analysis.json`：候选、评分和模型说明。
