# Robot Pipeline Tool

这个仓库负责把下面这条链路串起来：

`Mimic -> split -> render -> Cosmos -> LeRobot`

它本身不实现 Isaac Lab Mimic 或 NVIDIA Cosmos，只负责：

- 按顺序调用各 stage 的命令
- 在 stage 之间传递文件
- 生成 manifest，支持续跑和排查

## 直接使用

### 1. 先准备好环境

你需要先有这些东西：

- 可运行的 Isaac Lab / Mimic 环境
- 可运行的 Cosmos proxy service（如果要跑 `cosmos`）
- 本仓库的 Python 依赖

安装仓库依赖：

```bash
cd /path/to/robot-pipeline-tool
python3 -m pip install -r requirements.txt
```

### 2. 复制配置文件

```bash
cd /path/to/robot-pipeline-tool
cp pipeline.example.json pipeline.json
```

然后编辑 `pipeline.json`。最少需要确认这些字段：

- `run.work_dir`
- `instructions.default`
- `mimic.command` / `mimic.args`
- `render.command` / `render.args`
- `cosmos.command` / `cosmos.args`（如果启用）
- `lerobot.command` / `lerobot.args`

### 3. 先做 dry-run（冒烟测试）

```bash
python3 pipeline_tool.py --config pipeline.json --dry-run
```

### 4. 跑完整流程

```bash
python3 pipeline_tool.py --config pipeline.json
```

### 5. 常用命令

只跑 `cosmos + lerobot`：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot
```

强制重跑：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

指定 Cosmos 并发数：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force --cosmos-workers 4
```

### 6. 最终会产出什么

主要输出都在 `{work_dir}` 下：

- `mimic/generated_dataset.hdf5`
- `episodes/*.hdf5`
- `rendered/*.mp4`
- `cosmos/.../*.mp4`
- `lerobot/...`
- `manifests/*.jsonl`

如果只看最终训练数据，重点就是 `{work_dir}/lerobot`。

## 详细说明

### 流程说明

#### `mimic`

输入标注数据或源数据，调用 `generate_data.py`，产出一个 HDF5。这个 HDF5 可能是单 episode，也可能是多 episode。

#### `split`

如果上一步得到的是多 episode HDF5，就用 [split_hdf5.py](./split_hdf5.py) 拆成每个 episode 一个文件。

#### `render`

把每个 episode 的 HDF5 渲染成一个基础视频。可选脚本：

- [render_mimic_frames.py](./render_mimic_frames.py)
- [render_state_trajectory.py](./render_state_trajectory.py)
- [render_episode.py](./render_episode.py)

#### `cosmos`

把基础视频送给 Cosmos proxy，生成 world-model 视频。一个 episode 可以生成多个变体。

#### `lerobot`

把 `HDF5 + 视频 + instruction` 转成 LeRobot 风格的数据集目录，使用脚本 [hdf5_video_to_lerobot.py](./hdf5_video_to_lerobot.py)。

### Stage 顺序

默认顺序是：

```text
mimic -> split -> render -> cosmos -> lerobot
```

允许的 stage：

- `mimic`
- `split`
- `render`
- `cosmos`
- `lerobot`

你可以只跑部分 stage，但前面依赖的文件必须已经存在。

### 配置结构

配置支持 JSON 和 TOML。常见顶层 section：

- `run`
- `instructions`
- `mimic`
- `split`
- `sources`
- `render`
- `cosmos`
- `lerobot`

#### `run`

- `run.name`：本次 run 的名字
- `run.work_dir`：输出根目录

#### `instructions`

- `instructions.default`：默认 instruction
- `instructions.file`：可选，按 episode 指定 instruction 的 JSON / JSONL

#### `mimic`

- `mimic.enabled`
- `mimic.command`
- `mimic.args`
- `mimic.batch_output`

如果你要多 GPU 跑 Mimic，推荐直接用 [run_mimic_multi_gpu.py](./run_mimic_multi_gpu.py)。

常用参数：

- `task`
- `generation_num_trials`
- `parallelism`
- `gpu-ids`
- `input_file`
- `output_file`
- `image_root`
- `seed`

#### `split`

- `split.enabled`
- `split.command`
- `split.args.input`
- `split.args.output-dir`
- `split.output_glob`

#### `sources`

- `sources.glob`

如果不写，默认回退到 `split.output_glob`。

#### `render`

- `render.enabled`
- `render.command`
- `render.args`
- `render.output_pattern`

常用参数：

- `input`
- `output`
- `frames-root` / `frames-dir`
- `camera-name`
- `data-type` / `image-key`
- `fps`
- `overwrite`

#### `cosmos`

- `cosmos.enabled`
- `cosmos.variants`
- `cosmos.parallelism`
- `cosmos.command`
- `cosmos.args`
- `cosmos.output_pattern`

常用参数：

- `input`
- `output`
- `prompt`
- `server-url`
- `endpoint`
- `seed`
- `control-weight`
- `sigma-max`
- `canny-strength`
- `timeout`
- `poll-interval`
- `overwrite`

如果有多个变体，可以用：

- `cosmos.prompt_templates`
- `cosmos.background_variants`
- `cosmos.lighting_variants`

#### `lerobot`

- `lerobot.enabled`
- `lerobot.root`
- `lerobot.manifest_path`
- `lerobot.command`
- `lerobot.args`
- `lerobot.success_marker`

常用参数：

- `source-hdf5`
- `video`
- `episode-id`
- `instruction`
- `output-root`
- `fps`
- `robot-type`
- `camera-key`
- `action-key`
- `state-key`
- `timestamp-key`
- `overwrite`

### 占位符

配置里的 `command`、`args` 和路径模板可以使用这些占位符：

- `{config_dir}`
- `{work_dir}`
- `{manifests_dir}`
- `{mimic_output}`
- `{source_hdf5}`
- `{episode_id}`
- `{instruction}`
- `{base_video}`
- `{variant_index}`
- `{variant_episode_id}`
- `{variant_video}`
- `{video_path}`
- `{lerobot_root}`

最常用的是：

- `{source_hdf5}`：当前 episode 的 HDF5
- `{base_video}`：render 产出的基础视频
- `{video_path}`：最终导出时使用的视频
- `{lerobot_root}`：LeRobot 输出根目录

### LeRobot 导出格式

当前 exporter 默认会把视频写成：

```text
videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

数据写成：

```text
data/chunk-000/episode_000000.parquet
```

完整目录大致如下：

```text
{lerobot_root}/
├─meta/
│ ├─info.json
│ ├─episodes.jsonl
│ ├─episodes.json
│ ├─tasks.jsonl
│ ├─tasks.json
│ └─modality.json
├─data/
│ └─chunk-000/
│   ├─episode_000000.parquet
│   └─episode_000001.parquet
└─videos/
  └─chunk-000/
    └─observation.images.ego_view/
      ├─episode_000000.mp4
      └─episode_000001.mp4
```

当前 Parquet 里会写这些关键列：

- `observation.state`
- `action`
- `timestamp`
- `annotation.human.action.task_description`
- `task_index`
- `episode_index`
- `index`
- `next.reward`
- `next.done`

其中：

- `observation.state` 是拼接后的状态向量
- `action` 是拼接后的动作向量
- `annotation.human.action.task_description` 写的是 `task_index`
- `index` 是整个数据集范围内的全局样本索引
- `next.done` 只在每个 episode 最后一帧为 `true`

### 续跑机制

工具会在 `{work_dir}/manifests/` 下写这些文件：

- `sources.jsonl`
- `rendered.jsonl`
- `cosmos.jsonl`
- `lerobot_pairs.jsonl`

规则很简单：

- 如果目标文件已经存在，且没传 `--force`，就跳过
- 如果 `cosmos.jsonl` 存在，LeRobot 默认优先使用 Cosmos 视频
- 如果 manifest 指向的文件已经丢失，工具会直接报错

### 常见问题

#### 1. `cosmos` 没跑起来

优先检查：

- `cosmos.enabled` 是否为 `true`
- `server-url` 是否可达
- proxy service 是否真的已经启动

#### 2. Cosmos 只用一张 GPU

优先检查：

- `cosmos.parallelism`
- CLI 是否传了 `--cosmos-workers`
- proxy 自己是否支持多 GPU 调度

只提高本仓库里的提交并发，不代表 proxy 会自动把任务分配到不同 GPU。

#### 3. LeRobot 用的是 render 视频，不是 Cosmos 视频

通常原因：

- `cosmos` 没有实际执行
- 或者 `cosmos.jsonl` 不存在

#### 4. `cosmos.jsonl` 在，但视频文件丢了

这是典型的 stale manifest。处理方法：

- 删掉 stale manifest 后重跑
- 或者直接 `--force` 重跑 Cosmos

## 当前推荐做法

如果你只是想稳定使用，不想折腾太多配置，推荐：

1. 用 `run_mimic_multi_gpu.py` 跑 Mimic
2. 用 `pipeline_tool.py` 统一串后续流程
3. 用 `pipeline.example.json` 或 `pipeline.example.toml` 作为模板
4. 平时主要调这几个参数：

- `generation_num_trials`
- `parallelism`
- `gpu-ids`
- `cosmos.parallelism`
- `instructions.default`

## 说明

- 这个仓库只负责编排，不替代 Isaac Lab 或 Cosmos 的运行时安装
- `mimic` 和 `cosmos` 是两个独立运行时问题
- 如果你只需要渲染视频，不需要 world-model，可以关闭 `cosmos`，直接跑 `render -> lerobot`
