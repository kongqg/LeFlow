# Robot Pipeline Tool 仓库笔记

本文基于当前仓库内容整理，目标是说明这个 repo 的用途、使用方式、pipeline、输入输出约定、关键技术，以及配置文件参数的含义，方便直接放到 GitHub。

## 1. 仓库定位

这个仓库本质上是一个“数据处理流水线编排器”，负责把几个独立步骤串起来：

`Mimic -> 可选 split -> render -> 可选 Cosmos -> LeRobot`

它本身不实现 Isaac Lab / Mimic 训练环境，也不实现 Cosmos 视频生成服务。它做的事情是：

- 用配置文件定义每个阶段要执行的命令。
- 用统一的上下文变量把各阶段的输入输出串起来。
- 把中间结果写入 `JSONL manifest`，便于断点续跑和排查问题。
- 最终把轨迹 HDF5 和视频整理成 LeRobot 风格的数据集目录。

## 2. 当前目录结构

核心文件如下：

| 文件 | 作用 |
| --- | --- |
| `pipeline_tool.py` | 主入口，按阶段执行整条 pipeline |
| `generate_data.py` | Mimic 数据生成包装脚本，调用 Isaac Lab / Isaac Lab Mimic |
| `split_hdf5.py` | 把多 episode 的 HDF5 拆成单 episode HDF5 |
| `render_episode.py` | 从图像帧目录或 HDF5 图像数组渲染 MP4 |
| `render_mimic_frames.py` | 从 Mimic 导出的 PNG 序列渲染 MP4 |
| `render_state_trajectory.py` | 从状态轨迹渲染一个 3D 轨迹动画 MP4 |
| `replay_render_episode.py` | 用 Isaac Lab replay demo，再把重放帧编码成 MP4 |
| `cosmos_transfer.py` | 调用本地 Cosmos 代理服务，得到处理后的视频 |
| `hdf5_video_to_lerobot.py` | 把 HDF5 + MP4 转成 LeRobot 风格数据集 |
| `stage_utils.py` | HDF5 / JSONL / 路径 / dataset 自动探测等公共工具 |
| `pipeline.example.json` | 推荐样例配置，偏“state-only 渲染” |
| `pipeline.example.toml` | TOML 版本样例配置 |
| `requirements.txt` | 非 Isaac Lab 部分依赖 |

说明：

- 当前仓库没有单独的 `configs/` 目录。
- 公开保留的样例配置放在仓库根目录，主要是 `pipeline.example.json` 和 `pipeline.example.toml`。
- 本地运行时建议从样例复制出自己的 `pipeline.json` 或 `pipeline.local.json`，这些文件不建议提交。

## 3. 这个 repo 解决什么问题

典型需求是：已经有一个机器人轨迹数据来源，或者你能通过 Mimic 生成轨迹，希望把它转成“带视频、带动作、带状态、带语言指令”的训练数据格式。

这个仓库把这个过程拆成 5 个阶段：

1. `mimic`
   从标注 HDF5 出发，调用 Isaac Lab Mimic 生成新的轨迹 HDF5，并可选导出相机帧。
2. `split`
   如果 Mimic 产出的是一个多 episode HDF5，就拆成单 episode 文件。
3. `render`
   把每个 episode 渲染成基础视频。
4. `cosmos`
   把基础视频送到本地 Cosmos 服务，生成若干视频变体。
5. `lerobot`
   把 `HDF5 + 视频 + instruction` 转成 LeRobot 风格目录和 Parquet。

## 4. 基本使用方法

### 4.1 安装基础依赖

```bash
cd /home/kqg/robot-pipeline-tool
python3 -m pip install -r requirements.txt
```

如果要跑 `mimic` 阶段，还需要额外准备：

- Isaac Lab
- Isaac Lab Mimic
- `../IsaacLab/isaaclab.sh`
- `generate_data.py` 依赖的任务环境与相关扩展

如果要跑 `cosmos` 阶段，还需要本地已有 Cosmos 代理服务，例如：

- `http://127.0.0.1:5000/process_video`

### 4.2 准备配置

建议从样例复制一份再改：

```bash
cp pipeline.example.json pipeline.local.json
```

或者使用仓库里现成的某个 `pipeline*.json` 作为起点。

### 4.3 先做 dry run

```bash
python3 pipeline_tool.py --config pipeline.local.json --dry-run
```

这个命令会打印每个阶段最终要执行的 shell command，但不会真的执行。

### 4.4 实际运行

```bash
python3 pipeline_tool.py --config pipeline.local.json
```

### 4.5 只跑部分阶段

```bash
python3 pipeline_tool.py --config pipeline.local.json --stages render,lerobot
```

可选阶段名固定为：

- `mimic`
- `split`
- `render`
- `cosmos`
- `lerobot`

`--stages all` 表示按默认顺序全跑。

### 4.6 强制重跑

```bash
python3 pipeline_tool.py --config pipeline.local.json --force
```

`--force` 会忽略已存在的输出文件，重新执行对应阶段。

## 5. Pipeline 流程与输入输出

### 5.1 阶段总览

| 阶段 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `mimic` | 标注 HDF5、Isaac Lab task 配置 | 一个生成后的 HDF5，可选相机帧目录 | 实际命令由配置决定 |
| `split` | 多 episode HDF5 | 多个单 episode HDF5 | 每个文件只保留一个 episode |
| `render` | 单 episode HDF5，或额外帧目录 | 基础 MP4 | 具体渲染方式由选用脚本决定 |
| `cosmos` | 基础 MP4 | 一个或多个变体 MP4 | 依赖本地 Cosmos HTTP 服务 |
| `lerobot` | 单 episode HDF5 + 视频 + instruction | LeRobot 风格数据集目录 | 导出 Parquet、meta、视频 |

### 5.2 主流程默认顺序

`pipeline_tool.py` 的固定阶段顺序是：

`mimic -> split -> render -> cosmos -> lerobot`

即使某个阶段在配置里存在，也只有在：

- 该阶段在 `--stages` 里被选中
- 且 `enabled != false`

时才会执行。

### 5.3 `mimic` 阶段

推荐理解为“外部生成器包装层”。

通常输入：

- 一个已标注的源 HDF5，例如 `annotated_dataset.hdf5`
- Isaac Lab task 名称
- 设备、试验次数、是否导出图像等参数

通常输出：

- `generated_dataset.hdf5`
- 可选的 `mimic_frames/` PNG 帧目录

这个阶段的命令通常类似：

```bash
../IsaacLab/isaaclab.sh -p ./generate_data.py ...
```

### 5.4 `split` 阶段

作用是把一个多 episode HDF5 拆成多个单 episode HDF5。

输入：

- 一个包含多个 episode 的 HDF5

输出：

- `output-dir/*.hdf5`

补充行为：

- 如果配置指定的输入 HDF5 不含 episode，`pipeline_tool.py` 会尝试自动回退到同名的 `_failed.hdf5` 文件。

### 5.5 `render` 阶段

这是最灵活的一层。仓库里目前有 4 种可选实现：

| 脚本 | 适用场景 | 输入 | 输出 |
| --- | --- | --- | --- |
| `render_episode.py` | HDF5 内已有图像数组，或已有帧目录 | 单 episode HDF5 / 帧目录 | MP4 |
| `render_mimic_frames.py` | Mimic 已把 PNG 帧导到磁盘 | 单 episode HDF5 + 帧根目录 | MP4 |
| `render_state_trajectory.py` | 没有 RGB，只能用状态轨迹可视化 | 单 episode HDF5 | 3D 轨迹动画 MP4 |
| `replay_render_episode.py` | 想用 Isaac Lab replay 真实重放一遍 | 单 episode HDF5 + Isaac Lab replay 脚本 | MP4 |

当前 `pipeline.example.json` / `pipeline.example.toml` 走的是：

- `render_state_trajectory.py`

原因是仓库说明里明确提到当前 stack-cube Mimic 输出偏 state-only，不一定带可直接渲染的 RGB 数据。

### 5.6 `cosmos` 阶段

作用是把基础视频送进本地 Cosmos 代理服务，产出一个或多个变体视频。

输入：

- 基础 MP4

输出：

- `variant_00.mp4`
- `variant_01.mp4`
- ...

约束：

- `cosmos_transfer.py` 通过 HTTP `POST` 调一个本地 endpoint。
- 服务响应里必须返回 `processed_video_path`。

### 5.7 `lerobot` 阶段

这是最终导出阶段。

输入：

- 单 episode HDF5
- 对应视频文件
- 语言指令 `instruction`

输出：

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/episodes.json`
- `meta/tasks.jsonl`
- `meta/tasks.json`
- `meta/modality.json`
- `data/chunk-xxx/episode_xxxxxx.parquet`
- `videos/chunk-xxx/observation.images.ego_view/episode_xxxxxx.mp4`

其中：

- `Parquet` 里保存状态、动作、时间戳、任务描述等结构化数据。
- 视频文件会被复制到 LeRobot 数据集目录中。

## 6. 中间产物与 manifest

`pipeline_tool.py` 会在 `{work_dir}/manifests/` 下维护 4 个 JSONL 文件：

| 文件 | 内容 |
| --- | --- |
| `sources.jsonl` | 发现到的 source HDF5 episode 列表 |
| `rendered.jsonl` | source HDF5 与基础视频的对应关系 |
| `cosmos.jsonl` | source HDF5 与 Cosmos 变体视频的对应关系 |
| `lerobot_pairs.jsonl` | 最终用于导出训练数据的视频/HDF5/instruction 对 |

这些文件的意义是：

- 支持从中间阶段继续跑，而不是每次重头开始。
- 让下游阶段有稳定的输入契约。
- 排查错配问题时，可以直接看 manifest。

## 7. 关键技术点

这个仓库的关键技术不在“模型”，而在“数据管道搭建”：

- `Python CLI orchestration`
  通过 `argparse + subprocess` 按配置拼命令、串阶段。
- `配置驱动`
  每个阶段只认配置文件，不把路径和参数写死在代码里。
- `HDF5 处理`
  使用 `h5py` 自动发现 episode 容器、状态、动作、图像、时间戳。
- `视频生成`
  使用 `imageio + imageio-ffmpeg + Pillow` 把图像序列编码成 MP4。
- `状态轨迹可视化`
  使用 `matplotlib` 直接把末端执行器和 cube 轨迹画成 3D 动画。
- `LeRobot 数据导出`
  使用 `pyarrow/parquet` 写出结构化训练样本。
- `HTTP 服务集成`
  使用 `requests` 调本地 Cosmos proxy。
- `可恢复流水线`
  用 `JSONL manifest` 记录每一步的发现结果和输出结果。

## 8. 配置文件总规则

### 8.1 当前配置文件在哪里

仓库当前没有独立 `configs/` 目录，配置样例在根目录：

- `pipeline.example.json`
- `pipeline.example.toml`

本地使用时，建议从样例复制自己的工作配置，例如：

- `pipeline.json`
- `pipeline.local.json`

这类本地配置不建议提交到 GitHub。

### 8.2 配置格式

`pipeline_tool.py` 支持：

- JSON
- TOML

TOML 依赖 Python 3.11+ 的 `tomllib`。

### 8.3 阶段配置通用字段

大多数 stage 都支持下面这些字段：

| 字段 | 作用 |
| --- | --- |
| `enabled` | 是否启用该阶段，默认 `true` |
| `command` | 基础命令字符串 |
| `args` | 一个对象，会被自动转成 CLI 参数 |

`args` 的转义规则是：

- 普通键值：`fps = 30` 会变成 `--fps 30`
- 布尔值 `true`：只输出 flag，例如 `--overwrite`
- 布尔值 `false`：不输出这个 flag
- 列表：同一个 flag 重复输出多次
- 字符串里可以使用占位符，例如 `{source_hdf5}`

### 8.4 路径与占位符规则

配置里常见的占位符如下：

| 占位符 | 含义 |
| --- | --- |
| `{config_dir}` | 配置文件所在目录 |
| `{run_name}` | `run.name` |
| `{work_dir}` | 本次运行工作目录 |
| `{manifests_dir}` | manifest 目录 |
| `{mimic_output}` | Mimic 批量输出 HDF5 |
| `{lerobot_root}` | LeRobot 输出根目录 |
| `{episode_id}` | 当前 source episode 的逻辑 id |
| `{source_hdf5}` | 当前单 episode HDF5 路径 |
| `{instruction}` | 当前 episode 对应指令 |
| `{base_video}` | render 阶段产出的基础视频 |
| `{variant_index}` | Cosmos 变体编号 |
| `{variant_episode_id}` | 带变体编号的 episode id |
| `{variant_video}` | Cosmos 输出视频路径 |
| `{video_path}` | 当前最终用于训练的视频路径 |
| `{final_episode_id}` | lerobot 阶段最终使用的 episode id |

补充：

- 每个上下文变量还会自动生成一个 `{name_shell}` 版本，值是 shell-safe quoted 字符串。
- 相对路径默认相对于运行 `pipeline_tool.py` 时的当前工作目录。
- 如果希望路径相对配置文件，而不是相对执行目录，应该显式用 `{config_dir}`。

## 9. 各配置段使用说明

### 9.1 `run`

| 字段 | 作用 |
| --- | --- |
| `name` | 这次运行的名字 |
| `work_dir` | 工作目录模板，支持占位符 |

示例：

```json
"run": {
  "name": "stack-cube",
  "work_dir": "runs/{run_name}"
}
```

效果是把当前运行的所有中间产物和最终结果统一落到 `runs/stack-cube/` 下面。

### 9.2 `instructions`

| 字段 | 作用 |
| --- | --- |
| `default` | 默认语言指令 |
| `file` | 可选，按 episode 映射 instruction 的文件 |

`file` 支持：

- `.json`
- `.jsonl`
- JSON list

记录格式通常是：

```json
{"episode_id": "demo_001", "instruction": "stack the cube onto the target"}
```

如果没命中映射，就回退到 `default`。

### 9.3 `mimic`

这个阶段除了通用字段，还常见：

| 字段 | 作用 |
| --- | --- |
| `batch_output` | Mimic 的批量输出路径 |
| `batch_output_key` | 把 `batch_output` 写回上下文时使用的 key 名 |

`mimic.args` 在当前仓库里最常见的是：

| 参数 | 作用 |
| --- | --- |
| `task` | Isaac Lab 任务名 |
| `num_envs` | 并行环境数 |
| `generation_num_trials` | 生成尝试次数 |
| `generation_guarantee` | 是否强制 guarantee，字符串 `"true"` / `"false"` |
| `episode_length_s` | 单次 episode 时长，常用于快速调试 |
| `input_file` | 源标注 HDF5 |
| `output_file` | 生成后 HDF5 路径 |
| `image_root` | 相机帧输出根目录 |
| `image_data_type` | 导出图像类型，如 `rgb` |
| `pause_subtask` | 是否暂停 subtask |
| `seed` | 随机种子 |
| `device` | 计算设备，如 `cuda:0` |
| `isolate_device` | 是否在启动前隔离单卡 |
| `headless` | 是否无头运行 |
| `enable_cameras` | 是否开启相机 |
| `debug` | 输出调试信息 |

补充说明：

- `generate_data.py` 还会接收 `AppLauncher.add_app_launcher_args(parser)` 注入的 Isaac Lab 启动参数。
- 在多 GPU 机器上，`isolate_device = true` 很有用，可以避免 Isaac Sim 在 P2P 校验上卡很久。
- 如果当前任务默认不导出 RGB，可以设置 `image_data_type = "rgb"`。

### 9.4 `split`

除了通用字段，还依赖：

| 字段 | 作用 |
| --- | --- |
| `output_glob` | split 后单 episode HDF5 的 glob 表达式 |

`split.args` 支持的核心参数：

| 参数 | 作用 |
| --- | --- |
| `input` | 输入 HDF5 |
| `output-dir` | 输出目录 |
| `episodes` | 可选，只拆某些 episode，逗号分隔 |
| `overwrite` | 是否覆盖已有拆分结果 |

### 9.5 `sources`

只有一个核心字段：

| 字段 | 作用 |
| --- | --- |
| `glob` | 用于发现 source HDF5 的 glob |

说明：

- 如果没写 `sources.glob`，程序会尝试用 `split.output_glob`。
- 这是 render 阶段和后续阶段真正的数据入口。

### 9.6 `render`

除了通用字段，还需要：

| 字段 | 作用 |
| --- | --- |
| `output_pattern` | 每个 episode 的基础视频输出路径模板 |

`render.args` 的具体参数，取决于 `command` 指向哪个脚本。

如果用 `render_episode.py`：

| 参数 | 作用 |
| --- | --- |
| `input` | 单 episode HDF5 |
| `output` | 输出 MP4 |
| `fps` | 视频帧率 |
| `frames-dir` | 已有帧目录 |
| `frames-root` | 自动搜索帧目录的根路径 |
| `image-key` | 显式指定 HDF5 图像 dataset |
| `overwrite` | 覆盖输出 |

如果用 `render_mimic_frames.py`：

| 参数 | 作用 |
| --- | --- |
| `input` | 单 episode HDF5 |
| `output` | 输出 MP4 |
| `frames-root` | Mimic PNG 帧根目录 |
| `camera-name` | 相机前缀 |
| `data-type` | 图像类型后缀，如 `rgb` |
| `tile` | tile 编号 |
| `trial-index` | 手动指定 trial 编号 |
| `fps` | 视频帧率 |
| `overwrite` | 覆盖输出 |

如果用 `render_state_trajectory.py`：

| 参数 | 作用 |
| --- | --- |
| `input` | 单 episode HDF5 |
| `output` | 输出 MP4 |
| `fps` | 视频帧率 |
| `episode-name` | 指定 HDF5 内部 episode 名 |
| `eef-key` | 末端执行器位置 dataset 路径 |
| `cube-key` | cube 位置 dataset 路径 |
| `overwrite` | 覆盖输出 |

如果用 `replay_render_episode.py`：

| 参数 | 作用 |
| --- | --- |
| `input` | 单 episode HDF5 |
| `output` | 输出 MP4 |
| `task` | 可选 task 名 |
| `fps` | 视频帧率 |
| `device` | replay 所用设备 |
| `num-envs` | replay 并行环境数 |
| `replay-script` | Isaac Lab replay 脚本路径 |
| `headless` | 无头模式 |
| `enable-cameras` | 开启相机 |
| `keep-frames` | 是否保留 replay 帧 |
| `frames-dir` | replay 帧保存目录 |
| `overwrite` | 覆盖输出 |

### 9.7 `cosmos`

除了通用字段，还常见：

| 字段 | 作用 |
| --- | --- |
| `variants` | 每个 episode 生成几个视频变体 |
| `output_pattern` | 每个变体视频的输出路径模板 |

`cosmos.args` 支持：

| 参数 | 作用 |
| --- | --- |
| `input` | 输入 MP4 |
| `output` | 输出 MP4 |
| `server-url` | Cosmos proxy 基地址 |
| `endpoint` | 处理接口路径 |
| `seed` | 随机种子 |
| `control-weight` | 控制权重 |
| `sigma-max` | sigma 上限 |
| `canny-strength` | canny 强度 |
| `timeout` | HTTP 超时秒数 |
| `overwrite` | 覆盖输出 |

### 9.8 `lerobot`

除了通用字段，还常见：

| 字段 | 作用 |
| --- | --- |
| `root` | LeRobot 数据集输出根目录 |
| `manifest_path` | 最终 pair manifest 路径 |
| `success_marker` | 可选，若该标记文件已存在则跳过 |

`lerobot.args` 支持：

| 参数 | 作用 |
| --- | --- |
| `source-hdf5` | 单 episode HDF5 |
| `video` | 对应视频 |
| `episode-id` | 对外稳定 episode 标识 |
| `instruction` | 语言指令 |
| `output-root` | 导出根目录 |
| `episode-name` | 可选，指定 HDF5 内部 episode 名 |
| `action-key` | 动作 dataset 路径 |
| `state-key` | 状态 dataset 路径，可重复出现或传列表 |
| `timestamp-key` | 时间戳 dataset 路径 |
| `fps` | 没有时间戳时用于补时间轴 |
| `robot-type` | 写入 `meta/info.json` 的机器人类型 |
| `overwrite` | 覆盖已有 episode |

特别说明：

- `state-key` 可以重复出现，多个数组会按列拼接成一个 `observation.state` 向量。
- 如果不显式传 `action-key` / `state-key` / `timestamp-key`，脚本会尝试自动探测。

## 10. 样例配置怎么用

公开仓库里建议只保留两份样例配置：

| 配置文件 | 用途 |
| --- | --- |
| `pipeline.example.json` | 推荐基线，偏 state-only 渲染 |
| `pipeline.example.toml` | 与上面等价的 TOML 样例 |

如果你是第一次跑，建议：

1. 从 `pipeline.example.json` 复制出本地配置。
2. 先把 `mimic` 和 `cosmos` 关掉，只验证 `render -> lerobot`。
3. 确认 HDF5 key 路径正确，再逐步打开 `mimic` 或 `cosmos`。

## 11. LeRobot 输出结果长什么样

最终数据集目录大致如下：

```text
{lerobot_root}/
  meta/
    info.json
    episodes.jsonl
    episodes.json
    tasks.jsonl
    tasks.json
    modality.json
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
  videos/
    chunk-000/
      observation.images.ego_view/
        episode_000000.mp4
        episode_000001.mp4
```

其中：

- `episodes.jsonl` 记录 episode 索引、任务列表 `tasks`、长度，以及源 episode 与导出文件路径；同时会生成 `episodes.json` 数组版索引。
- `tasks.jsonl` 维护 instruction 到 `task_index` 的映射；同时会生成 `tasks.json` 数组版索引。
- `info.json` 汇总总 episode 数、总帧数、fps、robot_type 等元信息。

## 12. 这个仓库最关键的几个注意点

- 这个 repo 的核心价值是“编排”和“转换”，不是生成环境本身。
- `mimic` 阶段强依赖外部 Isaac Lab 运行时，不能只靠 `requirements.txt`。
- `render` 阶段才是适配不同数据源差异的关键，是否有 RGB、是否只有状态，会直接决定你选哪个 renderer。
- `lerobot` 导出不重建标签，它复用原始 HDF5 里的动作、状态和时间信息，再绑定视频。
- 配置中的路径相对执行目录，而不是天然相对配置文件目录；要避免踩坑时，优先使用 `{config_dir}`。

## 13. 一句话总结

这个仓库是一个面向机器人轨迹数据的可配置流水线工具：前面接 Mimic 或现成 HDF5，中间做拆分和渲染，可选走 Cosmos 做视频增强，最后导出成 LeRobot 风格训练数据集。
