# Pipeline 说明

这篇文档解释 **LeFlow 的数据流水线是怎么流转的**。

如果你只是想知道怎么运行，请先看：

```text
docs/usage.md
```

如果你想理解每个脚本文件是干什么的，请看：

```text
docs/files.md
```

---

## 1. LeFlow 的核心目标

LeFlow 的目标不是重新实现某个算法，而是把机器人数据生成过程中的多个独立工具串起来。

默认 pipeline 是：

```text
mimic -> split -> render -> cosmos -> lerobot
```

也可以理解为：

```text
生成轨迹数据 -> 拆分 episode -> 渲染基础视频 -> 生成视频变体 -> 导出训练数据集
```

完整数据流如下：

```text
annotated_dataset.hdf5
        |
        v
[mimic]
        |
        v
{work_dir}/mimic/generated_dataset.hdf5
        |
        v
[split]
        |
        v
{work_dir}/episodes/*.hdf5
        |
        v
[render]
        |
        v
{work_dir}/rendered/*.mp4
        |
        v
[cosmos]
        |
        v
{work_dir}/cosmos/{episode_id}/variant_*.mp4
        |
        v
[lerobot]
        |
        v
{work_dir}/lerobot/
```

---

## 2. Stage 总览

| Stage | 输入 | 输出 | 作用 |
|---|---|---|---|
| `mimic` | annotated HDF5、task config | `mimic/generated_dataset.hdf5` | 调用 Isaac Lab Mimic 生成 demonstration 数据 |
| `split` | multi-episode HDF5 | `episodes/*.hdf5` | 把多个 episode 拆成单个 HDF5 文件 |
| `render` | 单 episode HDF5、相机帧或图像数据 | `rendered/*.mp4` | 渲染基础视频 |
| `cosmos` | render 生成的基础视频 | `cosmos/.../*.mp4` | 调用 Cosmos proxy 生成视频变体 |
| `lerobot` | HDF5、视频、instruction | `lerobot/` | 导出 LeRobot 风格训练数据集 |

这几个 stage 是默认顺序，但不一定每次都要全部运行。

例如：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,lerobot
```

表示只运行：

```text
render -> lerobot
```

这种情况下，前面的 HDF5 输入必须已经存在。

---

## 3. mimic stage

`mimic` 阶段负责调用 Isaac Lab Mimic 生成机器人 demonstration 数据。

典型配置：

```json
{
  "mimic": {
    "enabled": true,
    "command": "python3 run_mimic_multi_gpu.py",
    "args": {
      "isaaclab-sh": "../IsaacLab/isaaclab.sh",
      "generate-script": "./generate_data.py",
      "parallelism": 8,
      "gpu-ids": "0,1,2,3,4,5,6,7",
      "task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
      "generation_num_trials": 32,
      "input_file": "annotated_dataset.hdf5",
      "output_file": "{mimic_output}",
      "image_root": "{work_dir}/mimic_frames",
      "overwrite": true
    },
    "batch_output": "{work_dir}/mimic/generated_dataset.hdf5",
    "batch_output_key": "mimic_output"
  }
}
```

这个 stage 的核心产物是：

```text
{work_dir}/mimic/generated_dataset.hdf5
```

如果配置里使用 `run_mimic_multi_gpu.py`，它会把总的 `generation_num_trials` 按 GPU / worker 切分成多个 shard，并行调用 `generate_data.py`，最后再把多个 shard 的 HDF5 和 frame 输出合并回来。

需要注意：

1. `mimic` 依赖 Isaac Lab / Isaac Lab Mimic 环境。
2. LeFlow 不负责安装 Isaac Lab。
3. `input_file` 通常是人工标注或已有 demonstration 的 HDF5。
4. `image_root` 用来保存 Mimic 导出的相机帧，后续 `render_mimic_frames.py` 会用到。

---

## 4. split stage

`split` 阶段负责把一个 multi-episode HDF5 拆成多个 per-episode HDF5。

典型配置：

```json
{
  "split": {
    "enabled": true,
    "command": "python3 split_hdf5.py",
    "args": {
      "input": "{mimic_output}",
      "output-dir": "{work_dir}/episodes"
    },
    "output_glob": "{work_dir}/episodes/*.hdf5"
  }
}
```

输入：

```text
{work_dir}/mimic/generated_dataset.hdf5
```

输出：

```text
{work_dir}/episodes/*.hdf5
```

为什么需要 split？

因为后续的 `render`、`cosmos`、`lerobot` 更适合以单个 episode 为单位处理数据。

`split_hdf5.py` 会尝试自动识别 HDF5 里的 episode container，例如：

```text
data
episodes
demonstrations
/
```

如果你的 HDF5 本来就是每个 episode 一个文件，可以关闭 `mimic` 和 `split`，然后直接用 `sources.glob` 指向已有文件。

---

## 5. sources

`sources` 不是一个真正执行命令的 stage，它告诉 LeFlow 后续应该处理哪些 episode HDF5。

典型配置：

```json
{
  "sources": {
    "glob": "{work_dir}/episodes/*.hdf5"
  }
}
```

如果不写 `sources.glob`，LeFlow 会尝试回退到：

```text
split.output_glob
```

每个匹配到的 HDF5 文件会形成一个 source item，包含：

```text
episode_id
source_hdf5
instruction
base_video
```

其中：

- `episode_id` 来自 HDF5 文件名。
- `source_hdf5` 是当前 episode 的 HDF5 路径。
- `instruction` 来自 `instructions.default` 或 `instructions.file`。
- `base_video` 会在 render 之后填入。

---

## 6. render stage

`render` 阶段负责把每个 episode 转成基础视频。

典型配置：

```json
{
  "render": {
    "enabled": true,
    "command": "python3 render_mimic_frames.py",
    "args": {
      "input": "{source_hdf5}",
      "output": "{base_video}",
      "frames-root": "{work_dir}/mimic_frames",
      "camera-name": "table_cam",
      "data-type": "rgb",
      "fps": 30,
      "overwrite": true
    },
    "output_pattern": "{work_dir}/rendered/{episode_id}.mp4"
  }
}
```

输入：

```text
{source_hdf5}
{work_dir}/mimic_frames
```

输出：

```text
{work_dir}/rendered/{episode_id}.mp4
```

这里的输出路径会写入：

```text
{work_dir}/manifests/rendered.jsonl
```

LeFlow 提供了几种渲染脚本：

| 脚本 | 用途 |
|---|---|
| `render_mimic_frames.py` | 从 Mimic 导出的 PNG camera frames 合成 MP4 |
| `render_episode.py` | 从 frame directory 或 HDF5 image dataset 合成 MP4 |
| `render_state_trajectory.py` | 把状态轨迹渲染成调试视频 |

如果你用的是 Mimic 导出的相机帧，通常用：

```text
render_mimic_frames.py
```

如果你的 HDF5 里已经包含图像数据，或者你已经有 frame directory，可以考虑：

```text
render_episode.py
```

---

## 7. cosmos stage

`cosmos` 阶段负责把基础视频送给 Cosmos proxy service，生成 world-model 视频或视觉变体。

典型配置：

```json
{
  "cosmos": {
    "enabled": true,
    "variants": 4,
    "background_variants": 2,
    "lighting_variants": 2,
    "parallelism": 8,
    "command": "python3 cosmos_transfer.py",
    "args": {
      "input": "{base_video}",
      "output": "{variant_video}",
      "prompt": "{instruction}. Keep the robot motion, object motion, and contact sequence unchanged. Only vary the visual appearance.",
      "seed": "{variant_index}",
      "server-url": "http://127.0.0.1:5000",
      "endpoint": "/canny/submit",
      "control-weight": 0.5,
      "sigma-max": 70,
      "canny-strength": "medium",
      "timeout": 7200
    },
    "output_pattern": "{work_dir}/cosmos/{episode_id}/variant_{variant_index}.mp4"
  }
}
```

输入：

```text
{base_video}
```

输出：

```text
{work_dir}/cosmos/{episode_id}/variant_{variant_index}.mp4
```

如果 `variants = 4`，那么每个 episode 会生成 4 个视频变体：

```text
variant_0.mp4
variant_1.mp4
variant_2.mp4
variant_3.mp4
```

LeFlow 会把 Cosmos 输出记录到：

```text
{work_dir}/manifests/cosmos.jsonl
```

后续 `lerobot` 阶段会优先使用 `cosmos.jsonl` 里的视频。如果没有 Cosmos 输出，则回退使用 `render` 生成的基础视频。

需要注意：

1. `cosmos.parallelism` 控制 LeFlow 提交 Cosmos job 的并发数。
2. 它不一定等价于 Cosmos proxy 的 GPU 数量。
3. 是否真正多 GPU，需要看你的 proxy service 怎么实现。
4. 使用 `/canny/submit` 时必须提供 prompt。

---

## 8. lerobot stage

`lerobot` 阶段负责把 HDF5、视频和 instruction 导出成 LeRobot 风格的数据集目录。

典型配置：

```json
{
  "lerobot": {
    "enabled": true,
    "root": "{work_dir}/lerobot",
    "manifest_path": "{work_dir}/manifests/lerobot_pairs.jsonl",
    "command": "python3 hdf5_video_to_lerobot.py",
    "args": {
      "source-hdf5": "{source_hdf5}",
      "video": "{video_path}",
      "episode-id": "{final_episode_id}",
      "instruction": "{instruction}",
      "output-root": "{lerobot_root}",
      "fps": 30,
      "robot-type": "custom",
      "camera-key": "observation.images.ego_view",
      "action-key": "actions",
      "state-key": [
        "obs/joint_pos",
        "obs/joint_vel",
        "obs/eef_pos",
        "obs/eef_quat",
        "obs/gripper_pos",
        "obs/object"
      ],
      "overwrite": true
    }
  }
}
```

输入：

```text
source_hdf5
video_path
instruction
```

输出：

```text
{work_dir}/lerobot/
```

输出目录大致是：

```text
lerobot/
├─ meta/
│  ├─ info.json
│  ├─ episodes.jsonl
│  ├─ episodes.json
│  ├─ tasks.jsonl
│  ├─ tasks.json
│  └─ modality.json
├─ data/
│  └─ chunk-000/
│     ├─ episode_000000.parquet
│     └─ episode_000001.parquet
└─ videos/
   └─ chunk-000/
      └─ observation.images.ego_view/
         ├─ episode_000000.mp4
         └─ episode_000001.mp4
```

Parquet 里会包含这些关键列：

```text
episode_index
index
timestamp
task_index
annotation.human.action.task_description
next.reward
next.done
observation.state
action
```

其中：

- `observation.state` 是从 HDF5 中读取并拼接的状态向量。
- `action` 是动作向量。
- `timestamp` 来自 HDF5；如果找不到，会根据 `fps` 生成。
- `next.done` 只在每个 episode 的最后一帧为 `true`。
- `annotation.human.action.task_description` 当前写入的是 `task_index`。

---

## 9. 占位符系统

配置里的命令、参数和路径模板可以使用占位符。

常见占位符：

| 占位符 | 含义 |
|---|---|
| `{config_dir}` | 配置文件所在目录 |
| `{run_name}` | 当前 run 名称 |
| `{work_dir}` | 当前 run 的输出根目录 |
| `{manifests_dir}` | manifest 输出目录 |
| `{mimic_output}` | mimic 输出 HDF5 路径 |
| `{source_hdf5}` | 当前 episode 的 HDF5 |
| `{episode_id}` | 当前 episode ID |
| `{instruction}` | 当前 episode 的语言指令 |
| `{base_video}` | render 阶段生成的基础视频 |
| `{variant_index}` | Cosmos 变体编号 |
| `{variant_episode_id}` | Cosmos 变体 episode ID |
| `{variant_video}` | Cosmos 变体视频路径 |
| `{video_path}` | 最终交给 LeRobot 的视频路径 |
| `{lerobot_root}` | LeRobot 输出根目录 |
| `{final_episode_id}` | 最终导出使用的 episode ID |

最常用的是：

```text
{source_hdf5}
{base_video}
{variant_video}
{video_path}
{lerobot_root}
```

理解这些占位符，基本就能看懂整个配置文件。

---

## 10. manifest 和续跑逻辑

LeFlow 会在：

```text
{work_dir}/manifests/
```

下面写入中间记录。

主要文件：

| Manifest | 记录什么 |
|---|---|
| `sources.jsonl` | 当前有哪些 episode HDF5 要处理 |
| `rendered.jsonl` | 每个 episode 对应的基础视频 |
| `cosmos.jsonl` | 每个 Cosmos 变体视频 |
| `lerobot_pairs.jsonl` | 最终用于导出 LeRobot 的 HDF5、视频、instruction 配对 |

续跑规则：

1. 如果目标文件已经存在，并且没有传 `--force`，LeFlow 会跳过对应步骤。
2. 如果 `cosmos.jsonl` 存在，`lerobot` 阶段会优先使用 Cosmos 视频。
3. 如果 `cosmos.jsonl` 存在，但里面记录的视频文件已经丢失，LeFlow 会报错。
4. 如果想忽略旧结果，需要使用 `--force` 或删除过期 manifest。

常见重跑命令：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

如果 manifest 已经过期，也可以手动删除：

```bash
rm {work_dir}/manifests/cosmos.jsonl
```

然后重跑：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot
```

---

## 11. 几种典型数据流

### 11.1 完整数据生成

```text
mimic -> split -> render -> cosmos -> lerobot
```

适合从 annotated dataset 开始，完整生成 LeRobot 数据集。

---

### 11.2 不使用 Cosmos

```text
mimic -> split -> render -> lerobot
```

适合只需要基础视频，不需要 world-model 增强的情况。

配置里关闭：

```json
{
  "cosmos": {
    "enabled": false
  }
}
```

---

### 11.3 已有单 episode HDF5

```text
sources -> render -> lerobot
```

配置：

```json
{
  "mimic": {
    "enabled": false
  },
  "split": {
    "enabled": false
  },
  "sources": {
    "glob": "/path/to/episodes/*.hdf5"
  }
}
```

---

### 11.4 只重新生成 Cosmos 变体

```text
cosmos -> lerobot
```

命令：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

适合你改了 prompt、seed、control weight、lighting/background variants 等参数后重新生成视频。

---

## 12. 理解 pipeline 的关键点

最重要的逻辑可以总结成三句话：

1. `mimic` 和 `split` 负责准备每个 episode 的 HDF5。
2. `render` 和 `cosmos` 负责准备每个 episode 对应的视频。
3. `lerobot` 把 HDF5、视频和 instruction 配对，导出成最终训练数据集。

如果某一步出错，先确认：

1. 上游产物是否真实存在。
2. manifest 是否记录了正确路径。
3. 配置里的占位符是否展开正确。
4. 当前 stage 是否真的 enabled。
5. 是否需要 `--force` 重跑。
