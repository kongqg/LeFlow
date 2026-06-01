# 文件说明

这篇文档解释 **LeFlow 仓库里每个主要文件是干什么的**。

如果你只是想跑起来，请看：

```text
docs/usage.md
```

如果你想理解整个数据流，请看：

```text
docs/pipeline.md
```

---

## 1. 文件总览

LeFlow 里的文件可以分成几类：

```text
入口与编排
配置模板
Mimic 数据生成
HDF5 拆分
视频渲染
Cosmos 调用
LeRobot 导出
公共工具
```

核心文件如下：

| 文件 | 作用 | 什么时候看 |
|---|---|---|
| `pipeline_tool.py` | 主入口，负责编排整个 pipeline | 想理解整体流程、stage 调度、续跑逻辑 |
| `pipeline.example.json` | JSON 配置模板 | 第一次跑 pipeline 时复制它 |
| `pipeline.example.toml` | TOML 配置模板 | 想用 TOML 写配置时看它 |
| `generate_data.py` | 调 Isaac Lab Mimic 生成 demonstration 数据 | 想理解 Mimic 数据怎么生成 |
| `run_mimic_multi_gpu.py` | 多 GPU 并行运行 Mimic，并合并输出 | 想加速 Mimic 数据生成 |
| `split_hdf5.py` | 把 multi-episode HDF5 拆成单 episode HDF5 | HDF5 里有多个 episode 时 |
| `render_mimic_frames.py` | 把 Mimic 导出的 PNG camera frames 合成 MP4 | 用 Mimic 相机帧渲染视频时 |
| `render_episode.py` | 从 frame directory 或 HDF5 image dataset 渲染视频 | 想用更通用的渲染方式时 |
| `render_state_trajectory.py` | 把 state trajectory 渲染成 3D debug 视频 | 想检查轨迹是否正常时 |
| `cosmos_transfer.py` | 调用本地 Cosmos proxy service | Cosmos 阶段报错或要改参数时 |
| `hdf5_video_to_lerobot.py` | 把 HDF5 + video + instruction 导出成 LeRobot 风格数据集 | 最终导出数据有问题时 |
| `stage_utils.py` | 公共工具函数 | 需要改 HDF5 解析、episode 检测、JSONL 读写时 |
| `requirements.txt` | Python 依赖 | 安装环境时 |

---

## 2. `pipeline_tool.py`

这是整个仓库最重要的入口文件。

它负责：

1. 读取 JSON / TOML 配置。
2. 解析命令行参数。
3. 按顺序运行各个 stage。
4. 展开配置里的占位符。
5. 把配置里的 `args` 转换成命令行参数。
6. 写入和读取 manifest。
7. 支持 dry-run、force、partial stages、Cosmos 并发。

常用命令：

```bash
python3 pipeline_tool.py --config pipeline.json
```

只跑部分 stage：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,cosmos,lerobot
```

dry-run：

```bash
python3 pipeline_tool.py --config pipeline.json --dry-run
```

强制重跑：

```bash
python3 pipeline_tool.py --config pipeline.json --force
```

### 2.1 它怎么调度 stage

默认顺序是：

```text
mimic -> split -> render -> cosmos -> lerobot
```

内部大致逻辑是：

```text
读取 config
构造 base_context
解析 stages
依次检查每个 stage 是否 enabled
如果是 mimic/split，调用 run_single_stage
如果是 render，调用 run_render_stage
如果是 cosmos，调用 run_cosmos_stage
如果是 lerobot，调用 run_lerobot_stage
```

### 2.2 什么时候需要改它

一般用户不需要改 `pipeline_tool.py`。

只有这些情况才建议改：

1. 你要新增一个新的 stage。
2. 你要改变 manifest 逻辑。
3. 你要改变占位符规则。
4. 你要改变 stage 的跳过 / 续跑规则。
5. 你要让 pipeline 支持新的调度方式。

如果只是换命令、换参数、换路径，应该优先改 `pipeline.json`，不要改这个文件。

---

## 3. `pipeline.example.json`

这是 JSON 配置模板。

第一次使用时，建议复制它：

```bash
cp pipeline.example.json pipeline.json
```

然后修改 `pipeline.json`。

它包含完整配置：

```text
run
instructions
mimic
split
sources
render
cosmos
lerobot
```

它适合用来跑完整流程：

```text
mimic -> split -> render -> cosmos -> lerobot
```

### 3.1 什么时候改它

一般不要直接改 `pipeline.example.json`。

更推荐：

```bash
cp pipeline.example.json pipeline.json
```

然后改自己的 `pipeline.json`。

---

## 4. `pipeline.example.toml`

这是 TOML 版本的配置模板。

它和 `pipeline.example.json` 表达的是同一类配置，只是格式不同。

如果你觉得 JSON 写复杂嵌套不方便，可以用 TOML。

注意：读取 TOML 需要 Python 3.11+ 的 `tomllib`。

如果机器上 Python 版本不够，建议使用 JSON 配置。

---

## 5. `generate_data.py`

这个文件负责真正调用 Isaac Lab Mimic 生成 demonstration 数据。

它做的事情包括：

1. 启动 Isaac Lab / Isaac Sim app。
2. 创建对应的 Mimic environment。
3. 读取 annotated HDF5 输入。
4. 调用 Isaac Lab Mimic 的数据生成逻辑。
5. 保存输出 HDF5。
6. 可选保存相机帧到 `image_root`。

常见参数：

```text
--task
--num_envs
--generation_num_trials
--input_file
--output_file
--image_root
--image_data_type
--seed
--device
--headless
--enable_cameras
--isolate_device
--debug
```

### 5.1 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. Mimic 环境启动失败。
2. Isaac Lab task 找不到。
3. 生成的 HDF5 为空。
4. 相机帧没有保存。
5. GPU 设备选择不对。
6. headless / display 相关问题。

### 5.2 和 `run_mimic_multi_gpu.py` 的关系

通常不直接手动运行 `generate_data.py`。

更常见的方式是让：

```text
run_mimic_multi_gpu.py
```

去多进程 / 多 GPU 调用它。

---

## 6. `run_mimic_multi_gpu.py`

这个文件是 Mimic 多 GPU 辅助脚本。

它负责：

1. 根据 `generation_num_trials` 和 `parallelism` 切分任务。
2. 给每个 shard 分配 GPU。
3. 每个 shard 调用一次 `generate_data.py`。
4. 等所有 shard 完成。
5. 合并多个 HDF5 shard。
6. 合并多个相机帧目录。

典型配置里会这样调用：

```json
{
  "mimic": {
    "command": "python3 run_mimic_multi_gpu.py",
    "args": {
      "parallelism": 8,
      "gpu-ids": "0,1,2,3,4,5,6,7",
      "generation_num_trials": 32
    }
  }
}
```

这表示：

```text
总共想生成 32 条 demo
最多并行 8 个 worker
使用 GPU 0 到 7
```

### 6.1 输出结构

它会先生成 shard 目录，例如：

```text
{work_dir}/mimic/shards/
{work_dir}/mimic_frames_shards/
```

最后合并成：

```text
{work_dir}/mimic/generated_dataset.hdf5
{work_dir}/mimic_frames/
```

### 6.2 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. 多 GPU 没有按预期运行。
2. 某个 shard 失败。
3. HDF5 合并后 episode 数不对。
4. frame 文件没有正确合并。
5. `generation_num_trials` 和实际生成数量不一致。

---

## 7. `split_hdf5.py`

这个文件负责把一个 multi-episode HDF5 拆成多个单 episode HDF5。

常用命令：

```bash
python3 split_hdf5.py --input generated_dataset.hdf5 --output-dir episodes
```

常见参数：

```text
--input
--output-dir
--episodes
--overwrite
```

### 7.1 它怎么识别 episode

它会调用 `stage_utils.detect_episode_container` 自动找 episode 容器。

常见结构包括：

```text
data/demo_0
data/demo_1
```

或者：

```text
episodes/demo_0
episodes/demo_1
```

或者 episode 直接在 HDF5 根目录下。

### 7.2 输出

输出一般是：

```text
{work_dir}/episodes/demo_0.hdf5
{work_dir}/episodes/demo_1.hdf5
{work_dir}/episodes/demo_2.hdf5
```

后续 `render` 阶段会逐个处理这些文件。

### 7.3 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. split 后没有生成 episode 文件。
2. episode 名字不符合预期。
3. HDF5 结构无法自动识别。
4. 只想拆某几个 episode。

---

## 8. `render_mimic_frames.py`

这个文件负责把 Mimic 保存出来的 PNG 相机帧合成 MP4。

它适合这种情况：

```text
Mimic 已经把每一帧图片保存到了 image_root
你需要把这些图片转成视频
```

常见参数：

```text
--input
--output
--frames-root
--camera-name
--data-type
--tile
--trial-index
--fps
--overwrite
```

典型配置：

```json
{
  "render": {
    "command": "python3 render_mimic_frames.py",
    "args": {
      "input": "{source_hdf5}",
      "output": "{base_video}",
      "frames-root": "{work_dir}/mimic_frames",
      "camera-name": "table_cam",
      "data-type": "rgb",
      "fps": 30,
      "overwrite": true
    }
  }
}
```

### 8.1 它怎么找 frame

它会根据下面这些信息拼出文件名模式：

```text
camera_name
data_type
trial_index
tile
step
```

类似：

```text
table_cam_rgb_trial_0_tile_0_step_0000.png
```

如果找不到 frame，会报错。

### 8.2 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. render 阶段找不到图片。
2. 视频帧顺序不对。
3. camera name 写错。
4. data type 写错。
5. trial index 推断错误。

---

## 9. `render_episode.py`

这是一个更通用的视频渲染脚本。

它可以从两类输入生成视频：

1. 已经存在的 frame directory。
2. HDF5 里的 image dataset。

常见参数：

```text
--input
--output
--fps
--frames-dir
--frames-root
--image-key
--overwrite
```

### 9.1 适合什么时候用

如果你的图片不是 Mimic 那种固定命名格式，或者图像已经直接存在 HDF5 里，可以考虑用它。

例如：

```bash
python3 render_episode.py \
  --input episode_000.hdf5 \
  --output episode_000.mp4 \
  --image-key observations/rgb \
  --fps 30
```

### 9.2 和 `render_mimic_frames.py` 的区别

| 文件 | 适合场景 |
|---|---|
| `render_mimic_frames.py` | Mimic 导出的标准 PNG frame 命名 |
| `render_episode.py` | 更通用，支持 frame directory 或 HDF5 image dataset |

---

## 10. `render_state_trajectory.py`

这个文件用于调试，不是最终训练数据必须步骤。

它会从 HDF5 中读取轨迹数据，然后渲染成一个 3D trajectory MP4。

默认关注：

```text
eef position
cube positions
```

常见参数：

```text
--input
--output
--fps
--episode-name
--eef-key
--cube-key
--overwrite
```

### 10.1 适合什么时候用

当你不确定 HDF5 里的轨迹是否正常时，可以用它快速检查：

1. 末端执行器轨迹是否合理。
2. 物体轨迹是否合理。
3. episode 是否明显失败。
4. 数据长度是否异常。

它更像 debug visualization，不是 pipeline 的必选项。

---

## 11. `cosmos_transfer.py`

这个文件负责调用 Cosmos proxy service。

它不直接实现 Cosmos 模型，而是向本地或远程 proxy 发请求。

常见参数：

```text
--input
--output
--server-url
--endpoint
--prompt
--seed
--control-weight
--sigma-max
--canny-strength
--timeout
--poll-interval
--overwrite
```

### 11.1 支持两类接口

第一类是普通同步接口，例如：

```text
/process_video
```

这种接口返回一个处理后的视频路径，脚本再把视频复制到目标位置。

第二类是异步接口，例如：

```text
/canny/submit
```

这种接口会：

```text
submit job -> polling status -> download result
```

如果使用 `/canny/submit`，必须提供 `prompt`。

### 11.2 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. Cosmos proxy 连接失败。
2. endpoint 配错。
3. submit 后一直没有结果。
4. job failed。
5. 输出视频没有保存。
6. prompt 没有传进去。

---

## 12. `hdf5_video_to_lerobot.py`

这个文件负责最终导出 LeRobot 风格数据集。

输入是：

```text
HDF5 trajectory
MP4 video
instruction
```

输出是：

```text
LeRobot-style dataset directory
```

常见参数：

```text
--source-hdf5
--video
--episode-id
--instruction
--output-root
--episode-name
--action-key
--state-key
--timestamp-key
--fps
--robot-type
--camera-key
--overwrite
```

### 12.1 它做了什么

它主要做这几件事：

1. 打开单 episode HDF5。
2. 读取 action。
3. 读取 state。
4. 读取 timestamp；如果没有 timestamp，就按 fps 生成。
5. 写入 parquet。
6. 把视频复制到 `videos/` 目录。
7. 更新 `meta/info.json`、`episodes.jsonl`、`tasks.jsonl` 等元信息。

输出目录大致是：

```text
lerobot/
├─ meta/
├─ data/
│  └─ chunk-000/
└─ videos/
   └─ chunk-000/
```

### 12.2 state-key 怎么理解

`state-key` 可以是一个字段，也可以是多个字段。

例如：

```json
{
  "state-key": [
    "obs/joint_pos",
    "obs/joint_vel",
    "obs/eef_pos",
    "obs/eef_quat",
    "obs/gripper_pos",
    "obs/object"
  ]
}
```

表示把这些数组按最后一维拼接成一个大的：

```text
observation.state
```

如果某个 key 找不到，会报错。

### 12.3 什么时候看它

当你遇到这些问题时，需要看这个文件：

1. LeRobot 目录结构不对。
2. parquet 写入失败。
3. action 或 state 维度不对。
4. HDF5 key 找不到。
5. 视频没有复制进去。
6. episode index 重复。
7. instruction / task index 不符合预期。

---

## 13. `stage_utils.py`

这是公共工具文件。

它里面放了很多被其他脚本共享的逻辑，例如：

```text
JSON / JSONL 读写
目录创建
episode 名字清洗
HDF5 episode container 检测
单 episode group 读取
dataset path 查找
image dataset 判断
视频数组归一化
frame directory 搜索
```

### 13.1 默认 key 候选

它里面定义了默认 action、state、timestamp、image key。

例如 action 可能会自动找：

```text
actions
action
obs/actions
data/actions
```

state 可能会自动找：

```text
states
state
obs/state
observations/state
robot_state
obs/robot_state
observations/proprio
proprio
obs/joint_pos
obs/joint_vel
obs/object
obs/eef_pos
obs/eef_quat
obs/gripper_pos
```

如果你的 HDF5 结构和这些默认候选不一致，推荐优先在配置里显式指定 key，而不是直接改 `stage_utils.py`。

例如：

```json
{
  "action-key": "actions",
  "state-key": ["obs/joint_pos", "obs/joint_vel"],
  "timestamp-key": "timestamps"
}
```

### 13.2 什么时候需要改它

只有当你希望全局支持一种新的 HDF5 结构时，才建议改它。

否则更推荐在配置文件里指定 key。

---

## 14. `requirements.txt`

这个文件记录 LeFlow 自己的 Python 依赖。

安装方式：

```bash
python3 -m pip install -r requirements.txt
```

注意：这里安装的是 LeFlow 的基础依赖，不包括完整 Isaac Lab / Cosmos 环境。

如果你要跑完整 pipeline，还需要另外准备：

```text
Isaac Lab
Isaac Lab Mimic
Cosmos proxy service
```

---

## 15. 推荐阅读顺序

如果你是第一次接触这个 repo，建议按这个顺序看：

```text
README.md
    |
    v
docs/usage.md
    |
    v
docs/pipeline.md
    |
    v
docs/files.md
    |
    v
pipeline.example.json
    |
    v
pipeline_tool.py
```

如果你只是使用者，看到 `pipeline.example.json` 基本就够了。

如果你要 debug，按报错所在 stage 去看对应文件：

| 报错阶段 | 优先看 |
|---|---|
| 配置解析 / stage 调度 | `pipeline_tool.py` |
| Mimic 生成失败 | `generate_data.py`、`run_mimic_multi_gpu.py` |
| HDF5 拆分失败 | `split_hdf5.py`、`stage_utils.py` |
| 视频渲染失败 | `render_mimic_frames.py` 或 `render_episode.py` |
| Cosmos 失败 | `cosmos_transfer.py` |
| LeRobot 导出失败 | `hdf5_video_to_lerobot.py`、`stage_utils.py` |

---

## 16. 修改代码时的原则

优先级建议是：

```text
先改配置
再改单个 stage 脚本
最后才改 pipeline_tool.py
```

原因是：

1. 大多数路径、参数、开关都可以通过配置解决。
2. 单个 stage 的问题应该尽量限制在对应脚本里。
3. `pipeline_tool.py` 是总调度器，改动它容易影响整个流程。

例如：

- 只是换 camera name：改 `pipeline.json`。
- 只是换 state key：改 `pipeline.json`。
- 只是关闭 Cosmos：改 `pipeline.json`。
- render 脚本不支持你的图片格式：改 `render_episode.py`。
- HDF5 的 episode 检测规则不适配：考虑改 `stage_utils.py`。
- 要新增一个全新的 stage：再考虑改 `pipeline_tool.py`。

---

## 17. 一句话总结

这个 repo 的文件关系可以这样理解：

```text
pipeline_tool.py 负责调度
pipeline.example.json 负责配置
stage 脚本负责干活
stage_utils.py 负责公共能力
manifests 负责记录中间状态
```

如果只记住一件事：

> LeFlow 的核心不是某一个脚本，而是把 HDF5、视频、instruction 和 manifest 串成一条稳定可续跑的数据生成链路。
