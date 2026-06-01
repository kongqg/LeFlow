# 使用方式

这篇文档说明 **LeFlow 怎么跑起来**。

LeFlow 不是 Isaac Lab、Isaac Lab Mimic、NVIDIA Cosmos 或 LeRobot 的替代品。它的定位是一个轻量级流水线编排工具，把下面这些步骤串成一个可重复执行、可续跑、可排查的数据生成流程：

```text
Mimic -> split -> render -> Cosmos -> LeRobot
```

最常用的入口命令是：

```bash
python3 pipeline_tool.py --config pipeline.json
```

---

## 1. 运行前需要准备什么

如果只使用 LeFlow 本身，先安装仓库依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果要跑完整流程，还需要提前准备好外部运行环境：

1. 可正常运行的 Isaac Lab / Isaac Lab Mimic 环境。
2. 可用于 Mimic 的 annotated HDF5 输入文件。
3. 如果启用 `cosmos`，需要可访问的 Cosmos proxy service。
4. 足够的磁盘空间，用来保存 HDF5、视频、Cosmos 变体和最终 LeRobot 数据集。

不同使用场景对环境要求不同：

| 使用场景 | 是否需要 Mimic | 是否需要 Cosmos | 说明 |
|---|---:|---:|---|
| 完整流程 | 需要 | 需要 | 从 Mimic 生成数据，一直到 LeRobot 导出 |
| 不做 Cosmos 增强 | 需要 | 不需要 | 只跑到基础视频和 LeRobot 数据 |
| 已经有 HDF5 | 不需要 | 可选 | 直接从已有 episode HDF5 开始 |
| 只重新跑后半段 | 不需要 | 视情况 | 例如只重跑 `cosmos,lerobot` |

---

## 2. 准备配置文件

从示例配置复制一份：

```bash
cp pipeline.example.json pipeline.json
```

然后编辑 `pipeline.json`。

最少需要检查这些字段：

```text
run.work_dir
instructions.default
mimic.command / mimic.args
split.command / split.args
sources.glob
render.command / render.args
cosmos.command / cosmos.args
lerobot.command / lerobot.args
```

一个典型的 `run` 配置是：

```json
{
  "run": {
    "name": "stack-cube",
    "work_dir": "runs/{run_name}"
  }
}
```

这里的含义是：

- `run.name`：本次任务的名字。
- `run.work_dir`：所有中间产物和最终数据集的输出根目录。
- `{run_name}`：占位符，会被替换成 `run.name`。

---

## 3. 先跑 dry-run

正式运行前，建议先 dry-run：

```bash
python3 pipeline_tool.py --config pipeline.json --dry-run
```

`--dry-run` 只打印将要执行的命令，不真正运行外部程序。

它主要用来检查：

1. 配置文件能否被正常解析。
2. 每个 stage 的命令是否拼接正确。
3. 路径占位符是否展开正确。
4. `source_hdf5`、`base_video`、`variant_video` 等路径是否符合预期。

如果 dry-run 里看到路径明显不对，先改配置，不要直接跑完整流程。

---

## 4. 跑完整流程

确认配置没问题后，运行：

```bash
python3 pipeline_tool.py --config pipeline.json
```

默认流程是：

```text
mimic -> split -> render -> cosmos -> lerobot
```

运行结束后，主要结果会在 `{work_dir}` 下：

```text
{work_dir}/
├─ mimic/
│  └─ generated_dataset.hdf5
├─ episodes/
│  └─ *.hdf5
├─ rendered/
│  └─ *.mp4
├─ cosmos/
│  └─ .../*.mp4
├─ lerobot/
│  ├─ meta/
│  ├─ data/
│  └─ videos/
└─ manifests/
   ├─ sources.jsonl
   ├─ rendered.jsonl
   ├─ cosmos.jsonl
   └─ lerobot_pairs.jsonl
```

如果只关心最终训练数据，重点看：

```text
{work_dir}/lerobot
```

---

## 5. 常用运行方式

### 5.1 跑完整流程

```bash
python3 pipeline_tool.py --config pipeline.json
```

适合从 Mimic 生成数据开始，一直跑到 LeRobot 导出。

---

### 5.2 只跑部分 stage

使用 `--stages` 指定需要运行的阶段：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,cosmos,lerobot
```

注意：只跑部分 stage 时，前置产物必须已经存在。

例如只跑 `cosmos,lerobot`，就要求前面已经有：

```text
{work_dir}/manifests/rendered.jsonl
{work_dir}/rendered/*.mp4
```

---

### 5.3 只跑 Cosmos 和 LeRobot

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot
```

适合基础视频已经渲染好，只想重新生成 Cosmos 视频并导出最终数据集的情况。

---

### 5.4 强制重跑

默认情况下，如果目标文件已经存在，LeFlow 会跳过对应步骤。

如果希望忽略已有输出并重新执行，使用 `--force`：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

常见用途：

1. 修改了 Cosmos prompt。
2. 修改了 Cosmos 参数。
3. 发现已有视频或 manifest 过期。
4. 想重新覆盖 LeRobot 导出结果。

---

### 5.5 指定 Cosmos 并发数

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --cosmos-workers 4
```

`--cosmos-workers` 会覆盖配置里的 `cosmos.parallelism`。

注意：这里控制的是 LeFlow 向 Cosmos proxy 提交任务的并发数，不保证 Cosmos proxy 一定会自动使用多张 GPU。是否真的多 GPU 调度，取决于你的 Cosmos proxy service 自己怎么实现。

---

### 5.6 不跑 Cosmos，直接导出 LeRobot

如果不需要 world-model 视频增强，可以在配置里关闭 Cosmos：

```json
{
  "cosmos": {
    "enabled": false
  }
}
```

然后运行：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,lerobot
```

这种情况下，LeRobot exporter 会使用 `render` 阶段生成的基础视频。

---

### 5.7 使用已有 HDF5 数据

如果你已经有每个 episode 一个 HDF5 文件，可以不跑 `mimic` 和 `split`。

配置示例：

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

然后根据需要运行：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,lerobot
```

或者：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,cosmos,lerobot
```

---

## 6. 配置里的 args 会怎么变成命令行

每个 stage 通常有：

```json
{
  "command": "python3 some_script.py",
  "args": {
    "input": "{source_hdf5}",
    "output": "{base_video}",
    "fps": 30,
    "overwrite": true
  }
}
```

LeFlow 会把它转换成类似这样的命令：

```bash
python3 some_script.py --input xxx.hdf5 --output xxx.mp4 --fps 30 --overwrite
```

规则是：

1. 普通值会变成 `--key value`。
2. `true` 会变成单独的 flag，例如 `--overwrite`。
3. `false` 或 `null` 不会输出。
4. list 会展开成多个同名参数。

例如：

```json
{
  "state-key": [
    "obs/joint_pos",
    "obs/joint_vel",
    "obs/eef_pos"
  ]
}
```

会被渲染成多个：

```bash
--state-key obs/joint_pos --state-key obs/joint_vel --state-key obs/eef_pos
```

---

## 7. 常见问题

### 7.1 找不到 source HDF5

错误通常类似：

```text
No source HDF5 files matched
```

优先检查：

1. `sources.glob` 是否写对。
2. `split.output_glob` 是否写对。
3. `split` 阶段是否真的生成了 `episodes/*.hdf5`。
4. `{work_dir}` 是否展开到了你预期的位置。

---

### 7.2 LeRobot 用的是 render 视频，不是 Cosmos 视频

通常原因是：

1. `cosmos` 没有实际执行。
2. `{work_dir}/manifests/cosmos.jsonl` 不存在。
3. `cosmos.enabled` 是 `false`。
4. Cosmos manifest 存在，但里面记录的视频路径已经失效。

解决方式：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

如果 manifest 已经过期，也可以删除旧的：

```bash
rm {work_dir}/manifests/cosmos.jsonl
```

然后重跑。

---

### 7.3 Cosmos 没跑起来

优先检查：

1. `cosmos.enabled` 是否为 `true`。
2. `cosmos.args.server-url` 是否可访问。
3. Cosmos proxy service 是否真的启动。
4. `cosmos.args.endpoint` 是否和 proxy 提供的接口一致。
5. 使用 `/canny/submit` 时，是否传了 `prompt`。

---

### 7.4 HDF5 里找不到 action 或 state

LeRobot 导出时需要从 HDF5 中读取 action、state 和 timestamp。

如果自动搜索失败，可以在 `lerobot.args` 里显式指定：

```json
{
  "action-key": "actions",
  "state-key": [
    "obs/joint_pos",
    "obs/joint_vel",
    "obs/eef_pos",
    "obs/eef_quat",
    "obs/gripper_pos",
    "obs/object"
  ],
  "timestamp-key": "timestamps"
}
```

如果没有 timestamp，exporter 会根据 `fps` 生成默认时间戳。

---

## 8. 推荐使用顺序

第一次使用时，推荐按这个顺序来：

1. 先看 `docs/pipeline.md`，理解数据怎么流。
2. 再复制 `pipeline.example.json`。
3. 修改 `pipeline.json` 里的路径、task、instruction、Cosmos 参数。
4. 先跑 `--dry-run`。
5. 确认命令没问题后，再跑完整 pipeline。
6. 如果报错，再看 `docs/files.md` 找到对应脚本，然后定位问题。
