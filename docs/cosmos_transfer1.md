# Cosmos-Transfer1 在 LeFlow 中的使用说明

本文档说明如何在 LeFlow 中安装、配置并使用 NVIDIA Cosmos-Transfer1 生成视频变体。

重点：

- LeFlow 不直接加载 Cosmos 模型权重。
- LeFlow 通过 `cosmos_transfer.py` 调用外部 Cosmos proxy service。
- 真正运行 Cosmos-Transfer1 推理的是 proxy service。
- Cosmos 权重和 Cosmos 推理环境应该放在 proxy 所在机器上。

---

## 1. 整体关系

LeFlow 的完整数据流是：

```text
Mimic -> split -> render -> cosmos -> lerobot
```

其中 `cosmos` 阶段的作用是：

```text
render 生成基础视频
    |
    v
LeFlow 调用 Cosmos proxy
    |
    v
Cosmos-Transfer1 生成视觉变体
    |
    v
LeFlow 保存 Cosmos 视频
    |
    v
lerobot 导出最终数据集
```

需要区分两个环境：

| 环境 | 作用 | 是否需要 Cosmos 权重 |
| --- | --- | --- |
| LeFlow 环境 | 调度 pipeline、上传视频、保存结果 | 不需要 |
| Cosmos 环境 | 加载权重、运行 Cosmos-Transfer1 推理 | 需要 |

---

## 2. ms 服务器上的 Cosmos 路径

当前 ms 上 Cosmos-Transfer1 源码目录是：

```text
/data/120T/kqg/cosmos-transfer1-src
```

Cosmos checkpoints 根目录应该配置为：

```text
/data/120T/kqg/cosmos-transfer1-src/checkpoints
```

注意：推理命令里的 `--checkpoint_dir` 应该指向 `checkpoints` 根目录，而不是直接指向 `Cosmos-Transfer1-7B` 子目录。

当前 ms 上主要权重目录是：

```text
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B
```

主要权重文件：

```text
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/base_model.pt
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control.pt
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control_distilled.pt
```

文件大小：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `base_model.pt` | 14.47 GB | Cosmos-Transfer1 基础模型 |
| `edge_control.pt` | 3.57 GB | Edge control 原始权重 |
| `edge_control_distilled.pt` | 16.26 GB | Edge control 蒸馏权重 |

如果使用普通 Edge / Canny 控制，重点会用到：

```text
base_model.pt
edge_control.pt
```

如果使用 distilled edge 推理，则会用到：

```text
base_model.pt
edge_control_distilled.pt
```

蒸馏模型的推理速度更快，但是质量可能不太好，需要看具体场景的复杂程度。

---

## 3. 从零安装 Cosmos-Transfer1

如果 ms 上已经有可用环境，可以跳过本节，直接看第 4 节。

### 3.1 系统要求

Cosmos-Transfer1 官方要求：

```text
系统：Linux
测试系统：Ubuntu 20.04 / 22.04 / 24.04
Python：3.12.x
```

建议 Cosmos-Transfer1 使用独立 conda 环境，不要和 LeFlow 环境混在一起。

---

### 3.2 克隆源码

```bash
git clone git@github.com:nvidia-cosmos/cosmos-transfer1.git
cd cosmos-transfer1
git submodule update --init --recursive
```

如果使用 ms 上已有源码，则直接进入：

```bash
cd /data/120T/kqg/cosmos-transfer1-src
```

---

### 3.3 检查 libnvrtc

先检查机器上是否有 `libnvrtc.so`：

```bash
find /usr -name "libnvrtc.so*" 2>/dev/null | head -n 10
```

如果没有输出，需要根据 CUDA 版本安装相关依赖。

查看 CUDA 版本：

```bash
nvidia-smi | grep "CUDA Version"
```

例如 CUDA 12.8：

```bash
CUDA_VERSION=12-8

apt-get update && apt-get install -y wget gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

dpkg -i cuda-keyring_1.1-1_all.deb

apt-get -y update && apt-get install -y \
  cuda-nvrtc-$CUDA_VERSION \
  libcublas-$CUDA_VERSION \
  libcurand-$CUDA_VERSION \
  libcusparse-$CUDA_VERSION
```

如果是 Ubuntu 20.04，需要把 keyring 下载命令换成：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
```

---

### 3.4 创建 conda 环境

在 Cosmos-Transfer1 源码根目录执行：

```bash
conda env create --file cosmos-transfer1.yaml
conda activate cosmos-transfer1
```

安装基础依赖：

```bash
pip install -r requirements.txt
```

安装推理相关依赖：

```bash
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl

export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.2

pip install decord==0.6.0

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/apex-0.1+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/flash_attn-2.6.3+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/natten-0.21.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/transformer_engine-1.13.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
```

修复 conda 环境下 Transformer Engine 的头文件链接问题：

```bash
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/

ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
```

安装系统依赖：

```bash
apt-get install -y libmagic1
```

---

### 3.5 测试 Cosmos 环境

在 Cosmos-Transfer1 根目录执行：

```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py
```

如果这一步通过，说明 Cosmos-Transfer1 推理环境基本可用。

---

## 4. 下载 Cosmos 权重

如果 ms 上已经有下面这些文件，可以跳过本节：

```text
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/base_model.pt
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control.pt
/data/120T/kqg/cosmos-transfer1-src/checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control_distilled.pt
```

如果是从零下载，需要先登录 Hugging Face：

```bash
huggingface-cli login
```

还需要接受 Llama-Guard-3-8B 的使用条款。

然后在 Cosmos-Transfer1 根目录执行：

```bash
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

注意：

```text
完整 checkpoints 大约需要 300GB 磁盘空间。
不是每次推理都会用到所有 checkpoint，但建议先按官方方式完整下载。
```

下载后目录结构大致是：

```text
checkpoints/
├── nvidia/
│   ├── Cosmos-Guardrail1/
│   ├── Cosmos-Transfer1-7B/
│   │   ├── base_model.pt
│   │   ├── vis_control.pt
│   │   ├── edge_control.pt
│   │   ├── edge_control_distilled.pt
│   │   ├── seg_control.pt
│   │   ├── depth_control.pt
│   │   ├── 4kupscaler_control.pt
│   │   └── config.json
│   ├── Cosmos-Transfer1-7B-Sample-AV/
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   └── Cosmos-UpsamplePrompt1-12B-Transfer/
├── depth-anything/
├── facebook/
├── google-t5/
├── IDEA-Research/
└── meta-llama/
```

---

## 5. 单独测试 Cosmos-Transfer1 推理

在接入 LeFlow 之前，建议先单独跑通一次 Cosmos-Transfer1 推理。

进入源码目录：

```bash
cd /data/120T/kqg/cosmos-transfer1-src
conda activate cosmos-transfer1
```

设置环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export CHECKPOINT_DIR=/data/120T/kqg/cosmos-transfer1-src/checkpoints
```

运行普通 Edge control 推理：

```bash
PYTHONPATH=$(pwd) torchrun \
  --nproc_per_node=$NUM_GPU \
  --nnodes=1 \
  --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --video_save_folder outputs/leflow_test_edge \
  --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus $NUM_GPU
```

如果要使用 distilled edge，加上：

```bash
--use_distilled
```

完整示例：

```bash
PYTHONPATH=$(pwd) torchrun \
  --nproc_per_node=$NUM_GPU \
  --nnodes=1 \
  --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --video_save_folder outputs/leflow_test_edge_distilled \
  --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus $NUM_GPU \
  --use_distilled
```

---

## 6. controlnet spec 核心格式

Cosmos-Transfer1 通过 `--controlnet_specs` 读取 JSON 配置。

Edge control 的最小示例：

```json
{
  "prompt": "A realistic robot manipulation scene on a clean tabletop.",
  "input_video_path": "/path/to/input.mp4",
  "sigma_max": 70,
  "edge": {
    "control_weight": 0.5,
    "canny_threshold": "medium"
  }
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `prompt` | 生成提示词 |
| `input_video_path` | 输入视频 |
| `sigma_max` | 输入视频加噪强度，越大生成自由度越高 |
| `edge.control_weight` | Edge 控制强度 |
| `edge.canny_threshold` | Canny 阈值，可选 `very_low / low / medium / high / very_high` |

建议初始值：

```text
sigma_max = 70
control_weight = 0.5
canny_threshold = medium
```

如果使用 distilled edge，可以优先尝试：

```text
control_weight = 1.0
```

---

## 7. Cosmos proxy 是什么

LeFlow 不直接调用 `transfer.py`，而是调用一个 HTTP proxy。

LeFlow 侧调用流程：

```text
POST /canny/submit
GET  /canny/status/{job_id}
GET  /canny/result/{job_id}
```

proxy 内部做的事情：

```text
1. 接收 LeFlow 上传的视频和 prompt
2. 保存 input.mp4
3. 生成 spec.json
4. 调用 Cosmos-Transfer1 的 transfer.py
5. 保存输出视频
6. 将结果返回给 LeFlow
```

也就是说，proxy 的本质是：

```text
HTTP server + Cosmos-Transfer1 torchrun 推理命令
```

---

## 8. 启动 Cosmos proxy

假设有一个兼容 LeFlow 的 proxy 脚本：

```text
cosmos_proxy_server.py
```

启动方式：

```bash
conda activate cosmos-transfer1

python3 cosmos_proxy_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --cosmos-root /data/120T/kqg/cosmos-transfer1-src \
  --checkpoint-dir /data/120T/kqg/cosmos-transfer1-src/checkpoints \
  --work-dir /data/120T/kqg/leflow_cosmos_jobs \
  --num-gpus 1 \
  --cuda-visible-devices 0
```

如果使用 distilled edge：

```bash
python3 cosmos_proxy_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --cosmos-root /data/120T/kqg/cosmos-transfer1-src \
  --checkpoint-dir /data/120T/kqg/cosmos-transfer1-src/checkpoints \
  --work-dir /data/120T/kqg/leflow_cosmos_jobs \
  --num-gpus 1 \
  --cuda-visible-devices 0 \
  --use-distilled
```

启动后测试：

```bash
curl http://127.0.0.1:5000/health
```

期望返回：

```json
{
  "status": "ok"
}
```

---

## 9. LeFlow 中的 cosmos 配置

在 `pipeline.json` 中启用：

```json
"cosmos": {
  "enabled": true,
  "variants": 4,
  "background_variants": 2,
  "lighting_variants": 2,
  "parallelism": 1,
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
```

重点字段：

| 字段 | 说明 |
| --- | --- |
| `server-url` | Cosmos proxy 地址 |
| `endpoint` | 推荐使用 `/canny/submit` |
| `parallelism` | LeFlow 同时提交多少个 Cosmos job |
| `variants` | 每个 episode 生成几个变体 |
| `control-weight` | 传给 Cosmos 的控制强度 |
| `sigma-max` | 传给 Cosmos 的生成自由度 |
| `canny-strength` | 对应 Cosmos 的 canny threshold |
| `timeout` | 等待 Cosmos job 的最长时间 |

注意：

```text
parallelism 不是 GPU 数量。
parallelism=4 可能意味着同时启动 4 个 Cosmos 推理任务，容易爆显存。
第一次测试建议 parallelism=1。
```

---

## 10. 在 LeFlow 中运行 Cosmos

如果前面的 `render` 阶段已经完成，可以只跑：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot
```

强制重跑：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --force
```

指定并发：

```bash
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --cosmos-workers 1
```

运行成功后，Cosmos 视频会保存到：

```text
{work_dir}/cosmos/{episode_id}/variant_{variant_index}.mp4
```

manifest 会写到：

```text
{work_dir}/manifests/cosmos.jsonl
```

后续 `lerobot` 阶段会优先使用 `cosmos.jsonl` 里的视频。

---

## 11. 不使用 Cosmos

如果暂时不想跑 Cosmos，可以关闭：

```json
"cosmos": {
  "enabled": false
}
```

然后运行：

```bash
python3 pipeline_tool.py --config pipeline.json --stages render,lerobot
```

这种情况下，LeFlow 会直接使用 `render` 阶段生成的基础视频导出 LeRobot 数据集。

---

## 12. 推荐最小使用流程

```bash
# 1. 进入 Cosmos-Transfer1 环境
cd /data/120T/kqg/cosmos-transfer1-src
conda activate cosmos-transfer1

# 2. 测试 Cosmos 环境
PYTHONPATH=$(pwd) python scripts/test_environment.py

# 3. 单独测试一次 Cosmos 推理
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export CHECKPOINT_DIR=/data/120T/kqg/cosmos-transfer1-src/checkpoints

PYTHONPATH=$(pwd) torchrun \
  --nproc_per_node=$NUM_GPU \
  --nnodes=1 \
  --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --video_save_folder outputs/leflow_test_edge \
  --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus $NUM_GPU

# 4. 启动 Cosmos proxy
python3 /path/to/LeFlow/cosmos_proxy_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --cosmos-root /data/120T/kqg/cosmos-transfer1-src \
  --checkpoint-dir /data/120T/kqg/cosmos-transfer1-src/checkpoints \
  --work-dir /data/120T/kqg/leflow_cosmos_jobs \
  --num-gpus 1 \
  --cuda-visible-devices 0

# 5. 测试 proxy
curl http://127.0.0.1:5000/health

# 6. 回到 LeFlow 跑 cosmos + lerobot
cd /path/to/LeFlow
python3 pipeline_tool.py --config pipeline.json --stages cosmos,lerobot --cosmos-workers 1
```
