#!/usr/bin/env python3
"""多 GPU 运行 Isaac Lab Mimic 的辅助脚本。

这个脚本不是新的 Mimic 算法，而是一个并行调度器：
1. 把总的 generation_num_trials 切分成多个 shard。
2. 每个 shard 在一张 GPU 上单独运行 generate_data.py。
3. 所有 shard 完成后，把多个 HDF5 和相机帧目录合并成一个统一输出。

典型用法是被 pipeline_tool.py 的 mimic stage 调用。
"""
from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError as error:
    raise SystemExit("run_mimic_multi_gpu.py requires h5py. Install with: pip install h5py") from error

from stage_utils import copy_attrs, detect_episode_container, ensure_dir

# Mimic 保存图片时的文件名里通常带有 trial 编号，合并 frame shard 时要把 local trial 改成 global trial。
TRIAL_PATTERN = re.compile(r"(.*_trial_)(\d+)(_.+)")
EPISODE_SUFFIX_PATTERN = re.compile(r"(\d+)$")


@dataclass
class ShardResult:
    """单个 Mimic shard 的运行结果。"""

    shard_index: int
    output_path: Path
    frames_root: Path
    local_episode_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mimic across multiple GPUs, then merge HDF5 and frame outputs.")
    parser.add_argument("--isaaclab-sh", default="../IsaacLab/isaaclab.sh", help="Path to isaaclab.sh")
    parser.add_argument("--generate-script", default="./generate_data.py", help="Path to generate_data.py")
    parser.add_argument("--parallelism", type=int, default=1, help="How many Mimic shards to launch in parallel")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU ids, for example 0,1,2,3")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--generation_num_trials", type=int, default=10, help="Total desired successful demos across all GPUs")
    parser.add_argument("--generation_guarantee", type=str, default="")
    parser.add_argument("--episode_length_s", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="", help="Ignored by the multi-GPU helper; each shard sets its own cuda device")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--image_data_type", type=str, default="rgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--pause_subtask", action="store_true", default=False)
    parser.add_argument("--isolate-device", "--isolate_device", dest="isolate_device", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing merged outputs before running")
    args, extra = parser.parse_known_args()
    # 保留未知参数，继续透传给 generate_data.py / Isaac Lab，避免 wrapper 限制底层能力。
    args.extra = extra
    return args


def parse_gpu_ids(raw_value: str, parallelism: int) -> list[str]:
    """解析用户指定的 GPU 列表；未指定时默认使用 0..parallelism-1。"""

    if raw_value.strip():
        gpu_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
        if not gpu_ids:
            raise SystemExit("--gpu-ids was provided but no GPU ids were parsed")
        return gpu_ids
    return [str(index) for index in range(parallelism)]


def split_counts(total: int, workers: int) -> list[int]:
    """把总 trial 数尽量均匀分配给多个 worker。"""

    if total < 1:
        raise SystemExit("--generation_num_trials must be >= 1")
    base = total // workers
    remainder = total % workers
    counts = [base + (1 if index < remainder else 0) for index in range(workers)]
    return [count for count in counts if count > 0]


def shell_join(command: list[str]) -> str:
    """只用于日志打印，保证路径中有空格时也能看清楚。"""

    return " ".join(shlex.quote(part) for part in command)


def parse_episode_index(name: str, fallback: int) -> int:
    """从 episode 名字末尾提取 trial index；提取不到时用 fallback。"""

    match = EPISODE_SUFFIX_PATTERN.search(name)
    if match:
        return int(match.group(1))
    return fallback


def collect_episode_names(path: Path) -> list[str]:
    """读取某个 shard HDF5 中的 episode 列表。"""

    with h5py.File(path, "r") as handle:
        _, episode_names = detect_episode_container(handle)
    return episode_names


def build_shard_command(args: argparse.Namespace, gpu_id: str, shard_trials: int, shard_output: Path, shard_frames_root: Path, seed: int) -> list[str]:
    """构造单个 shard 实际执行的 generate_data.py 命令。"""

    command = [
        args.isaaclab_sh,
        "-p",
        args.generate_script,
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--generation_num_trials",
        str(shard_trials),
        "--input_file",
        str(Path(args.input_file).expanduser()),
        "--output_file",
        str(shard_output),
        "--image_root",
        str(shard_frames_root),
        "--image_data_type",
        args.image_data_type,
        "--device",
        f"cuda:{gpu_id}",
        "--seed",
        str(seed),
    ]

    if args.generation_guarantee:
        command.extend(["--generation_guarantee", args.generation_guarantee])
    if args.episode_length_s > 0:
        command.extend(["--episode_length_s", str(args.episode_length_s)])
    if args.headless:
        command.append("--headless")
    if args.enable_cameras:
        command.append("--enable_cameras")
    if args.pause_subtask:
        command.append("--pause_subtask")
    if args.isolate_device:
        command.append("--isolate_device")
    if args.debug:
        command.append("--debug")

    command.extend(args.extra)
    return command


def run_shard(args: argparse.Namespace, shard_index: int, gpu_id: str, shard_trials: int, shard_output: Path, shard_frames_root: Path) -> ShardResult:
    """运行单个 Mimic shard，并检查它是否生成了预期数量的 episode。"""

    shard_output.parent.mkdir(parents=True, exist_ok=True)
    shard_frames_root.mkdir(parents=True, exist_ok=True)
    command = build_shard_command(
        args=args,
        gpu_id=gpu_id,
        shard_trials=shard_trials,
        shard_output=shard_output,
        shard_frames_root=shard_frames_root,
        seed=args.seed + shard_index,
    )
    print(f"[mimic-shard {shard_index:02d}] {shell_join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(Path(__file__).resolve().parent))

    if not shard_output.exists():
        raise SystemExit(f"Shard {shard_index:02d} did not produce {shard_output}")

    episode_names = collect_episode_names(shard_output)
    if len(episode_names) != shard_trials:
        raise SystemExit(
            f"Shard {shard_index:02d} produced {len(episode_names)} episodes, expected {shard_trials}: {episode_names}"
        )

    return ShardResult(
        shard_index=shard_index,
        output_path=shard_output,
        frames_root=shard_frames_root,
        local_episode_names=episode_names,
    )


def merge_hdf5_shards(shards: list[ShardResult], merged_output: Path) -> list[dict[str, int]]:
    """把多个 shard HDF5 合并成一个 HDF5。

    每个 shard 里的 episode 名字可能都是 demo_0/demo_1 这种局部编号，合并时会统一
    改成全局 demo_0、demo_1、demo_2...，并返回 local trial 到 global trial 的映射，
    后续合并 frame 文件名时会用到。
    """

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    if merged_output.exists():
        merged_output.unlink()

    trial_maps: list[dict[str, int]] = []
    global_episode_index = 0
    total_count = 0

    with h5py.File(merged_output, "w") as target:
        target_container = None
        target_container_name = ""

        for shard in shards:
            with h5py.File(shard.output_path, "r") as source:
                container_name, episode_names = detect_episode_container(source)

                if target_container is None:
                    copy_attrs(source, target)

                    if container_name == "/":
                        target_container = target
                    else:
                        target_container_name = container_name
                        target_container = target.create_group(container_name)
                        copy_attrs(source[container_name], target_container)

                    # 复制非 episode 的全局元信息，只在第一个 shard 复制一次。
                    for name in source.keys():
                        if name == container_name:
                            continue
                        source.copy(name, target, name=name)

                source_container = source if container_name == "/" else source[container_name]
                local_to_global: dict[str, int] = {}

                for fallback_index, episode_name in enumerate(episode_names):
                    new_name = f"demo_{global_episode_index}"
                    source_container.copy(episode_name, target_container, name=new_name)
                    local_trial = parse_episode_index(episode_name, fallback_index)
                    local_to_global[str(local_trial)] = global_episode_index
                    global_episode_index += 1

                trial_maps.append(local_to_global)
                total_count += len(episode_names)

        if target_container_name and target_container is not None:
            target_container.attrs["total"] = total_count

    return trial_maps


def merge_frame_shards(shards: list[ShardResult], trial_maps: list[dict[str, int]], merged_frames_root: Path) -> None:
    """合并多个 shard 的相机帧，并把文件名里的 local trial 改成 global trial。"""

    if merged_frames_root.exists():
        shutil.rmtree(merged_frames_root)
    merged_frames_root.mkdir(parents=True, exist_ok=True)

    for shard, local_to_global in zip(shards, trial_maps, strict=True):
        for frame_path in sorted(shard.frames_root.rglob("*.png")):
            match = TRIAL_PATTERN.match(frame_path.name)
            if not match:
                continue

            prefix, local_trial, suffix = match.groups()
            if local_trial not in local_to_global:
                continue

            global_trial = local_to_global[local_trial]
            target_name = f"{prefix}{global_trial}{suffix}"
            shutil.copy2(frame_path, merged_frames_root / target_name)


def clear_path(path: Path) -> None:
    """删除旧输出，支持文件和目录。"""

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    args = parse_args()
    output_file = Path(args.output_file).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    shard_root = ensure_dir(output_file.parent / "shards")
    frame_shard_root = ensure_dir(image_root.parent / f"{image_root.name}_shards")

    if output_file.exists():
        if args.overwrite:
            output_file.unlink()
        else:
            print(output_file)
            return 0

    if args.overwrite:
        clear_path(shard_root)
        clear_path(frame_shard_root)
        clear_path(image_root)

    shard_root.mkdir(parents=True, exist_ok=True)
    frame_shard_root.mkdir(parents=True, exist_ok=True)

    requested_parallelism = max(1, args.parallelism)
    gpu_ids = parse_gpu_ids(args.gpu_ids, requested_parallelism)
    worker_count = min(requested_parallelism, len(gpu_ids), args.generation_num_trials)
    shard_counts = split_counts(args.generation_num_trials, worker_count)
    active_gpu_ids = gpu_ids[: len(shard_counts)]

    print(
        f"[mimic-multi] total_trials={args.generation_num_trials} parallelism={worker_count} gpu_ids={active_gpu_ids}",
        flush=True,
    )

    futures = {}
    shards: list[ShardResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for shard_index, (gpu_id, shard_trials) in enumerate(zip(active_gpu_ids, shard_counts, strict=True)):
            shard_dir = shard_root / f"shard_{shard_index:02d}"
            shard_output = shard_dir / "generated_dataset.hdf5"
            shard_frames_root = frame_shard_root / f"shard_{shard_index:02d}"
            futures[
                executor.submit(
                    run_shard,
                    args,
                    shard_index,
                    gpu_id,
                    shard_trials,
                    shard_output,
                    shard_frames_root,
                )
            ] = shard_index

        for future in as_completed(futures):
            shard = future.result()
            shards.append(shard)
            print(
                f"[mimic-multi] completed shard {shard.shard_index:02d} with {len(shard.local_episode_names)} episodes",
                flush=True,
            )

    # 为了保证最终 episode 编号稳定，合并前按 shard_index 排序。
    shards.sort(key=lambda item: item.shard_index)
    trial_maps = merge_hdf5_shards(shards, output_file)
    merge_frame_shards(shards, trial_maps, image_root)

    print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
