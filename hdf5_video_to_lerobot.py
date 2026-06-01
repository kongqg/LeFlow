#!/usr/bin/env python3
"""把单个 HDF5 trajectory + MP4 视频导出成 LeRobot 风格 episode。

这个脚本通常由 pipeline_tool.py 的 lerobot stage 调用。它的输入是一条 episode：
- source_hdf5：轨迹数据，里面包含 action/state/timestamp 等。
- video：与这条轨迹对应的视频，可以是 render 视频，也可以是 Cosmos 生成的视频。
- instruction：语言任务描述。

输出是一个 LeRobot-like 数据集目录：
- meta/*.json / *.jsonl
- data/chunk-xxx/episode_xxxxxx.parquet
- videos/chunk-xxx/{camera_key}/episode_xxxxxx.mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError as error:
    raise SystemExit("hdf5_video_to_lerobot.py requires h5py. Install with: pip install h5py") from error

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit("hdf5_video_to_lerobot.py requires numpy. Install with: pip install numpy") from error

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as error:
    raise SystemExit("hdf5_video_to_lerobot.py requires pyarrow. Install with: pip install pyarrow") from error

from stage_utils import (
    DEFAULT_ACTION_KEYS,
    DEFAULT_STATE_KEYS,
    DEFAULT_TIMESTAMP_KEYS,
    append_jsonl,
    dump_json,
    ensure_dir,
    find_dataset_path,
    get_single_episode_group,
    read_jsonl,
    safe_episode_name,
)

DEFAULT_CAMERA_KEY = "observation.images.ego_view"

# 按 LeRobot 常见目录习惯，每 1000 条 episode 放进一个 chunk 目录。
CHUNK_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one HDF5 trajectory + MP4 into a LeRobot-style episode")
    parser.add_argument("--source-hdf5", required=True, help="Per-episode HDF5 file")
    parser.add_argument("--video", required=True, help="MP4 file paired with the trajectory")
    parser.add_argument("--episode-id", required=True, help="Stable user-facing episode identifier")
    parser.add_argument("--instruction", required=True, help="Language instruction for the episode")
    parser.add_argument("--output-root", required=True, help="Dataset root directory")
    parser.add_argument("--episode-name", default="", help="Optional episode group name inside the HDF5 file")
    parser.add_argument("--action-key", default="", help="Optional explicit dataset path for actions")
    parser.add_argument(
        "--state-key",
        action="append",
        default=[],
        help="Optional dataset path for states. Repeat the flag, or pass a comma-separated list, to concatenate multiple arrays",
    )
    parser.add_argument("--timestamp-key", default="", help="Optional explicit dataset path for timestamps")
    parser.add_argument("--fps", type=int, default=30, help="Fallback FPS if timestamps are missing")
    parser.add_argument("--robot-type", default="custom", help="Robot type stored in meta/info.json")
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help="Video modality key used under videos/ and stored in meta files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing episode with the same episode index")
    return parser.parse_args()


def to_2d(array: np.ndarray, label: str) -> np.ndarray:
    """把任意 feature 数组整理成 (T, D)。

    LeRobot parquet 里 state/action 都按每帧一个向量存储，所以这里把高维观测展平到
    第二维；一维数组会变成 (T, 1)。
    """

    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    raise SystemExit(f"Unexpected {label} shape: {array.shape}")


def make_timestamps(length: int, fps: int) -> np.ndarray:
    """当 HDF5 里没有 timestamp 时，用 fps 生成等间隔时间戳。"""

    return np.arange(length, dtype=np.float64) / float(fps)


def normalize_explicit_keys(values: str | list[str]) -> list[str]:
    """统一解析 CLI 传入的 key。

    支持两种写法：
    - --state-key a --state-key b
    - --state-key a,b
    """

    if isinstance(values, str):
        values = [values]

    keys: list[str] = []
    for raw_value in values:
        for part in str(raw_value).split(","):
            cleaned = part.strip().strip("/")
            if cleaned:
                keys.append(cleaned)
    return keys


def load_feature_array(
    group: h5py.Group,
    explicit_keys: str | list[str],
    candidates: list[str],
    label: str,
) -> np.ndarray:
    """从 episode group 里读取 action/state 等特征数组。

    逻辑：
    1. 如果用户显式传了 key，就只按这些 key 读取。
    2. 如果没传，就从默认候选路径里自动查找。
    3. 多个 key 会按 feature 维度拼接，用于把 joint/eef/object 等组成一个 state。
    """

    dataset_paths = normalize_explicit_keys(explicit_keys)
    if not dataset_paths:
        dataset_path = find_dataset_path(group, candidates)
        if dataset_path is None:
            raise SystemExit(f"Could not find a dataset for {label}. Pass --{label}-key explicitly")
        return to_2d(np.asarray(group[dataset_path][()]), label)

    parts: list[np.ndarray] = []
    expected_length: int | None = None

    for dataset_path in dataset_paths:
        try:
            dataset = group[dataset_path]
        except Exception as error:
            raise SystemExit(f"Could not find dataset '{dataset_path}' for {label}") from error

        array = to_2d(np.asarray(dataset[()]), label)
        if expected_length is None:
            expected_length = array.shape[0]
        elif array.shape[0] != expected_length:
            raise SystemExit(
                f"Mismatched lengths for {label}: '{dataset_paths[0]}' has {expected_length}, "
                f"but '{dataset_path}' has {array.shape[0]}"
            )
        parts.append(array)

    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def task_index_for_instruction(tasks_path: Path, instruction: str) -> int:
    """为 instruction 分配稳定 task_index。

    同一个 instruction 复用已有 task_index；新 instruction 追加到 tasks.jsonl。
    """

    existing = read_jsonl(tasks_path)
    for record in existing:
        if record.get("task") == instruction:
            return int(record["task_index"])

    task_index = len(existing)
    append_jsonl(tasks_path, {"task_index": task_index, "task": instruction})
    return task_index


def next_episode_index(episodes_path: Path) -> int:
    """获取下一个 episode_index。"""

    existing = read_jsonl(episodes_path)
    if not existing:
        return 0
    return max(int(record["episode_index"]) for record in existing) + 1


def load_episode_records(episodes_path: Path) -> list[dict[str, object]]:
    return read_jsonl(episodes_path)


def resolve_episode_index(episodes_path: Path, episode_id: str, overwrite: bool) -> tuple[int, list[dict[str, object]]]:
    """确定当前 episode 应该写到哪个 episode_index。

    如果 episode_id 已存在：
    - overwrite=false：直接报错，避免静默覆盖。
    - overwrite=true：复用原 episode_index，并从 meta 里移除旧记录。
    """

    existing_records = load_episode_records(episodes_path)
    for record in existing_records:
        if record.get("episode_id") != episode_id:
            continue
        if not overwrite:
            raise SystemExit(
                f"Episode id '{episode_id}' already exists in {episodes_path}. Pass --overwrite to replace it"
            )
        episode_index = int(record["episode_index"])
        remaining = [item for item in existing_records if item.get("episode_id") != episode_id]
        return episode_index, remaining

    return next_episode_index(episodes_path), existing_records


def chunk_name(episode_index: int) -> str:
    """按 episode_index 计算 chunk 目录名。"""

    return f"chunk-{episode_index // CHUNK_SIZE:03d}"


def list_array(values: np.ndarray) -> pa.Array:
    """把二维 numpy 数组转成 Arrow list column。"""

    return pa.array(values.tolist())


def global_index_start(episode_records: list[dict[str, object]], episode_index: int) -> int:
    """计算当前 episode 的全局 frame index 起点。

    index 是整个数据集范围内连续递增的 frame 编号，所以要累加前面 episode 的长度。
    """

    total = 0
    for record in sorted(episode_records, key=lambda item: int(item.get("episode_index", 0))):
        record_episode_index = int(record.get("episode_index", 0))
        if record_episode_index >= episode_index:
            break
        total += int(record.get("length", 0))
    return total


def write_parquet(
    target_path: Path,
    episode_index: int,
    global_index_offset: int,
    task_index: int,
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> int:
    """写单个 episode 的 parquet 数据文件。"""

    # 三者长度可能略有差异，这里取最短长度保证逐帧对齐。
    length = min(len(states), len(actions), len(timestamps))
    states = states[:length]
    actions = actions[:length]
    timestamps = timestamps[:length]
    next_done = [False] * length
    if next_done:
        next_done[-1] = True

    table = pa.table(
        {
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "index": pa.array(list(range(global_index_offset, global_index_offset + length)), type=pa.int64()),
            "timestamp": pa.array(timestamps.tolist(), type=pa.float64()),
            "task_index": pa.array([task_index] * length, type=pa.int64()),
            # 当前格式里这里写 task_index，而不是原始文本；文本会记录在 meta/tasks.jsonl。
            "annotation.human.action.task_description": pa.array([task_index] * length, type=pa.int64()),
            "next.reward": pa.array([0.0] * length, type=pa.float32()),
            "next.done": pa.array(next_done, type=pa.bool_()),
            "observation.state": list_array(states),
            "action": list_array(actions),
        }
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target_path)
    return length


def update_meta(output_root: Path, robot_type: str, fps: int, camera_key: str) -> None:
    """根据当前 episodes/tasks 重写 meta 信息。"""

    meta_dir = ensure_dir(output_root / "meta")
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_json_path = meta_dir / "episodes.json"
    tasks_path = meta_dir / "tasks.jsonl"
    tasks_json_path = meta_dir / "tasks.json"
    modality_path = meta_dir / "modality.json"

    episodes = read_jsonl(episodes_path)
    tasks = read_jsonl(tasks_path)
    total_frames = sum(int(record.get("length", 0)) for record in episodes)
    info = {
        "dataset_type": "gr00t-flavored-lerobot-v2",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "camera_key": camera_key,
        "tasks_path": str(tasks_path.relative_to(output_root)),
        "episodes_path": str(episodes_path.relative_to(output_root)),
    }
    dump_json(info_path, info)
    dump_json(
        modality_path,
        {
            "state": ["observation.state"],
            "action": ["action"],
            "video": [camera_key],
            "language": ["annotation.human.action.task_description"],
        },
    )
    dump_json(episodes_json_path, episodes)
    dump_json(tasks_json_path, tasks)


def main() -> int:
    args = parse_args()
    source_hdf5 = Path(args.source_hdf5).expanduser().resolve()
    source_video = Path(args.video).expanduser().resolve()
    output_root = ensure_dir(args.output_root)
    camera_key = str(args.camera_key).strip().strip("/")
    if not camera_key:
        raise SystemExit("--camera-key must not be empty")
    meta_dir = ensure_dir(output_root / "meta")
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"

    with h5py.File(source_hdf5, "r") as h5_file:
        _, source_episode_name, episode_group = get_single_episode_group(
            h5_file,
            episode_name=args.episode_name or None,
        )
        actions = load_feature_array(episode_group, args.action_key, DEFAULT_ACTION_KEYS, "action")
        states = load_feature_array(episode_group, args.state_key, DEFAULT_STATE_KEYS, "state")

        timestamp_key = args.timestamp_key.strip("/") if args.timestamp_key else ""
        timestamps_path = timestamp_key or find_dataset_path(episode_group, DEFAULT_TIMESTAMP_KEYS)
        if timestamps_path is None:
            timestamps = make_timestamps(min(len(states), len(actions)), args.fps)
        else:
            timestamps = np.asarray(episode_group[timestamps_path][()]).reshape(-1)

    safe_episode_id = safe_episode_name(args.episode_id)
    episode_index, episode_records = resolve_episode_index(
        episodes_path,
        episode_id=safe_episode_id,
        overwrite=args.overwrite,
    )
    task_index = task_index_for_instruction(tasks_path, args.instruction)
    chunk = chunk_name(episode_index)

    parquet_path = output_root / "data" / chunk / f"episode_{episode_index:06d}.parquet"
    video_path = output_root / "videos" / chunk / camera_key / f"episode_{episode_index:06d}.mp4"

    if (parquet_path.exists() or video_path.exists()) and not args.overwrite:
        raise SystemExit(
            f"Episode index {episode_index} already has output files. Pass --overwrite or remove the files first"
        )

    global_index_offset = global_index_start(episode_records, episode_index)
    length = write_parquet(
        parquet_path,
        episode_index=episode_index,
        global_index_offset=global_index_offset,
        task_index=task_index,
        states=states,
        actions=actions,
        timestamps=timestamps,
    )

    # 视频不重新编码，直接复制，避免改变帧率/质量/编码细节。
    video_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, video_path)

    episode_records.append(
        {
            "episode_index": episode_index,
            "tasks": [task_index],
            "length": length,
            "episode_id": safe_episode_id,
            "source_episode_name": source_episode_name,
            "parquet_path": str(parquet_path.relative_to(output_root)),
            "video_path": str(video_path.relative_to(output_root)),
        }
    )
    episode_records.sort(key=lambda record: int(record["episode_index"]))
    with episodes_path.open("w", encoding="utf-8") as handle:
        for record in episode_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    update_meta(output_root, robot_type=args.robot_type, fps=args.fps, camera_key=camera_key)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
