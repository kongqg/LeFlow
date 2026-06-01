#!/usr/bin/env python3
"""各个 stage 共用的工具函数。

这个文件主要解决两类重复问题：
1. HDF5 结构差异：不同数据源可能把 episode、state、action、image 放在不同路径下。
2. 文件/manifest 操作：JSON、JSONL、目录创建、episode 名字清洗等。

如果只是某个数据集的 key 不一样，优先在配置里显式指定 key；
只有想让整个项目默认支持一种新 HDF5 结构时，才建议改这里。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

# 默认 action 候选路径。exporter 会按顺序查找，第一个命中的 dataset 会被使用。
DEFAULT_ACTION_KEYS = [
    "actions",
    "action",
    "obs/actions",
    "data/actions",
]

# 默认 state 候选路径。这里故意覆盖多种常见命名，方便兼容不同 HDF5 数据格式。
DEFAULT_STATE_KEYS = [
    "states",
    "state",
    "obs/state",
    "observations/state",
    "robot_state",
    "obs/robot_state",
    "observations/proprio",
    "proprio",
    "obs/joint_pos",
    "joint_pos",
    "obs/joint_vel",
    "joint_vel",
    "obs/object",
    "object",
    "obs/eef_pos",
    "eef_pos",
    "obs/eef_quat",
    "eef_quat",
    "obs/gripper_pos",
    "gripper_pos",
    "states/articulation/robot/joint_position",
    "states/articulation/robot/joint_velocity",
]

# 如果 HDF5 没有 timestamp，hdf5_video_to_lerobot.py 会根据 fps 生成默认时间戳。
DEFAULT_TIMESTAMP_KEYS = [
    "timestamps",
    "timestamp",
    "time",
    "obs/timestamps",
]

# render_episode.py 自动从 HDF5 里找图像数据时使用的候选路径。
DEFAULT_IMAGE_KEYS = [
    "images",
    "frames",
    "rgb",
    "obs/rgb",
    "observations/rgb",
    "observations/images",
    "camera/rgb",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def safe_episode_name(name: str) -> str:
    """把任意 episode 名字转换成安全文件名。

    只保留字母、数字、点、下划线和横线，避免路径分隔符或奇怪字符影响输出。
    """

    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return clean.strip("._") or "episode"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    """以稳定格式写 JSON，便于人读和 git diff。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；文件不存在时返回空列表，方便续跑逻辑使用。"""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """向 JSONL 追加一条记录，常用于 meta/tasks.jsonl。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回 resolve 后的 Path。"""

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_episode_container(h5_file: Any) -> tuple[str, list[str]]:
    """自动识别 HDF5 中 episode 所在的容器。

    不同工具导出的 HDF5 结构不完全一样，常见形式包括：
    - data/demo_0, data/demo_1
    - episodes/demo_0, episodes/demo_1
    - demonstrations/demo_0, demonstrations/demo_1
    - demo_0, demo_1 直接在根目录下

    返回值是：
    - container name：例如 "data"，如果 episode 在根目录下则是 "/"
    - episode name 列表
    """

    candidate_names = ["data", "episodes", "demonstrations"]

    for candidate in candidate_names:
        if candidate not in h5_file:
            continue
        group = h5_file[candidate]
        episode_names = [name for name, obj in group.items() if hasattr(obj, "keys")]
        if episode_names:
            return candidate, sorted(episode_names)

    root_episode_names = [name for name, obj in h5_file.items() if hasattr(obj, "keys")]
    if root_episode_names:
        return "/", sorted(root_episode_names)

    raise ValueError("Could not detect an episode container in the HDF5 file")


def get_single_episode_group(h5_file: Any, episode_name: str | None = None) -> tuple[str, str, Any]:
    """从 HDF5 中取出一个 episode group。

    如果文件里只有一个 episode，可以自动选择；如果有多个 episode，必须显式传
    episode_name 或者先用 split_hdf5.py 拆分。
    """

    container_name, episode_names = detect_episode_container(h5_file)

    if episode_name:
        if episode_name not in episode_names:
            raise ValueError(f"Episode '{episode_name}' was not found. Available: {episode_names}")
        selected = episode_name
    elif len(episode_names) == 1:
        selected = episode_names[0]
    else:
        raise ValueError(
            "The HDF5 file contains multiple episodes. Provide --episode-name or run split_hdf5.py first"
        )

    if container_name == "/":
        return container_name, selected, h5_file[selected]
    return container_name, selected, h5_file[container_name][selected]


def copy_attrs(source: Any, target: Any) -> None:
    """复制 HDF5 group/file attrs，避免 split/merge 时丢掉元信息。"""

    for key, value in source.attrs.items():
        target.attrs[key] = value


def find_dataset_path(
    group: Any,
    preferred_paths: list[str] | None = None,
    predicate: Callable[[Any, str], bool] | None = None,
) -> str | None:
    """在 HDF5 group 里寻找一个 dataset 路径。

    查找分两步：
    1. 先按 preferred_paths 精确查找，保证用户/默认配置的优先级。
    2. 如果没找到，再遍历 group，通过 basename 或 predicate 做宽松匹配。

    返回的是相对当前 group 的 dataset path，找不到则返回 None。
    """

    preferred_paths = preferred_paths or []
    for candidate in preferred_paths:
        cleaned = candidate.strip("/")
        if not cleaned:
            continue
        try:
            obj = group[cleaned]
        except Exception:
            continue
        if hasattr(obj, "shape"):
            return cleaned

    basenames = {Path(candidate).name for candidate in preferred_paths}
    matches: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if not hasattr(obj, "shape"):
            return
        leaf = name.rsplit("/", 1)[-1]
        if name in preferred_paths or leaf in basenames:
            matches.append(name)
            return
        if predicate is not None and predicate(obj, name):
            matches.append(name)

    group.visititems(visitor)
    if matches:
        # 更浅、更短的路径通常更像主数据，避免误选深层 debug 字段。
        matches.sort(key=lambda item: (item.count("/"), len(item)))
        return matches[0]
    return None


def first_matching_dataset(group: Any, predicate: Callable[[Any, str], bool]) -> str | None:
    return find_dataset_path(group, preferred_paths=[], predicate=predicate)


def looks_like_image_dataset(dataset: Any, _name: str) -> bool:
    """粗略判断一个 HDF5 dataset 是否像视频/图像序列。

    支持 channels-last: (T, H, W, C) 和 channels-first: (T, C, H, W)。
    """

    shape = getattr(dataset, "shape", None)
    if shape is None or len(shape) != 4:
        return False
    if shape[-1] in (1, 3, 4):
        return True
    if shape[1] in (1, 3, 4):
        return True
    return False


def looks_like_vector_dataset(dataset: Any, _name: str) -> bool:
    shape = getattr(dataset, "shape", None)
    return shape is not None and len(shape) >= 1


def normalize_video_array(frames: Any, np: Any) -> Any:
    """把 HDF5 里的图像数组统一转成 imageio 可写的 uint8 RGB。

    处理内容包括：
    - channels-first 转 channels-last
    - 单通道复制成 RGB
    - RGBA 丢掉 alpha
    - float [0,1] 或其他数值类型转 uint8
    """

    array = np.asarray(frames)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D frame array, got shape {array.shape}")

    if array.shape[-1] not in (1, 3, 4) and array.shape[1] in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))

    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def candidate_frame_dirs(input_hdf5: Path, frames_root: Path | None = None) -> list[Path]:
    """根据 episode HDF5 路径推测可能的 frame 目录。"""

    candidates = [
        input_hdf5.with_suffix(""),
        input_hdf5.parent / input_hdf5.stem,
        input_hdf5.parent / f"{input_hdf5.stem}_frames",
    ]
    if frames_root is not None:
        candidates.append(frames_root / input_hdf5.stem)
    return candidates


def list_image_files(directory: Path) -> list[Path]:
    """列出目录下按文件名排序的图片文件。"""

    if not directory.exists() or not directory.is_dir():
        return []
    images = [path for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    return images
