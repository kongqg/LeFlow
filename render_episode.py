#!/usr/bin/env python3
"""通用 episode 视频渲染脚本。

相比 render_mimic_frames.py，这个脚本不依赖 Mimic 固定的帧命名格式。它支持两种输入：
1. 已经存在的图片帧目录。
2. HDF5 文件内部的 image dataset。

因此它更适合作为 fallback 或处理非 Mimic 标准命名的数据。
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as error:
    raise SystemExit("render_episode.py requires imageio and imageio-ffmpeg. Install with: pip install imageio imageio-ffmpeg") from error

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit("render_episode.py requires numpy. Install with: pip install numpy") from error

from PIL import Image

from stage_utils import (
    DEFAULT_IMAGE_KEYS,
    candidate_frame_dirs,
    find_dataset_path,
    list_image_files,
    looks_like_image_dataset,
    normalize_video_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an episode video from frame files or HDF5 image datasets")
    parser.add_argument("--input", required=True, help="Per-episode HDF5 file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--frames-dir", default="", help="Optional directory that already contains frame images")
    parser.add_argument("--frames-root", default="", help="Optional root directory that contains per-episode frame folders")
    parser.add_argument("--image-key", default="", help="Optional explicit HDF5 dataset path for image frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output MP4 if it exists")
    return parser.parse_args()


def write_video_from_paths(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    """从图片路径列表写视频，逐帧读取避免一次性加载全部图片。"""

    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                writer.append_data(np.asarray(image.convert("RGB")))


def write_video_from_array(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """从 HDF5 里的图像数组写视频。

    normalize_video_array 会统一处理 channels-first/channels-last、float、RGBA 等格式差异。
    """

    normalized = normalize_video_array(frames, np)
    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame in normalized:
            writer.append_data(frame)


def resolve_frame_paths(input_path: Path, frames_dir: str, frames_root: str) -> list[Path]:
    """确定应该从哪个图片目录读取帧。

    优先级：
    1. 用户显式传入 --frames-dir。
    2. 根据 HDF5 文件名和 --frames-root 推测可能的目录。
    3. 如果都找不到，返回空列表，后续尝试从 HDF5 内部读取 image dataset。
    """

    if frames_dir:
        return list_image_files(Path(frames_dir).expanduser().resolve())

    root = Path(frames_root).expanduser().resolve() if frames_root else None
    for candidate in candidate_frame_dirs(input_path, root):
        files = list_image_files(candidate)
        if files:
            return files
    return []


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(output_path)
        return 0

    # 第一选择：从外部图片目录编码视频。
    frame_paths = resolve_frame_paths(input_path, args.frames_dir, args.frames_root)
    if frame_paths:
        write_video_from_paths(frame_paths, output_path, fps=args.fps)
        print(output_path)
        return 0

    # 第二选择：从 HDF5 内部 image dataset 编码视频。
    if h5py is None:
        raise SystemExit(
            "No frame directory was found and h5py is not installed. Install h5py or pass --frames-dir"
        )

    with h5py.File(input_path, "r") as h5_file:
        dataset_path = args.image_key.strip("/") if args.image_key else None
        if dataset_path and dataset_path not in h5_file:
            raise SystemExit(f"Dataset '{dataset_path}' was not found in {input_path}")

        if dataset_path is None:
            dataset_path = find_dataset_path(h5_file, DEFAULT_IMAGE_KEYS, predicate=looks_like_image_dataset)
        if dataset_path is None:
            raise SystemExit(
                "Could not find image frames in the HDF5 file. Pass --frames-dir or --image-key explicitly"
            )

        frames = h5_file[dataset_path][()]

    write_video_from_array(frames, output_path, fps=args.fps)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
