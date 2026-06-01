#!/usr/bin/env python3
"""把 Mimic 导出的 PNG 相机帧合成 MP4。

这个脚本适用于 generate_data.py / Isaac Lab Mimic 已经把相机帧保存到磁盘的情况。
它不会重新跑仿真，只是根据 episode HDF5 文件名推断 trial index，找到对应 PNG，
然后按 step 顺序编码成视频。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as error:
    raise SystemExit("render_mimic_frames.py requires imageio and imageio-ffmpeg") from error

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit("render_mimic_frames.py requires numpy") from error

from PIL import Image


# 用来从文件名里提取 step 编号，保证视频帧顺序正确。
STEP_PATTERN = re.compile(r"_step_(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an MP4 from Mimic-exported camera frames")
    parser.add_argument("--input", required=True, help="Single-episode HDF5 file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--frames-root", required=True, help="Directory containing Mimic-exported PNG frames")
    parser.add_argument("--camera-name", default="table_cam", help="Camera prefix used in saved frame names")
    parser.add_argument("--data-type", default="rgb", help="Saved data type suffix in frame names")
    parser.add_argument("--tile", type=int, default=0, help="Tile index in the saved frame names")
    parser.add_argument("--trial-index", type=int, default=-1, help="Override episode trial index; default derives from input stem")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output MP4 if it exists")
    return parser.parse_args()


def infer_trial_index(input_path: Path) -> int:
    """从 episode HDF5 文件名里推断 trial index。

    例如 demo_12.hdf5 会推断出 12。必要时可以通过 --trial-index 手动覆盖。
    """

    matches = re.findall(r"(\d+)", input_path.stem)
    if not matches:
        raise SystemExit(f"Could not infer trial index from episode file name: {input_path.name}")
    return int(matches[-1])


def frame_sort_key(path: Path) -> tuple[int, str]:
    """按 step 数字排序；没有 step 的文件放到最后。"""

    match = STEP_PATTERN.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def collect_frame_paths(frames_root: Path, camera_name: str, data_type: str, trial_index: int, tile: int) -> list[Path]:
    """按 Mimic 的帧命名规则收集当前 episode 对应的 PNG。"""

    pattern = f"{camera_name}_{data_type}_trial_{trial_index}_tile_{tile}_step_*.png"
    return sorted(frames_root.glob(pattern), key=frame_sort_key)


def write_video(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    """把 PNG 序列写成 MP4；逐帧读取避免一次性占用过多内存。"""

    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                writer.append_data(np.asarray(image.convert("RGB")))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()

    trial_index = args.trial_index if args.trial_index >= 0 else infer_trial_index(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(output_path)
        return 0

    frame_paths = collect_frame_paths(
        frames_root=frames_root,
        camera_name=args.camera_name,
        data_type=args.data_type,
        trial_index=trial_index,
        tile=args.tile,
    )
    if not frame_paths:
        raise SystemExit(
            f"No Mimic frames matched in {frames_root} for camera={args.camera_name}, "
            f"data_type={args.data_type}, trial={trial_index}, tile={args.tile}"
        )

    write_video(frame_paths, output_path, fps=args.fps)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
