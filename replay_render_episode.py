#!/usr/bin/env python3
"""通过 Isaac Lab replay 脚本重放 HDF5，并编码成 MP4。

这个脚本适合需要从仿真里重新 replay trajectory 的情况。它会调用 Isaac Lab 的
replay_demos_record.py，在临时目录里生成 replay frames，然后再把这些帧合成视频。

注意：它和 render_mimic_frames.py 不一样。render_mimic_frames.py 只读取已有 PNG；
这个脚本会重新调用 Isaac Lab replay 工具。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as error:
    raise SystemExit("replay_render_episode.py requires imageio and imageio-ffmpeg") from error

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit("replay_render_episode.py requires numpy") from error

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an Isaac Lab HDF5 demo and encode it as an MP4")
    parser.add_argument("--input", required=True, help="Single-episode HDF5 file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--task", default="", help="Optional Isaac Lab task name override")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--device", default="cuda:0", help="Simulation device passed to replay_demos_record.py")
    parser.add_argument("--num-envs", type=int, default=1, help="Replay environment count")
    parser.add_argument(
        "--replay-script",
        default="../IsaacLab/scripts/tools/replay_demos_record.py",
        help="Path to Isaac Lab replay_demos_record.py",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab without a viewer")
    parser.add_argument("--enable-cameras", action="store_true", help="Enable camera sensors for replay")
    parser.add_argument("--keep-frames", action="store_true", help="Keep replay_frames beside the output video")
    parser.add_argument(
        "--frames-dir",
        default="",
        help="Optional directory to copy replay frames into when --keep-frames is set",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing MP4 output")
    return parser.parse_args()


def resolve_replay_script(path_str: str) -> Path:
    """解析 replay_demos_record.py 的路径。

    相对路径按当前工作目录解析，方便从 repo 根目录运行。
    """

    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def write_video_from_paths(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    """把 replay 生成的 PNG 帧合成 MP4。"""

    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                writer.append_data(np.asarray(image.convert("RGB")))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    replay_script = resolve_replay_script(args.replay_script)

    if not input_path.exists():
        raise SystemExit(f"Input HDF5 does not exist: {input_path}")
    if not replay_script.exists():
        raise SystemExit(f"Replay script does not exist: {replay_script}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        print(output_path)
        return 0

    # 使用临时目录承接 replay_demos_record.py 的中间帧，避免污染当前目录。
    with tempfile.TemporaryDirectory(prefix=f"{input_path.stem}_replay_", dir=str(output_path.parent)) as temp_dir:
        temp_path = Path(temp_dir)
        command = [
            sys.executable,
            str(replay_script),
            "--dataset_file",
            str(input_path),
            "--num_envs",
            str(args.num_envs),
            "--device",
            args.device,
        ]
        if args.task:
            command.extend(["--task", args.task])
        if args.headless:
            command.append("--headless")
        if args.enable_cameras:
            command.append("--enable_cameras")

        subprocess.run(command, check=True, cwd=temp_path)

        replay_frames = temp_path / "replay_frames"
        frame_paths = sorted(replay_frames.glob("*.png"))
        if not frame_paths:
            raise SystemExit(f"No replay frames were generated in {replay_frames}")

        write_video_from_paths(frame_paths, output_path, fps=args.fps)

        if args.keep_frames:
            destination = Path(args.frames_dir).expanduser() if args.frames_dir else output_path.with_suffix("")
            destination = destination.resolve()
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(replay_frames, destination)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
