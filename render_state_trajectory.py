#!/usr/bin/env python3
"""把 HDF5 中的状态轨迹渲染成 3D 调试视频。

这个脚本主要用于 debug，不是最终 LeRobot 数据导出的必需步骤。它会读取
end-effector 位置和 cube 位置，把运动轨迹画成 3D animation，帮助快速判断：
- episode 是否明显失败。
- 轨迹是否跳变。
- 物体位置是否正常。
"""
from __future__ import annotations

import argparse

import matplotlib

# 在服务器/headless 环境中渲染视频时，不依赖 GUI backend。
matplotlib.use("Agg")

from pathlib import Path

try:
    import h5py
except ModuleNotFoundError as error:
    raise SystemExit("render_state_trajectory.py requires h5py") from error

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as error:
    raise SystemExit("render_state_trajectory.py requires imageio and imageio-ffmpeg") from error

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit("render_state_trajectory.py requires numpy") from error

import matplotlib.pyplot as plt

from stage_utils import find_dataset_path, get_single_episode_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D trajectory animation from a single-episode HDF5 file")
    parser.add_argument("--input", required=True, help="Single-episode HDF5 file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output MP4 if it exists")
    parser.add_argument("--episode-name", default="", help="Optional episode name override inside the HDF5")
    parser.add_argument("--eef-key", default="", help="Optional dataset path for end-effector positions")
    parser.add_argument("--cube-key", default="", help="Optional dataset path for stacked cube positions")
    return parser.parse_args()


def load_trajectory_arrays(input_path: Path, episode_name: str, eef_key: str, cube_key: str) -> tuple[np.ndarray, np.ndarray]:
    """从单条 episode 中读取 eef 和 cube 轨迹。"""

    with h5py.File(input_path, "r") as h5_file:
        _, _, episode_group = get_single_episode_group(h5_file, episode_name=episode_name or None)

        eef_path = eef_key.strip("/") if eef_key else find_dataset_path(episode_group, ["obs/eef_pos", "eef_pos"])
        cube_path = cube_key.strip("/") if cube_key else find_dataset_path(
            episode_group,
            ["obs/cube_positions", "cube_positions"],
        )

        if not eef_path:
            raise SystemExit(f"Could not find end-effector positions in {input_path}")
        if not cube_path:
            raise SystemExit(f"Could not find cube positions in {input_path}")

        eef_pos = np.asarray(episode_group[eef_path][()])
        cube_pos = np.asarray(episode_group[cube_path][()])

    if eef_pos.ndim != 2 or eef_pos.shape[1] != 3:
        raise SystemExit(f"Expected eef_pos shape (T, 3), got {eef_pos.shape}")
    if cube_pos.ndim != 2 or cube_pos.shape[1] % 3 != 0 or cube_pos.shape[1] < 9:
        raise SystemExit(f"Expected cube_positions shape (T, 9+), got {cube_pos.shape}")

    # 这里只取前三个 cube 的 xyz；对于 stack-cube 类任务足够做快速检查。
    return eef_pos, cube_pos[:, :9]


def set_scatter(scatter: any, point: np.ndarray) -> None:
    """更新 3D scatter 的坐标。

    matplotlib 3D scatter 没有普通 set_offsets 接口，所以需要设置内部 _offsets3d。
    """

    scatter._offsets3d = ([float(point[0])], [float(point[1])], [float(point[2])])


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(output_path)
        return 0

    eef_pos, cube_pos = load_trajectory_arrays(input_path, args.episode_name, args.eef_key, args.cube_key)
    cube1 = cube_pos[:, 0:3]
    cube2 = cube_pos[:, 3:6]
    cube3 = cube_pos[:, 6:9]

    # 根据所有轨迹点自动确定坐标范围，保证不同 episode 都能完整显示。
    all_pts = np.vstack([eef_pos, cube1, cube2, cube3])
    mins = all_pts.min(axis=0) - 0.03
    maxs = all_pts.max(axis=0) + 0.03

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    eef_line, = ax.plot([], [], [], linewidth=2, label="eef")
    c1_line, = ax.plot([], [], [], linewidth=1, label="cube1")
    c2_line, = ax.plot([], [], [], linewidth=1, label="cube2")
    c3_line, = ax.plot([], [], [], linewidth=1, label="cube3")

    eef_dot = ax.scatter([], [], [], s=50)
    c1_dot = ax.scatter([], [], [], s=40)
    c2_dot = ax.scatter([], [], [], s=40)
    c3_dot = ax.scatter([], [], [], s=40)

    title = ax.set_title(f"{input_path.stem} | frame 0/{len(eef_pos) - 1}")
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=45)
    fig.tight_layout()

    with imageio.get_writer(output_path, fps=args.fps, macro_block_size=None) as writer:
        for index in range(len(eef_pos)):
            # 每一帧都画“历史轨迹线 + 当前点”，更容易观察运动方向。
            eef_line.set_data(eef_pos[: index + 1, 0], eef_pos[: index + 1, 1])
            eef_line.set_3d_properties(eef_pos[: index + 1, 2])

            c1_line.set_data(cube1[: index + 1, 0], cube1[: index + 1, 1])
            c1_line.set_3d_properties(cube1[: index + 1, 2])

            c2_line.set_data(cube2[: index + 1, 0], cube2[: index + 1, 1])
            c2_line.set_3d_properties(cube2[: index + 1, 2])

            c3_line.set_data(cube3[: index + 1, 0], cube3[: index + 1, 1])
            c3_line.set_3d_properties(cube3[: index + 1, 2])

            set_scatter(eef_dot, eef_pos[index])
            set_scatter(c1_dot, cube1[index])
            set_scatter(c2_dot, cube2[index])
            set_scatter(c3_dot, cube3[index])
            title.set_text(f"{input_path.stem} | frame {index}/{len(eef_pos) - 1}")

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    plt.close(fig)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
