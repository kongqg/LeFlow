#!/usr/bin/env python3
"""把 multi-episode HDF5 拆成 per-episode HDF5。

Mimic 生成的 HDF5 里通常包含多条 demo。后续 render、Cosmos、LeRobot 导出
更适合以单条 episode 为单位处理，所以这里负责把一个大 HDF5 拆成多个小 HDF5。
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError as error:
    raise SystemExit("split_hdf5.py requires h5py. Install with: pip install h5py") from error

from stage_utils import copy_attrs, detect_episode_container, ensure_dir, safe_episode_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a multi-episode HDF5 into per-episode HDF5 files")
    parser.add_argument("--input", required=True, help="Source HDF5 file")
    parser.add_argument("--output-dir", required=True, help="Directory for per-episode HDF5 files")
    parser.add_argument(
        "--episodes",
        default="",
        help="Optional comma-separated episode names. Default: split all detected episodes",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files")
    return parser.parse_args()


def copy_single_episode(source_path: Path, output_dir: Path, episode_name: str, overwrite: bool) -> Path:
    """从源 HDF5 中复制单条 episode 到一个新 HDF5 文件。"""

    with h5py.File(source_path, "r") as source:
        container_name, episode_names = detect_episode_container(source)
        if episode_name not in episode_names:
            raise SystemExit(f"Episode '{episode_name}' not found in {source_path}")

        output_path = output_dir / f"{safe_episode_name(episode_name)}.hdf5"
        if output_path.exists() and not overwrite:
            return output_path

        with h5py.File(output_path, "w") as target:
            # 保留根节点 attrs，避免任务名、版本等元信息在 split 后丢失。
            copy_attrs(source, target)

            if container_name == "/":
                # episode 直接在根目录下：只复制目标 episode，跳过其他 episode，
                # 但保留非 episode 的全局 group/dataset。
                for name, obj in source.items():
                    if name == episode_name:
                        source.copy(name, target, name=name)
                        continue
                    if name in episode_names:
                        continue
                    source.copy(name, target, name=name)
                return output_path

            # episode 放在 data/episodes/demonstrations 等容器下：
            # 先复制容器以外的全局内容，再在目标文件中重建容器并放入单条 episode。
            for name, obj in source.items():
                if name == container_name:
                    continue
                source.copy(name, target, name=name)

            source_container = source[container_name]
            target_container = target.create_group(container_name)
            copy_attrs(source_container, target_container)
            source_container.copy(episode_name, target_container, name=episode_name)

        return output_path


def main() -> int:
    args = parse_args()
    source_path = Path(args.input).expanduser().resolve()
    output_dir = ensure_dir(args.output_dir)

    with h5py.File(source_path, "r") as source:
        _, detected_episode_names = detect_episode_container(source)

    requested = [item.strip() for item in args.episodes.split(",") if item.strip()]
    episode_names = requested or detected_episode_names

    for episode_name in episode_names:
        output_path = copy_single_episode(source_path, output_dir, episode_name, overwrite=args.overwrite)
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
