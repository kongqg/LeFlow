#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac Lab Mimic 数据生成入口。

这个脚本是真正和 Isaac Lab / Isaac Lab Mimic 交互的地方：
1. 启动 Isaac Sim 应用。
2. 构造指定 task 的 Mimic 环境。
3. 从 annotated HDF5 读取源示范。
4. 调用 Isaac Lab Mimic 的异步数据生成流程。
5. 输出 generated_dataset.hdf5，并可选保存相机帧。

通常不需要直接手动调用它，而是通过 run_mimic_multi_gpu.py 或 pipeline_tool.py 间接调用。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import random
import sys

import nest_asyncio

# Isaac Lab Mimic 里会使用 asyncio；在某些交互/嵌套环境中需要允许嵌套 event loop。
nest_asyncio.apply()


def maybe_isolate_device(argv: list[str]) -> None:
    """在导入 Isaac Lab 前，可选地只暴露一张 GPU。

    Isaac Sim 在大机器上有时会花很久做多 GPU P2P validation。启用
    --isolate_device 时，会先设置 CUDA_VISIBLE_DEVICES，只让子进程看到目标 GPU。
    后面再把进程内部的 device remap 成 cuda:0。
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", type=str, default="")
    pre_parser.add_argument("--isolate-device", "--isolate_device", dest="isolate_device", action="store_true")
    pre_args, _ = pre_parser.parse_known_args(argv)

    if not pre_args.isolate_device:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    if not pre_args.device.startswith("cuda:"):
        return

    requested_device = pre_args.device.split(":", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = requested_device


# 必须在导入 Isaac Lab 之前执行 GPU 隔离，否则 Isaac Sim 可能已经初始化 CUDA 设备。
maybe_isolate_device(sys.argv[1:])

from isaaclab.app import AppLauncher


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构造 Mimic 数据生成参数。"""

    parser = argparse.ArgumentParser("Generate Mimic dataset from annotated HDF5")

    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--generation_num_trials", type=int, default=10)
    parser.add_argument(
        "--generation_guarantee",
        type=str,
        default="",
        help="Optional override for datagen_config.generation_guarantee. Use true or false",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=0.0,
        help="Optional override for env_cfg.episode_length_s to shorten or extend each attempt",
    )
    parser.add_argument("--input_file", type=str, default="datasets/annotated_dataset.hdf5")
    parser.add_argument("--output_file", type=str, default="my_data/generated_dataset.hdf5")
    parser.add_argument("--image_root", type=str, default="", help="Root directory for saved camera frames")
    parser.add_argument(
        "--image_data_type",
        type=str,
        default="rgb",
        help="Camera data type to save to disk when image output is enabled; use 'task-default' to keep the task config",
    )
    parser.add_argument("--pause_subtask", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--isolate-device",
        "--isolate_device",
        dest="isolate_device",
        action="store_true",
        default=False,
        help="Set CUDA_VISIBLE_DEVICES from --device before launching Isaac Sim. Useful on multi-GPU servers",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    # AppLauncher 会补充 Isaac Sim/Isaac Lab 自己的参数，比如 --headless、--device 等。
    # 放在任务参数后面，避免对空 parser 添加参数时触发 warning。
    AppLauncher.add_app_launcher_args(parser)
    return parser


def configure_camera_outputs(env_cfg, image_root: str, image_data_type: str) -> None:
    """把环境里的 rgb camera observation 改成保存到磁盘。

    Mimic 原始流程主要生成 HDF5；为了后面 render 成视频，这里会把相机帧以 PNG
    形式保存到 image_root。多个 observation 如果共享同一个 image_path，只保存一次，
    避免重复写相同帧。
    """

    from isaaclab.managers import ObservationTermCfg as ObsTerm

    if not hasattr(env_cfg, "observations") or not hasattr(env_cfg.observations, "rgb_camera"):
        return

    configured_paths: set[str] = set()
    for obs in vars(env_cfg.observations.rgb_camera).values():
        if not isinstance(obs, ObsTerm) or "image_path" not in getattr(obs, "params", {}):
            continue

        base_path = os.path.join(image_root, obs.params["image_path"])
        obs.params["image_path"] = base_path

        if image_data_type == "task-default":
            continue

        if base_path in configured_paths:
            obs.params["save_image_to_file"] = False
            continue

        obs.params["data_type"] = image_data_type
        if image_data_type == "rgb":
            # RGB 帧用于后续视频编码，保持 0-255 更直观。
            obs.params["normalize"] = False
        obs.params["save_image_to_file"] = True
        configured_paths.add(base_path)


def print_debug_summary(args, env_name: str, env_cfg, async_components: dict[str, object], image_root: str) -> None:
    """打印关键运行信息，主要用于排查 Isaac/Mimic 环境问题。"""

    from isaaclab.managers import ObservationTermCfg as ObsTerm

    print(f"[debug] task={env_name}")
    print(f"[debug] device={args.device}, num_envs={args.num_envs}")
    print(f"[debug] headless={args.headless}, display={os.environ.get('DISPLAY', '')}")
    print(f"[debug] isolate_device={args.isolate_device}, cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[debug] input={args.input_file}")
    print(f"[debug] output={args.output_file}")
    print(f"[debug] image_root={image_root}")
    print(f"[debug] image_data_type={args.image_data_type}")
    if getattr(env_cfg, "episode_length_s", None) is not None:
        print(f"[debug] episode_length_s={env_cfg.episode_length_s}")

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "rgb_camera"):
        for name, obs in vars(env_cfg.observations.rgb_camera).items():
            if not isinstance(obs, ObsTerm) or "image_path" not in getattr(obs, "params", {}):
                continue
            print(
                f"[debug] camera_path[{name}]={obs.params['image_path']} "
                f"data_type={obs.params.get('data_type')} "
                f"save={obs.params.get('save_image_to_file')}"
            )

    print(f"[debug] tasks={len(async_components['tasks'])}")


def main() -> int:
    args = build_parser().parse_args()

    if args.isolate_device and args.device.startswith("cuda:"):
        # 由于 CUDA_VISIBLE_DEVICES 已经只暴露一张卡，进程内部应该使用 cuda:0。
        args.device = "cuda:0"

    if not getattr(args, "headless", False) and not os.environ.get("DISPLAY"):
        args.headless = True
    if not getattr(args, "device", None):
        args.device = "cuda:0"
    if not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
    if not getattr(args, "kit_args", None):
        setattr(args, "kit_args", "--enable omni.videoencoding")
    if not getattr(args, "enable", None):
        setattr(args, "enable", "omni.kit.renderer.capture")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # AppLauncher 必须先启动 Isaac Sim app，之后才能 import 依赖 simulation app 的模块。
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        from isaaclab.envs import ManagerBasedRLMimicEnv
        from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
        from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

        import isaaclab_mimic.envs  # noqa: F401
        import isaaclab_tasks  # noqa: F401

        image_root = args.image_root or (os.path.dirname(args.output_file) or ".")
        output_dir, output_file_name = setup_output_paths(args.output_file)
        env_name = args.task or get_env_name_from_dataset(args.input_file)

        env_cfg, success_term = setup_env_config(
            env_name=env_name,
            output_dir=output_dir,
            output_file_name=output_file_name,
            num_envs=args.num_envs,
            device=args.device,
            generation_num_trials=args.generation_num_trials,
        )
        if args.generation_guarantee:
            lowered = args.generation_guarantee.strip().lower()
            if lowered not in {"true", "false"}:
                raise ValueError("--generation_guarantee must be 'true' or 'false'")
            env_cfg.datagen_config.generation_guarantee = lowered == "true"
        if args.episode_length_s > 0:
            env_cfg.episode_length_s = args.episode_length_s

        configure_camera_outputs(env_cfg, image_root=image_root, image_data_type=args.image_data_type)

        env = gym.make(env_name, cfg=env_cfg).unwrapped
        if not isinstance(env, ManagerBasedRLMimicEnv):
            raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
        if args.debug:
            print("[debug] env_created", flush=True)

        if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
            logger.warning(
                'The "noise" parameter in the mimic API "target_eef_pose_to_action" is deprecated. '
                "Update the environment to take action_noise_dict instead."
            )

        seed = args.seed
        if getattr(getattr(env.cfg, "datagen_config", None), "seed", None) is not None:
            seed = env.cfg.datagen_config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.debug:
            print(f"[debug] seed={seed}", flush=True)

        env.reset()
        if args.debug:
            print("[debug] env_reset_done", flush=True)

        # Isaac Lab Mimic 的核心：构造异步生成组件，然后由 env_loop 驱动环境和任务队列。
        async_components = setup_async_generation(
            env=env,
            num_envs=args.num_envs,
            input_file=args.input_file,
            success_term=success_term,
            pause_subtask=args.pause_subtask,
        )
        if args.debug:
            print("[debug] async_setup_done", flush=True)

        if args.debug:
            print_debug_summary(args, env_name, env_cfg, async_components, image_root)

        data_gen_tasks = None
        try:
            data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # env_loop 结束后，确保所有异步任务被取消并清理，避免 Isaac 进程无法退出。
            if data_gen_tasks is not None:
                data_gen_tasks.cancel()
                try:
                    async_components["event_loop"].run_until_complete(data_gen_tasks)
                except asyncio.CancelledError:
                    print("Remaining async tasks cancelled and cleaned up.")
                except Exception as error:  # pragma: no cover - diagnostic path
                    print(f"Error cancelling remaining async tasks: {error}")

        print(f"Done. Generated dataset saved to: {args.output_file}")
        return 0

    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
