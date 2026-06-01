#!/usr/bin/env python3
"""LeFlow 主调度器。

这个文件是整个仓库的入口，负责把配置文件里的 stage 串起来执行：

    mimic -> split -> render -> cosmos -> lerobot

它本身不实现 Mimic、Cosmos 或 LeRobot，只做三件核心事情：
1. 解析配置和命令行参数。
2. 根据占位符把每个 stage 的命令拼出来。
3. 通过 manifest 记录中间产物，支持跳过已完成步骤和续跑。
"""
from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import h5py
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    tomllib = None

from stage_utils import detect_episode_container

# pipeline 的固定逻辑顺序；--stages all 会展开成这个顺序。
STAGE_ORDER = ("mimic", "split", "render", "cosmos", "lerobot")

# 如果用户只配置 background_variants / lighting_variants，而没有显式写 prompt_templates，
# 这里提供默认 prompt 模板，保证 Cosmos 变体语义是“只改变背景/光照，不改变动作”。
DEFAULT_BACKGROUND_PROMPT_TEMPLATE = (
    "{instruction}. Keep the robot motion, object motion, and contact sequence unchanged. "
    "Only change the background, scene texture, and table appearance. Preserve lighting, camera viewpoint, framing, and geometry."
)
DEFAULT_LIGHTING_PROMPT_TEMPLATE = (
    "{instruction}. Keep the robot motion, object motion, and contact sequence unchanged. "
    "Only change lighting, shadows, brightness, and color temperature. Preserve background, camera viewpoint, framing, and geometry."
)


class SafeDict(dict):
    """占位符展开用的安全字典。

    普通 str.format 如果遇到未知占位符会直接报错。这里保留未知占位符本身，
    这样可以允许某些字段在后续 stage 才被补进 context。
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass
class SourceItem:
    """一个基础 episode 的记录。

    render 阶段以前只需要 HDF5 和 instruction；render 后会补上 base_video。
    """

    episode_id: str
    source_hdf5: str
    instruction: str
    base_video: str | None = None


@dataclass
class VariantItem:
    """一个最终可导出的训练样本记录。

    如果启用了 Cosmos，一个 SourceItem 会展开成多个 VariantItem；
    如果没启用 Cosmos，则 render 视频会被包装成一个 variant_index=0 的 VariantItem。
    """

    episode_id: str
    source_hdf5: str
    instruction: str
    base_video: str
    video_path: str
    variant_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mimic -> render -> Cosmos -> LeRobot pipeline.")
    parser.add_argument("--config", required=True, help="Path to the pipeline JSON or TOML config")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages to run: mimic,split,render,cosmos,lerobot or all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun stages")
    parser.add_argument(
        "--cosmos-workers",
        type=int,
        default=None,
        help="Maximum number of Cosmos jobs to run in parallel. Overrides [cosmos].parallelism.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON / TOML 配置文件。"""

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if tomllib is None:
        raise SystemExit("TOML config requires Python 3.11+; use the JSON example on this machine instead")

    with path.open("rb") as handle:
        return tomllib.load(handle)


def normalize_stage_list(raw: str) -> list[str]:
    """把 --stages 参数解析成合法 stage 列表。"""

    if raw == "all":
        return list(STAGE_ORDER)
    stages = [stage.strip() for stage in raw.split(",") if stage.strip()]
    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise SystemExit(f"Invalid stages: {', '.join(invalid)}")
    return stages


def stage_enabled(config: dict[str, Any], stage: str) -> bool:
    """判断某个 stage 是否启用。

    约定：section 不存在视为禁用；section 存在但没写 enabled 时默认启用。
    """

    section = config.get(stage)
    if not section:
        return False
    return bool(section.get("enabled", True))


def build_base_context(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    """构造所有 stage 共用的基础占位符上下文。"""

    config_dir = config_path.parent
    run_cfg = config.get("run", {})
    run_name = run_cfg.get("name", "robot-pipeline")
    work_dir_template = run_cfg.get("work_dir", str(config_dir / "runs" / run_name))

    # work_dir 本身可能用到 {config_dir} / {run_name}，所以先用一个最小 context 展开。
    seed_context = {
        "cwd": ".",
        "config_dir": str(config_dir),
        "run_name": run_name,
    }
    work_dir = Path(expand(work_dir_template, seed_context)).expanduser()
    manifests_dir = work_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "cwd": ".",
        "config_dir": str(config_dir),
        "run_name": run_name,
        "work_dir": str(work_dir),
        "manifests_dir": str(manifests_dir),
        "mimic_output": str(work_dir / "mimic" / "generated_dataset.hdf5"),
        "lerobot_root": str(work_dir / "lerobot"),
    }
    return add_shell_context(context)


def add_shell_context(context: dict[str, Any]) -> dict[str, str]:
    """为每个占位符额外生成一个 shell-safe 版本。

    例如 context 里有 work_dir，就同时生成 work_dir_shell。这样配置里如果需要
    手写复杂 shell 命令，也可以选择使用已经 shlex.quote 过的路径。
    """

    expanded: dict[str, str] = {}
    for key, value in context.items():
        if value is None:
            continue
        string_value = str(value)
        expanded[key] = string_value
        expanded[f"{key}_shell"] = shlex.quote(string_value)
    return expanded


def expand(template: str, context: dict[str, Any]) -> str:
    """展开配置里的 {placeholder}。"""

    return template.format_map(SafeDict(add_shell_context(context)))


def expand_arg_value(value: Any, context: dict[str, Any]) -> str:
    if isinstance(value, str):
        return expand(value, context)
    return str(value)


def render_cli_args(args_cfg: dict[str, Any], context: dict[str, Any]) -> str:
    """把配置里的 args dict 渲染成命令行参数字符串。

    规则：
    - 普通值：key=value -> --key value
    - bool true：key=true -> --key
    - bool false / None：不输出
    - list：同一个 key 重复输出多次，适合 --state-key 这种参数
    """

    parts: list[str] = []

    for raw_flag, raw_value in args_cfg.items():
        flag = str(raw_flag)
        option = flag if flag.startswith("-") else f"--{flag}"

        if raw_value is None:
            continue

        if isinstance(raw_value, bool):
            if raw_value:
                parts.append(shlex.quote(option))
            continue

        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    parts.append(shlex.quote(option))
                continue
            parts.append(shlex.quote(option))
            parts.append(shlex.quote(expand_arg_value(value, context)))

    return " ".join(parts)


def build_stage_command(stage_cfg: dict[str, Any], context: dict[str, Any]) -> str:
    """根据 stage 配置生成最终要执行的 shell 命令。"""

    command_template = stage_cfg.get("command")
    if not command_template:
        return ""

    command = expand(command_template, context)
    args_cfg = stage_cfg.get("args", {})
    if not args_cfg:
        return command
    if not isinstance(args_cfg, dict):
        raise SystemExit("[pipeline] stage args must be a JSON/TOML object")

    cli_args = render_cli_args(args_cfg, context)
    if not cli_args:
        return command
    return f"{command} {cli_args}"


def print_step(message: str) -> None:
    print(f"[pipeline] {message}")


def run_command(command: str, dry_run: bool, cwd: Path) -> None:
    """统一执行 stage 命令；dry-run 时只打印不执行。"""

    print_step(command)
    if dry_run:
        return
    subprocess.run(command, shell=True, check=True, cwd=str(cwd), executable="/bin/bash")


def resolve_cosmos_workers(cosmos_cfg: dict[str, Any], override: int | None) -> int:
    """确定 Cosmos 并发数，CLI 参数优先级高于配置文件。"""

    raw_value = override
    field_name = "--cosmos-workers"

    if raw_value is None:
        raw_value = cosmos_cfg.get("parallelism", cosmos_cfg.get("workers", 1))
        field_name = "[cosmos].parallelism"

    try:
        workers = int(raw_value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{field_name} must be an integer") from error

    if workers < 1:
        raise SystemExit(f"{field_name} must be >= 1")

    return workers


def resolve_cosmos_prompt_templates(cosmos_cfg: dict[str, Any]) -> list[str]:
    """解析 Cosmos 多变体 prompt 模板。

    优先使用用户显式配置的 prompt_templates/prompts；如果没有，就根据
    background_variants 和 lighting_variants 自动生成默认模板。
    """

    explicit_templates = cosmos_cfg.get("prompt_templates", cosmos_cfg.get("prompts", []))
    if explicit_templates:
        if not isinstance(explicit_templates, list):
            raise SystemExit("[cosmos].prompt_templates must be a list of prompt strings")
        return [str(template) for template in explicit_templates]

    background_variants = int(cosmos_cfg.get("background_variants", 0))
    lighting_variants = int(cosmos_cfg.get("lighting_variants", 0))
    if background_variants < 0 or lighting_variants < 0:
        raise SystemExit("[cosmos].background_variants and [cosmos].lighting_variants must be >= 0")

    if background_variants == 0 and lighting_variants == 0:
        return []

    background_prompt = str(cosmos_cfg.get("background_prompt_template", DEFAULT_BACKGROUND_PROMPT_TEMPLATE))
    lighting_prompt = str(cosmos_cfg.get("lighting_prompt_template", DEFAULT_LIGHTING_PROMPT_TEMPLATE))
    return ([background_prompt] * background_variants) + ([lighting_prompt] * lighting_variants)


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def hdf5_has_episodes(path: Path) -> bool:
    """轻量判断一个 HDF5 是否真的包含 episode group。

    这里主要服务 split fallback：有些 Mimic 失败时会生成 *_failed.hdf5，
    原 output 可能没有有效 episode，所以需要判断文件是否真的可拆。
    """

    if h5py is None or not path.exists():
        return False

    try:
        with h5py.File(path, "r") as handle:
            container_name, episode_names = detect_episode_container(handle)

            if container_name == "/":
                groups = [handle[name] for name in episode_names if name in handle]
            else:
                container = handle[container_name]
                groups = [container[name] for name in episode_names if name in container]

            for group in groups:
                try:
                    if len(group.keys()) > 0:
                        return True
                except Exception:
                    continue
    except Exception:
        return False

    return False


def resolve_split_input(path_str: str) -> str:
    """解析 split 的输入文件。

    如果默认 mimic_output 没有有效 episode，但旁边存在 *_failed.hdf5 且里面有 episode，
    就自动 fallback 到 failed 文件，方便排查和继续处理失败样本。
    """

    source_path = Path(path_str).expanduser()
    if hdf5_has_episodes(source_path):
        return str(source_path)

    if source_path.stem.endswith("_failed"):
        return str(source_path)

    failed_path = source_path.with_name(f"{source_path.stem}_failed{source_path.suffix}")
    if hdf5_has_episodes(failed_path):
        print_step(f"split input fallback: using {failed_path} because {source_path} has no episodes")
        return str(failed_path)

    return str(source_path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """覆写 JSONL manifest。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；不存在时返回空列表，方便续跑逻辑判断。"""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_instruction_map(config_path: Path, config: dict[str, Any], base_context: dict[str, str]) -> dict[str, str]:
    """读取按 episode 指定的 instruction 映射。

    支持两种格式：
    - JSON dict: {"demo_0": "..."}
    - JSON/JSONL records: {"episode_id": "demo_0", "instruction": "..."}
    """

    instructions_cfg = config.get("instructions", {})
    file_value = instructions_cfg.get("file")
    if not file_value:
        return {}

    file_path = Path(expand(file_value, base_context)).expanduser()

    if file_path.suffix == ".jsonl":
        mapping: dict[str, str] = {}
        for record in read_jsonl(file_path):
            episode_id = record.get("episode_id")
            instruction = record.get("instruction")
            if episode_id and instruction:
                mapping[str(episode_id)] = str(instruction)
        return mapping

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}

    if isinstance(payload, list):
        mapping = {}
        for record in payload:
            episode_id = record.get("episode_id")
            instruction = record.get("instruction")
            if episode_id and instruction:
                mapping[str(episode_id)] = str(instruction)
        return mapping

    raise SystemExit(f"Unsupported instruction format: {file_path}")


def discover_sources(config_path: Path, config: dict[str, Any], base_context: dict[str, str]) -> list[SourceItem]:
    """发现本次要处理的 per-episode HDF5 列表。"""

    sources_cfg = config.get("sources", {})
    source_glob = sources_cfg.get("glob")
    if not source_glob:
        split_cfg = config.get("split", {})
        source_glob = split_cfg.get("output_glob")
    if not source_glob:
        raise SystemExit("Missing [sources].glob or [split].output_glob in the config")

    pattern = expand(source_glob, base_context)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"No source HDF5 files matched: {pattern}")

    instruction_map = load_instruction_map(config_path, config, base_context)
    default_instruction = config.get("instructions", {}).get("default", "")
    seen_ids: dict[str, int] = {}
    sources: list[SourceItem] = []

    for match in matches:
        path = Path(match).expanduser()
        base_id = path.stem

        # 如果两个文件 stem 相同，用 _01/_02 避免 episode_id 冲突。
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        episode_id = base_id if count == 0 else f"{base_id}_{count:02d}"
        instruction = instruction_map.get(base_id) or instruction_map.get(episode_id) or default_instruction
        sources.append(
            SourceItem(
                episode_id=episode_id,
                source_hdf5=str(path),
                instruction=instruction,
            )
        )

    return sources


def sources_manifest_path(base_context: dict[str, str]) -> Path:
    return Path(base_context["manifests_dir"]) / "sources.jsonl"


def rendered_manifest_path(base_context: dict[str, str]) -> Path:
    return Path(base_context["manifests_dir"]) / "rendered.jsonl"


def cosmos_manifest_path(base_context: dict[str, str]) -> Path:
    return Path(base_context["manifests_dir"]) / "cosmos.jsonl"


def lerobot_manifest_path(config: dict[str, Any], base_context: dict[str, str]) -> Path:
    lerobot_cfg = config.get("lerobot", {})
    manifest_value = lerobot_cfg.get("manifest_path", str(Path(base_context["manifests_dir"]) / "lerobot_pairs.jsonl"))
    return Path(expand(manifest_value, base_context)).expanduser()


def load_or_discover_sources(config_path: Path, config: dict[str, Any], base_context: dict[str, str]) -> list[SourceItem]:
    """优先从 manifest 恢复 sources，否则重新 glob 发现。"""

    manifest = sources_manifest_path(base_context)
    records = read_jsonl(manifest)
    if records:
        return [SourceItem(**record) for record in records]
    sources = discover_sources(config_path, config, base_context)
    write_jsonl(manifest, [asdict(item) for item in sources])
    return sources


def load_rendered_items(base_context: dict[str, str]) -> list[SourceItem]:
    return [SourceItem(**record) for record in read_jsonl(rendered_manifest_path(base_context))]


def load_variant_items(base_context: dict[str, str]) -> list[VariantItem]:
    return [VariantItem(**record) for record in read_jsonl(cosmos_manifest_path(base_context))]


def run_single_stage(stage_name: str, config: dict[str, Any], base_context: dict[str, str], dry_run: bool, force: bool) -> None:
    """运行 mimic / split 这类“一次命令产出一批结果”的 stage。"""

    stage_cfg = config.get(stage_name, {})
    if not stage_cfg.get("command"):
        print_step(f"skip {stage_name}: no command configured")
        return

    output_key = stage_cfg.get("batch_output_key")
    output_template = stage_cfg.get("batch_output")
    if output_template and not output_key:
        output_key = f"{stage_name}_output"

    stage_context = dict(base_context)
    if output_template and output_key:
        output_path = ensure_parent(expand(output_template, stage_context))
        stage_context[output_key] = str(output_path)
        stage_context = add_shell_context(stage_context)

        # batch_output 已存在时默认跳过；--force 才重跑。
        if output_path.exists() and not force:
            print_step(f"skip {stage_name}: {output_path} exists")
            return

    if stage_name == "split":
        args_cfg = stage_cfg.get("args", {})
        if isinstance(args_cfg, dict) and args_cfg.get("input"):
            resolved_input = expand(str(args_cfg["input"]), stage_context)
            split_input = resolve_split_input(resolved_input)
            if split_input != resolved_input:
                # 不直接改原 config，避免影响后续 stage；只为当前 split 命令临时注入 split_input。
                stage_cfg = dict(stage_cfg)
                stage_args = dict(args_cfg)
                stage_args["input"] = "{split_input}"
                stage_cfg["args"] = stage_args
                stage_context["split_input"] = split_input
                stage_context = add_shell_context(stage_context)

    command = build_stage_command(stage_cfg, stage_context)
    run_command(command, dry_run=dry_run, cwd=Path(base_context["cwd"]))


def run_render_stage(
    config_path: Path,
    config: dict[str, Any],
    base_context: dict[str, str],
    dry_run: bool,
    force: bool,
) -> list[SourceItem]:
    """逐个 source HDF5 渲染基础视频，并写 rendered.jsonl。"""

    render_cfg = config.get("render", {})
    output_pattern = render_cfg.get("output_pattern")
    if not render_cfg.get("command") or not output_pattern:
        raise SystemExit("[render] requires command and output_pattern")

    sources = discover_sources(config_path, config, base_context)
    if not dry_run:
        write_jsonl(sources_manifest_path(base_context), [asdict(item) for item in sources])
    rendered: list[SourceItem] = []
    command_cwd = Path(base_context["cwd"])

    for item in sources:
        item_context = dict(base_context)
        item_context.update(asdict(item))
        base_video = ensure_parent(expand(output_pattern, item_context))
        item_context["base_video"] = str(base_video)
        item_context = add_shell_context(item_context)

        if base_video.exists() and not force:
            print_step(f"skip render for {item.episode_id}: {base_video} exists")
        else:
            run_command(build_stage_command(render_cfg, item_context), dry_run=dry_run, cwd=command_cwd)

        rendered.append(
            SourceItem(
                episode_id=item.episode_id,
                source_hdf5=item.source_hdf5,
                instruction=item.instruction,
                base_video=str(base_video),
            )
        )

    if not dry_run:
        write_jsonl(rendered_manifest_path(base_context), [asdict(item) for item in rendered])
    return rendered


def run_cosmos_stage(
    base_context: dict[str, str],
    config: dict[str, Any],
    dry_run: bool,
    force: bool,
    cosmos_workers: int | None = None,
) -> list[VariantItem]:
    """基于 rendered.jsonl 逐个生成 Cosmos 视频变体。"""

    cosmos_cfg = config.get("cosmos", {})
    output_pattern = cosmos_cfg.get("output_pattern")
    prompt_templates = resolve_cosmos_prompt_templates(cosmos_cfg)
    if prompt_templates:
        configured_variants = int(cosmos_cfg.get("variants", len(prompt_templates)))
        if configured_variants != len(prompt_templates):
            raise SystemExit(
                f"[cosmos].variants ({configured_variants}) must match the number of prompt templates ({len(prompt_templates)})"
            )
        variants = len(prompt_templates)
    else:
        variants = int(cosmos_cfg.get("variants", 1))

    rendered_items = load_rendered_items(base_context)
    if not rendered_items:
        raise SystemExit("rendered manifest is missing; run render first or restore manifests/rendered.jsonl")

    if variants < 1:
        raise SystemExit("[cosmos].variants must be >= 1")

    if not cosmos_cfg.get("command") or not output_pattern:
        raise SystemExit("[cosmos] requires command and output_pattern")

    results: list[VariantItem] = []
    pending_commands: list[tuple[str, str]] = []
    command_cwd = Path(base_context["cwd"])

    for item in rendered_items:
        if not item.base_video:
            raise SystemExit(f"render manifest for {item.episode_id} is missing base_video")
        for variant_index in range(variants):
            variant_episode_id = f"{item.episode_id}__v{variant_index:02d}"
            item_context = dict(base_context)
            item_context.update(asdict(item))
            item_context["variant_index"] = str(variant_index)
            item_context["variant_episode_id"] = variant_episode_id
            variant_video = ensure_parent(expand(output_pattern, item_context))
            item_context["variant_video"] = str(variant_video)
            item_context["video_path"] = str(variant_video)
            variant_prompt = ""
            stage_cfg = cosmos_cfg
            if prompt_templates:
                # prompt 模板按 variant_index 一一对应，适合背景/光照等多变体生成。
                variant_prompt = expand(prompt_templates[variant_index], item_context)
                item_context["variant_prompt"] = variant_prompt
                stage_cfg = dict(cosmos_cfg)
                stage_args = dict(cosmos_cfg.get("args", {}))
                stage_args["prompt"] = variant_prompt
                stage_cfg["args"] = stage_args
            item_context = add_shell_context(item_context)

            if variant_video.exists() and not force:
                print_step(f"skip cosmos for {variant_episode_id}: {variant_video} exists")
            else:
                pending_commands.append((variant_episode_id, build_stage_command(stage_cfg, item_context)))

            results.append(
                VariantItem(
                    episode_id=variant_episode_id,
                    source_hdf5=item.source_hdf5,
                    instruction=item.instruction,
                    base_video=item.base_video,
                    video_path=str(variant_video),
                    variant_index=variant_index,
                )
            )

    workers = resolve_cosmos_workers(cosmos_cfg, cosmos_workers)
    if pending_commands:
        if dry_run or workers == 1:
            for _, command in pending_commands:
                run_command(command, dry_run=dry_run, cwd=command_cwd)
        else:
            worker_count = min(workers, len(pending_commands))
            print_step(f"run cosmos in parallel with {worker_count} workers across {len(pending_commands)} jobs")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(run_command, command, False, command_cwd): episode_id
                    for episode_id, command in pending_commands
                }
                for future in as_completed(futures):
                    episode_id = futures[future]
                    future.result()
                    print_step(f"completed cosmos for {episode_id}")

    if not dry_run:
        write_jsonl(cosmos_manifest_path(base_context), [asdict(item) for item in results])
    return results


def build_training_items(base_context: dict[str, str], config: dict[str, Any]) -> list[VariantItem]:
    """决定 LeRobot 阶段到底使用哪些视频。

    优先级：
    1. 如果启用 Cosmos 且 cosmos.jsonl 存在，用 Cosmos 视频；
    2. 否则回退到 render 生成的 base_video。
    """

    if stage_enabled(config, "cosmos"):
        variant_items = load_variant_items(base_context)
        if variant_items:
            missing_videos = [item.video_path for item in variant_items if not Path(item.video_path).expanduser().exists()]
            if missing_videos:
                preview = ", ".join(missing_videos[:3])
                suffix = "" if len(missing_videos) <= 3 else f" and {len(missing_videos) - 3} more"
                raise SystemExit(
                    f"cosmos manifest references missing videos: {preview}{suffix}; rerun cosmos with --force or remove the stale manifest"
                )
            return variant_items

    rendered_items = load_rendered_items(base_context)
    return [
        VariantItem(
            episode_id=item.episode_id,
            source_hdf5=item.source_hdf5,
            instruction=item.instruction,
            base_video=item.base_video or "",
            video_path=item.base_video or "",
            variant_index=0,
        )
        for item in rendered_items
        if item.base_video
    ]


def run_lerobot_stage(base_context: dict[str, str], config: dict[str, Any], dry_run: bool, force: bool) -> None:
    """把 HDF5 + video + instruction 逐条导出成 LeRobot 格式。"""

    items = build_training_items(base_context, config)
    if not items:
        raise SystemExit("No rendered or Cosmos video items are available for LeRobot export")

    lerobot_cfg = config.get("lerobot", {})
    lerobot_root = Path(expand(lerobot_cfg.get("root", base_context["lerobot_root"]), base_context)).expanduser()
    lerobot_root.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    command_cwd = Path(base_context["cwd"])

    for item in items:
        item_context = dict(base_context)
        item_context.update(asdict(item))
        item_context["final_episode_id"] = item.episode_id
        item_context["lerobot_root"] = str(lerobot_root)
        item_context = add_shell_context(item_context)
        manifest_records.append(
            {
                "episode_id": item.episode_id,
                "source_hdf5": item.source_hdf5,
                "video_path": item.video_path,
                "instruction": item.instruction,
                "variant_index": item.variant_index,
                "lerobot_root": str(lerobot_root),
            }
        )

        if not lerobot_cfg.get("command"):
            continue

        success_marker = lerobot_cfg.get("success_marker")
        if success_marker:
            marker_path = ensure_parent(expand(success_marker, item_context))
            if marker_path.exists() and not force:
                print_step(f"skip lerobot for {item.episode_id}: {marker_path} exists")
                continue

        run_command(build_stage_command(lerobot_cfg, item_context), dry_run=dry_run, cwd=command_cwd)

    write_jsonl(lerobot_manifest_path(config, base_context), manifest_records)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    stages = normalize_stage_list(args.stages)
    base_context = build_base_context(config_path, config)

    for stage in stages:
        if not stage_enabled(config, stage):
            print_step(f"skip {stage}: disabled or missing section")
            continue

        if stage in {"mimic", "split"}:
            run_single_stage(stage, config, base_context, dry_run=args.dry_run, force=args.force)
            continue

        if stage == "render":
            run_render_stage(config_path, config, base_context, dry_run=args.dry_run, force=args.force)
            continue

        if stage == "cosmos":
            run_cosmos_stage(
                base_context,
                config,
                dry_run=args.dry_run,
                force=args.force,
                cosmos_workers=args.cosmos_workers,
            )
            continue

        if stage == "lerobot":
            run_lerobot_stage(base_context, config, dry_run=args.dry_run, force=args.force)
            continue

    print_step(f"done. manifests are under {base_context['manifests_dir']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print_step(f"command failed with exit code {error.returncode}")
        raise SystemExit(error.returncode)
