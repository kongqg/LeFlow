#!/usr/bin/env python3
"""调用 Cosmos 服务处理视频。

这个脚本不实现视频生成模型，只负责把输入 MP4 交给外部服务，并把服务返回的结果
保存到指定输出路径。pipeline 的 cosmos stage 通常会调用这个文件。
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the local Cosmos proxy service and copy the processed MP4")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--server-url", default="http://127.0.0.1:5000", help="Base URL for the Cosmos proxy")
    parser.add_argument("--endpoint", default="/process_video", help="Processing endpoint on the proxy")
    parser.add_argument("--prompt", default="", help="Prompt used by the async NVIDIA Cosmos canny API")
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to the proxy")
    parser.add_argument("--control-weight", type=float, default=0.7, help="Proxy control weight")
    parser.add_argument("--sigma-max", type=float, default=0.2, help="Proxy sigma max")
    parser.add_argument("--canny-strength", default="0.5", help="Proxy canny strength")
    parser.add_argument("--timeout", type=int, default=7200, help="HTTP timeout or max poll time in seconds")
    parser.add_argument("--poll-interval", type=int, default=10, help="Polling interval for async Cosmos jobs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output MP4 if it exists")
    return parser.parse_args()


def build_url(server_url: str, endpoint: str) -> str:
    """拼接服务地址和接口路径。"""

    return server_url.rstrip("/") + "/" + endpoint.strip("/")


def copy_response_video(response: requests.Response, output_path: Path) -> None:
    """把服务返回的视频流写到本地文件。"""

    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def run_process_video(args: argparse.Namespace, input_path: Path, output_path: Path) -> int:
    """处理同步接口：请求结束后，服务直接返回处理后视频的本地路径。"""

    payload = {
        "video_path": str(input_path),
        "seed": args.seed,
        "control_weight": args.control_weight,
        "sigma_max": args.sigma_max,
        "canny_strength": args.canny_strength,
    }
    endpoint = build_url(args.server_url, args.endpoint)

    response = requests.post(endpoint, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()

    processed_path = data.get("processed_video_path")
    if not processed_path:
        raise SystemExit(f"Cosmos proxy response did not include processed_video_path: {data}")

    processed_video = Path(processed_path).expanduser().resolve()
    if not processed_video.exists():
        raise SystemExit(f"Cosmos proxy returned a non-existent output path: {processed_video}")

    if processed_video != output_path:
        shutil.copy2(processed_video, output_path)

    print(output_path)
    return 0


def run_async_canny(args: argparse.Namespace, input_path: Path, output_path: Path) -> int:
    """处理异步接口：先提交任务，再轮询状态，完成后下载结果。"""

    if not args.prompt:
        raise SystemExit("The /canny/submit API requires --prompt; set cosmos.args.prompt in the pipeline config")

    submit_url = build_url(args.server_url, args.endpoint)
    if not submit_url.rstrip("/").endswith("/submit"):
        raise SystemExit(f"Unsupported async Cosmos endpoint: {submit_url}")

    endpoint_root = submit_url.rstrip("/").rsplit("/", 1)[0]
    status_url = f"{endpoint_root}/status"
    result_url = f"{endpoint_root}/result"

    with input_path.open("rb") as handle:
        response = requests.post(
            submit_url,
            data={
                "prompt": args.prompt,
                "sigma_max": str(args.sigma_max),
                "control_weight": str(args.control_weight),
                "canny_strength": str(args.canny_strength),
                "seed": str(args.seed),
            },
            files={"video": (input_path.name, handle, "video/mp4")},
            timeout=min(args.timeout, 90),
        )
    response.raise_for_status()
    submit_data = response.json()

    job_id = submit_data.get("job_id")
    if not job_id:
        raise SystemExit(f"Cosmos submit response did not include job_id: {submit_data}")

    deadline = time.monotonic() + args.timeout
    while True:
        if time.monotonic() > deadline:
            raise SystemExit(f"Timed out waiting for Cosmos job {job_id} after {args.timeout} seconds")

        status_response = requests.get(f"{status_url}/{job_id}", timeout=min(args.poll_interval + 5, 30))
        status_response.raise_for_status()
        status_data = status_response.json()
        status = str(status_data.get("status", "")).lower()

        if status == "completed":
            break
        if status == "failed":
            error = status_data.get("error", "unknown error")
            raise SystemExit(f"Cosmos job {job_id} failed: {error}")

        time.sleep(args.poll_interval)

    with requests.get(f"{result_url}/{job_id}", stream=True, timeout=min(args.timeout, 300)) as result_response:
        result_response.raise_for_status()
        copy_response_video(result_response, output_path)

    print(output_path)
    return 0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(output_path)
        return 0

    normalized_endpoint = "/" + args.endpoint.strip("/")
    if normalized_endpoint.endswith("/submit"):
        return run_async_canny(args, input_path, output_path)

    return run_process_video(args, input_path, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
