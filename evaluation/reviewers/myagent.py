"""myagent 评审器：调用 skills_code_review_agent 框架（trpc-agent-python）。

链路：prepare_repo（clone -> checkout head）-> subprocess 调
run_agent.py --repo-path <clone> --commit-range <base> <head> -> 读
review_report.json -> 把 analysis.findings 转成评测标准 review_output ->
写结果文件 <safe_id>.json。

回退：若 base/head commit 在 GitHub 上悬空（squash-merge PR 的原始 head，
`not our ref`），按原始数据集的 githubPrUrl 直接拉 <pr>.diff，降级为
--diff-file 模式（无仓库上下文，接受低分）。

LLM / Docker 配置由框架自身 .env 提供（run_agent.py 读取）：
  TRPC_AGENT_API_KEY / TRPC_AGENT_BASE_URL / TRPC_AGENT_MODEL_NAME
  CODE_REVIEW_SANDBOX_BACKEND=docker 等。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import config
from repo_utils import RepoError, clean_worktree, diff_stat, log, prepare_repo, run_command
from schema import ReviewInstance

# 服务器迁移：通过环境变量覆盖路径，默认值兼容本地 WSL 布局
TRPC_ROOT = Path(os.getenv("MYAGENT_TRPC_ROOT", "/home/chan/trpc-agent-python"))
AGENT_REL = "examples/skills_code_review_agent/run_agent.py"
PROJECT_REL = "examples/skills_code_review_agent"

# 原始 AACR-Bench 数据（标准 JSONL 转换时丢弃了 githubPrUrl，回退时按
# (repo, base, head) 三元组反查 PR 地址）
RAW_DATASET_PATH = Path(
    os.getenv(
        "MYAGENT_RAW_DATASET",
        "/home/chan/aacr-bench/dataset/positive_samples.json",
    )
)

# 阶段 1 踩坑 #2：框架 .env 将 CODE_REVIEW_MAX_OUTPUT_BYTES 收紧为 15KB，
# 而 _sandbox_request 构造的命令用默认 64KB 预算，Filter 会判超预算拦截全部
# 命令。这里仅为评审进程恢复框架默认值，不改动用户 .env。
# 阶段 3 踩坑：大 diff 样本（lvgl 336 文件、typescript-go）在默认 110s/30 次调用
# 预算内跑不完（config.py 硬校验上限已放宽到 900s/80 次），评测统一放宽。
AGENT_ENV_OVERRIDES = {
    "CODE_REVIEW_MAX_OUTPUT_BYTES": "65536",
    "CODE_REVIEW_TOTAL_TIMEOUT_SECONDS": "600",
    "CODE_REVIEW_MAX_TOOL_CALLS": "60",
}


def ensure_agent_available() -> None:
    """预检：框架入口存在且 Docker daemon 可用。"""
    agent_entry = TRPC_ROOT / AGENT_REL
    if not agent_entry.is_file():
        raise SystemExit(f"agent entry not found: {agent_entry}")
    docker = run_command(["docker", "info", "--format", "{{.ServerVersion}}"])
    if docker.returncode != 0:
        raise SystemExit(
            "Docker daemon unavailable; skills_code_review_agent real mode requires Docker."
        )
    log(f"myagent available: {agent_entry}")


def check_env(preview: bool) -> None:
    """LLM 配置由框架 .env 承载；这里只做提示性检查。"""
    if preview:
        return
    log("myagent: model config is read from the framework .env "
        "(TRPC_AGENT_API_KEY / TRPC_AGENT_BASE_URL / TRPC_AGENT_MODEL_NAME)")


def _agent_command_base() -> list[str]:
    return [
        "uv", "run", "--project", PROJECT_REL, "--with-editable", ".",
        "python", AGENT_REL,
    ]


def _agent_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.update(AGENT_ENV_OVERRIDES)
    return env


def _report_from_stdout(stdout: str, agent_out: Path) -> Path:
    """优先从 stdout 提取报告路径，失败则回退 output-dir 下最新 task 目录。"""
    for line in stdout.splitlines():
        if line.startswith("JSON report: "):
            path = Path(line.removeprefix("JSON report: ").strip())
            if path.is_file():
                return path
    reports = sorted(
        agent_out.glob("*/review_report.json"), key=lambda p: p.stat().st_mtime
    )
    if not reports:
        raise RuntimeError(f"review_report.json not found under {agent_out}")
    return reports[-1]


def run_agent_commit_range(
    repo_path: Path,
    base_commit: str,
    head_commit: str,
    agent_out: Path,
    db_path: Path,
    timeout_seconds: int,
) -> tuple[Path, float, subprocess.CompletedProcess]:
    """commit-range 模式评审，返回 (review_report.json, 用时秒, 子进程结果)。"""
    agent_out.mkdir(parents=True, exist_ok=True)
    command = _agent_command_base() + [
        "--repo-path", str(repo_path),
        "--commit-range", base_commit, head_commit,
        "--output-dir", str(agent_out),
        "--database", str(db_path),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=str(TRPC_ROOT),
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
        env=_agent_env(),
    )
    duration = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"run_agent.py exit={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:500]}"
        )
    return _report_from_stdout(result.stdout, agent_out), duration, result


def run_agent_diff_file(
    diff_path: Path,
    agent_out: Path,
    db_path: Path,
    timeout_seconds: int,
) -> tuple[Path, float, subprocess.CompletedProcess]:
    """diff-file 模式评审（悬空 commit 回退用），返回同上。"""
    agent_out.mkdir(parents=True, exist_ok=True)
    command = _agent_command_base() + [
        "--diff-file", str(diff_path),
        "--output-dir", str(agent_out),
        "--database", str(db_path),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=str(TRPC_ROOT),
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
        env=_agent_env(),
    )
    duration = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"run_agent.py exit={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:500]}"
        )
    return _report_from_stdout(result.stdout, agent_out), duration, result


def _load_pr_urls() -> Dict[tuple[str, str, str], str]:
    """从原始数据集建 (repo, base, head) -> githubPrUrl 映射。"""
    if not RAW_DATASET_PATH.is_file():
        return {}
    records = json.loads(RAW_DATASET_PATH.read_text(encoding="utf-8"))
    mapping: Dict[tuple[str, str, str], str] = {}
    for record in records:
        pr_url = record.get("githubPrUrl")
        if not isinstance(pr_url, str) or "/pull/" not in pr_url:
            continue
        base = "/".join(pr_url.split("/pull/")[0].rstrip("/").split("/")[-2:])
        key = (
            base,
            str(record.get("source_commit", "")),
            str(record.get("target_commit", "")),
        )
        mapping[key] = pr_url
    return mapping


def fetch_pr_diff(
    instance: ReviewInstance, work_dir: Path, clone_timeout: int = 600
) -> Path:
    """悬空 commit 回退：按 PR URL 拉 <pr>.diff 到 work_dir，返回 patch 路径。"""
    import urllib.request
    import urllib.error

    pr_url = _load_pr_urls().get(
        (instance.repo, instance.base_commit, instance.head_commit), ""
    )
    if not pr_url:
        raise RepoError(
            f"commit {instance.head_commit[:12]} unavailable and no githubPrUrl "
            f"matched in {RAW_DATASET_PATH.name}"
        )
    diff_url = f"{pr_url}.diff"
    diff_path = work_dir / f"{instance.safe_id}.diff"
    log(f"commit unavailable; fetching PR diff {diff_url} -> {diff_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(diff_url, timeout=clone_timeout) as response:
            diff_path.write_bytes(response.read())
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise RepoError(f"PR diff fetch failed ({diff_url}): {error}") from error
    if not diff_path.read_text(encoding="utf-8", errors="replace").strip():
        raise RepoError(f"empty PR diff: {diff_url}")
    return diff_path


def convert_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """ReviewAnalysis.findings -> 评测 codex 形状 review_output。

    口径（docs/aacr_bench_eval_plan.md 3.7 节）：
    - 只取 analysis.findings（不含 warnings，confidence<0.70 低置信项）
    - 单行号 line -> start_line = end_line；null 保留（judge 跳过行号阶段）
    - side 由评测 builder 统一补 right
    """
    review_output: list[dict[str, Any]] = []
    for finding in report.get("analysis", {}).get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        summary = (finding.get("title") or "").strip()
        description = "\n".join(
            part for part in [
                (finding.get("evidence") or "").strip(),
                (finding.get("recommendation") or "").strip(),
            ] if part
        ).strip()
        if not summary and not description:
            continue
        line = finding.get("line")
        review_output.append({
            "file": finding.get("file", ""),
            "summary": summary or description[:100],
            "description": description or summary,
            "start_line": line,
            "end_line": line,
        })
    return review_output


def save_result(
    results_dir: Path,
    instance: ReviewInstance,
    review_output: list[dict[str, Any]],
    duration_seconds: float,
    report_path: Path,
    mode: str,
    started_at: str,
    agent_status: str = "",
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.result_path(results_dir, instance.instance_id)
    payload = {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "head_commit": instance.head_commit,
        "reviewer": "myagent",
        "started_at": started_at,
        "duration_seconds": round(duration_seconds, 2),
        "mode": mode,
        "agent_status": agent_status,
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "review_output": review_output,
        "report_path": str(report_path),
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return out_path


def review_instance(
    instance: ReviewInstance,
    repo_dir: Path,
    results_dir: Path,
    timeout_minutes: int = 20,
    preview: bool = False,
) -> Dict[str, Any]:
    """对单条样本执行 myagent 评审；任何内部异常转为 RepoError，
    保证 pipeline 的单样本失败不中断整批。"""
    try:
        return _review_instance_impl(instance, repo_dir, results_dir, timeout_minutes, preview)
    except RepoError:
        raise
    except Exception as error:  # noqa: BLE001 - 单样本失败不中断批次
        raise RepoError(f"myagent failed for {instance.instance_id}: {error}") from error


def _review_instance_impl(
    instance: ReviewInstance,
    repo_dir: Path,
    results_dir: Path,
    timeout_minutes: int,
    preview: bool,
) -> Dict[str, Any]:
    log(f"=== [myagent] Processing {instance.instance_id} ===")

    # 悬空 commit 回退：仅当 commit 不可用时降级 PR diff 模式；
    # clone/checkout 等网络类失败重试一次后按单样本错误上报，不降级。
    repo_path: Path | None = None
    diff_path: Path | None = None
    try:
        repo_path = prepare_repo(
            repo_dir=repo_dir,
            clone_url=instance.resolved_clone_url,
            repo_full_name=instance.repo,
            base_commit=instance.base_commit,
            head_commit=instance.head_commit,
        )
    except RepoError as error:
        if str(error).startswith("commit"):
            log(f"commit unavailable ({error}); falling back to PR diff mode")
            diff_path = fetch_pr_diff(instance, results_dir / "pr_diffs")
        else:
            log(f"prepare_repo failed ({error}); retrying once")
            try:
                repo_path = prepare_repo(
                    repo_dir=repo_dir,
                    clone_url=instance.resolved_clone_url,
                    repo_full_name=instance.repo,
                    base_commit=instance.base_commit,
                    head_commit=instance.head_commit,
                )
            except RepoError as retry_error:
                # 重试后才暴露的悬空 commit（如首次 clone 因网络失败）同样回退
                if str(retry_error).startswith("commit"):
                    log(f"commit unavailable ({retry_error}); falling back to PR diff mode")
                    diff_path = fetch_pr_diff(instance, results_dir / "pr_diffs")
                else:
                    raise RepoError(f"myagent: {retry_error}") from retry_error

    if preview:
        stat = (
            diff_stat(repo_path, instance.base_commit, instance.head_commit)
            if repo_path is not None
            else "(PR diff fallback)"
        )
        log(f"Preview mode: skipping agent call. diff stat: {stat}")
        return {
            "instance_id": instance.instance_id,
            "status": "preview",
            "diff_stat": stat,
        }

    run_id = f"{instance.safe_id}"
    agent_out = results_dir / "agent_out" / run_id
    db_path = results_dir / "myagent_reviews.sqlite3"
    process_timeout = timeout_minutes * 60 + 300

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    if diff_path is not None:
        report_path, duration, _proc = run_agent_diff_file(
            diff_path, agent_out, db_path, process_timeout
        )
        mode = "diff_file"
    else:
        assert repo_path is not None
        report_path, duration, _proc = run_agent_commit_range(
            repo_path,
            instance.base_commit,
            instance.head_commit,
            agent_out,
            db_path,
            process_timeout,
        )
        mode = "commit_range"
    duration = time.monotonic() - start_time

    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_output = convert_findings(report)
    out_path = save_result(
        results_dir=results_dir,
        instance=instance,
        review_output=review_output,
        duration_seconds=duration,
        report_path=report_path,
        mode=mode,
        started_at=started_at,
        agent_status=str(report.get("status", "")),
    )

    if repo_path is not None:
        clean_worktree(repo_path)

    # completed_with_warnings 表示评审执行完整、带低置信警告，计为成功
    agent_status = str(report.get("status", ""))
    status = "ok" if agent_status in ("completed", "completed_with_warnings") else "failed"
    log(f"--- [myagent] {instance.instance_id} {status} "
        f"(findings={len(review_output)}, mode={mode}) -> {out_path}")
    return {
        "instance_id": instance.instance_id,
        "status": status,
        "findings": len(review_output),
        "mode": mode,
        "result_path": str(out_path),
    }
