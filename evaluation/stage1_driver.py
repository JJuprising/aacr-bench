"""阶段 1 驱动脚本：AACR-Bench 样本 → 本框架 --diff-file fake 模式 → 评测结果文件。

链路（不动 skills_code_review_agent 框架本身）：
  1. 加载标准 JSONL（data/aacr_bench.jsonl）
  2. blobless 克隆仓库缓存（repo_stage1/，比全量 clone 快很多；仅本阶段用，
     正式接入时 reviewers/myagent.py 仍走框架 prepare_repo 全量缓存 repo/）
  3. git diff base..head 生成 patch（无需 checkout）
  4. subprocess 调 run_agent.py --diff-file <patch> --fake-model
     （fake 模式 = 确定性六类规则，无 Docker / 无模型依赖）
  5. 读取 review_report.json，把 findings 转换为评测标准结果文件
     <safe_id>.json（codex 同构形状，evaluate.py 的 codex builder 可直接解析）
  6. 之后运行：python -m pipeline run --stage eval --reviewer codex \\
       --dataset data/aacr_bench.jsonl --results-dir results/stage1_fake

用法（evaluation/ 目录内，已激活 venv）：
  python stage1_driver.py                       # 全部样本
  python stage1_driver.py --only lvgl__lvgl@4a57db3   # 单样本冒烟
  python stage1_driver.py --limit 3             # 前 3 条
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from schema import ReviewInstance, load_instances  # noqa: E402

# 本框架（skills_code_review_agent）调用配置
TRPC_ROOT = Path("/home/chan/trpc-agent-python")
AGENT_REL = "examples/skills_code_review_agent/run_agent.py"
PROJECT_REL = "examples/skills_code_review_agent"

DEFAULT_RESULTS = EVAL_DIR / "results" / "stage1_fake"
DEFAULT_REPO_DIR = EVAL_DIR / "repo_stage1"
DEFAULT_PATCH_DIR = EVAL_DIR / "work" / "stage1_patches"


def log(message: str) -> None:
    print(f"[stage1] {message}", flush=True)


def run_cmd(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def safe_id(instance_id: str) -> str:
    return instance_id.replace("/", "__")


# ---------------------------------------------------------------- git 操作


def ensure_repo(
    repo_dir: Path, instance: ReviewInstance, clone_timeout: int
) -> Path:
    """blobless 克隆缓存；已存在则补 fetch 缺失 commit。返回本地仓库路径。"""
    repo_path = repo_dir / safe_id(instance.repo)
    if not (repo_path / ".git").exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"blobless clone {instance.resolved_clone_url} -> {repo_path}")
        result = run_cmd(
            [
                "git", "clone",
                "--filter=blob:none", "--no-checkout",
                instance.resolved_clone_url, str(repo_path),
            ],
            timeout=clone_timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"clone failed: {result.stderr.strip()[:300]}")
    for commit in {instance.base_commit, instance.head_commit}:
        check = run_cmd(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_path
        )
        if check.returncode != 0:
            log(f"fetch missing commit {commit[:12]}")
            fetch = run_cmd(
                ["git", "fetch", "origin", commit], cwd=repo_path, timeout=1800
            )
            if fetch.returncode != 0:
                raise RuntimeError(
                    f"commit {commit[:12]} unavailable: {fetch.stderr.strip()[:300]}"
                )
    return repo_path


def generate_patch(
    repo_path: Path, instance: ReviewInstance, patch_path: Path
) -> bool:
    """生成 base..head unified diff；空 diff 返回 False。"""
    result = run_cmd(
        ["git", "diff", "--find-renames", "--no-ext-diff",
         f"{instance.base_commit}..{instance.head_commit}"],
        cwd=repo_path,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()[:300]}")
    if not result.stdout.strip():
        return False
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(result.stdout, encoding="utf-8")
    return True


# ------------------------------------------------------- 调用本框架 fake 模式


def run_agent_fake(
    patch_path: Path, agent_out: Path, db_path: Path, timeout: int
) -> tuple[Path, float]:
    """调用 run_agent.py --fake-model，返回 (review_report.json, 用时秒)。"""
    import os

    agent_out.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--project", PROJECT_REL, "--with-editable", ".",
        "python", AGENT_REL,
        "--diff-file", str(patch_path),
        "--fake-model",
        "--output-dir", str(agent_out),
        "--database", str(db_path),
    ]
    # .env 若把 CODE_REVIEW_MAX_OUTPUT_BYTES 收紧到 64KB 以下，
    # fake 路径的 SandboxCommand 默认预算(64KB)会被 Filter 判超预算而全部拦截。
    # 这里仅为 fake 评审进程恢复框架默认值，不改动用户 .env。
    env = {**os.environ, "CODE_REVIEW_MAX_OUTPUT_BYTES": "65536"}
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=str(TRPC_ROOT),
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    duration = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"run_agent.py exit={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:500]}"
        )
    # 优先从 stdout 提取报告路径，失败则回退到 output-dir 下最新 task 目录
    match = re.search(r"JSON report: (.+)", result.stdout)
    if match:
        report_path = Path(match.group(1).strip())
        if report_path.is_file():
            return report_path, duration
    task_dirs = sorted(agent_out.glob("*/review_report.json"), key=lambda p: p.stat().st_mtime)
    if not task_dirs:
        raise RuntimeError(f"review_report.json not found under {agent_out}")
    return task_dirs[-1], duration


def convert_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """ReviewAnalysis.findings -> 评测 codex 形状 review_output。

    决策（见 docs/aacr_bench_eval_plan.md 3.7 节）：
    - 只取 analysis.findings（不含 warnings，confidence<0.70 口径）
    - 单行号 line -> start_line = end_line；line 为 null 保留
      （judge 对 null 行号跳过行号阶段，仅语义匹配）
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
    results_dir: Path, instance: ReviewInstance,
    review_output: list[dict[str, Any]], duration: float,
    report_path: Path,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{safe_id(instance.instance_id)}.json"
    payload = {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "head_commit": instance.head_commit,
        "reviewer": "stage1-fake",
        "duration_seconds": round(duration, 2),
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "review_output": review_output,
        "report_path": str(report_path),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 1 评审驱动（fake 模式）")
    parser.add_argument("--dataset", type=Path,
                        default=EVAL_DIR / "data" / "aacr_bench.jsonl")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--only", help="仅跑指定 instance_id（冒烟用）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clone-timeout", type=int, default=3600,
                        help="单仓库克隆超时（秒）")
    parser.add_argument("--agent-timeout", type=int, default=900,
                        help="单样本评审超时（秒）")
    args = parser.parse_args()

    instances = load_instances(args.dataset, args.limit)
    if args.only:
        instances = [i for i in instances if i.instance_id == args.only]
        if not instances:
            log(f"instance not found: {args.only}")
            return 1
    log(f"loaded {len(instances)} instance(s) from {args.dataset}")

    agent_out_root = args.results_dir / "agent_out"
    db_path = args.results_dir / "stage1_reviews.sqlite3"
    summary: list[dict[str, Any]] = []

    for instance in instances:
        sid = safe_id(instance.instance_id)
        log(f"=== {instance.instance_id} ===")
        status = "ok"
        error = None
        try:
            repo_path = ensure_repo(args.repo_dir, instance, args.clone_timeout)
            patch_path = args.patch_dir / f"{sid}.diff"
            if not generate_patch(repo_path, instance, patch_path):
                log(f"empty diff, writing empty result")
                report_path, duration = None, 0.0
                out_path = save_result(args.results_dir, instance, [], 0.0, "")
            else:
                report_path, duration = run_agent_fake(
                    patch_path, agent_out_root / sid, db_path, args.agent_timeout
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                findings = convert_findings(report)
                log(f"findings: {len(findings)}, duration: {duration:.1f}s")
                out_path = save_result(
                    args.results_dir, instance, findings, duration, report_path
                )
        except Exception as exc:  # noqa: BLE001 - 单样本失败不阻断批次
            status, error = "error", str(exc)
            log(f"ERROR {instance.instance_id}: {exc}")
            out_path = None
        summary.append({
            "instance_id": instance.instance_id,
            "status": status,
            "error": error,
            "result_path": str(out_path) if out_path else None,
        })

    summary_path = args.results_dir / "summary_stage1.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = sum(1 for s in summary if s["status"] == "ok")
    log(f"done: {ok}/{len(summary)} ok, summary -> {summary_path}")
    log(
        "next: python -m pipeline run --stage eval --reviewer codex "
        f"--dataset {args.dataset} --results-dir {args.results_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
