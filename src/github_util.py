# src/github_util.py

import os
import json
from pathlib import Path

# from dotenv import load_dotenv
from github import Github, GithubException

# ─── Load .env ────────────────────────────────────────────────────────────────
# load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")
GITHUB_PAT = os.getenv("PAT")
REPO_NAME = "dshan12/algo-trader"
if not GITHUB_PAT:
    raise RuntimeError("GITHUB PAT must be set in your .env")


def commit_trade(trade: dict):
    """
    Append a trade record to:
      1) top‑level trades.json
      2) a daily Markdown file trades/YYYY‑MM‑DD.md
    both locally and on GitHub.
    """
    # ── Prepare local paths ─────────────────────────────────────────────────────
    repo_root = Path(__file__).parents[1]
    top_path = repo_root / "trades.json"
    daily_dir = repo_root / "trades"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # parse the trade date (YYYY‑MM‑DD) from timestamp
    date_str = trade["timestamp"][:10]
    daily_md = daily_dir / f"{date_str}.md"

    # ── 1) Update top‑level trades.json ─────────────────────────────────────────
    if top_path.exists():
        all_trades = json.loads(top_path.read_text())
    else:
        all_trades = []
    all_trades.append(trade)
    top_path.write_text(json.dumps(all_trades, indent=2))

    # ── 2) Update daily Markdown ────────────────────────────────────────────────
    heading = f"# Trades for {date_str}\n\n"
    entry = (
        f"## {trade['symbol']} — {trade['side'].upper()} {trade['qty']} @ ${trade['price']:.2f}\n"
        f"- **Time:** {trade['timestamp']}\n"
    )

    reason = trade["reason"]
    if isinstance(reason, list):
        entry += "- **Reason:**\n"
        for line in reason:
            entry += f"  - {line}\n"
        entry += "\n"
    else:
        entry += f"- **Reason:** {reason}\n\n"

    if daily_md.exists():
        content = daily_md.read_text()
        if not content.startswith(heading):
            content = heading + content
        content += entry
    else:
        content = heading + entry

    daily_md.write_text(content)

    # ── Push both to GitHub ─────────────────────────────────────────────────────
    gh = Github(GITHUB_PAT)
    repo = gh.get_repo(REPO_NAME)
    branch = repo.default_branch  # e.g. "main" or "master"

    def _sync_file(local_path: Path, remote_path: str, commit_msg: str):
        body = local_path.read_text()
        try:
            existing = repo.get_contents(remote_path, ref=branch)
            repo.update_file(
                path=existing.path,
                message=commit_msg,
                content=body,
                sha=existing.sha,
                branch=branch,
            )
        except GithubException as e:
            if e.status == 404:
                repo.create_file(
                    path=remote_path,
                    message=commit_msg,
                    content=body,
                    branch=branch,
                )
            else:
                raise

    # sync top‑level JSON
    _sync_file(
        top_path,
        "trades.json",
        f"Log trade {trade['symbol']} {trade['side']} @ {trade['price']}",
    )
    # sync daily Markdown
    _sync_file(daily_md, f"trades/{date_str}.md", f"Log trades for {date_str}")
