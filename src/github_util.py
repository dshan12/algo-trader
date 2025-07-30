from pathlib import Path
from dotenv import load_dotenv
import os
import json
from github import Github

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

GITHUB_PAT = os.getenv("GITHUB_PAT")
REPO_NAME = "dshan12/algo-trader"


def commit_trade(trade: dict):
    """
    Appends `trade` to local trades.json and commits the updated file to GitHub.
    """
    file_path = Path(__file__).parents[1] / "trades.json"

    # Load existing trades
    trades = []
    if file_path.exists():
        trades = json.loads(file_path.read_text())

    # Append new trade
    trades.append(trade)
    new_content = json.dumps(trades, indent=2)

    # Commit to GitHub
    gh = Github(GITHUB_PAT)
    repo = gh.get_repo(REPO_NAME)
    contents = repo.get_contents("trades.json")
    repo.update_file(
        path=contents.path,
        message=f"Log trade {trade['symbol']} {trade['side']} @ {trade['price']}",
        content=new_content,
        sha=contents.sha,
    )

    # Also write locally
    file_path.write_text(new_content)
