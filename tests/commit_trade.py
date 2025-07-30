#!/usr/bin/env python3
# test_commit_trade.py

from pathlib import Path
from datetime import datetime
from src.github_util import commit_trade

if __name__ == "__main__":
    # Build a dummy trade
    fake_trade = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "TEST",
        "side": "buy",
        "qty": 1,
        "price": 0.01,
        "reason": "🚀 Testing commit_trade function",
    }

    # Call commit_trade
    commit_trade(fake_trade)
    print("✅ commit_trade invoked. Check trades.json and GitHub.")
