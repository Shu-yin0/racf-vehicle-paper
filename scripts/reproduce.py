"""Run the public, deterministic reproduction gate."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    smoke = ROOT / "experiments" / "run_smoke.py"
    subprocess.run([sys.executable, str(smoke), "--output-dir", str(ROOT / "results" / "smoke")], check=True)
    info = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        "scope=mock test; synthetic transitions only",
    ]
    (ROOT / "results" / "smoke" / "session_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print("RACF public smoke reproduction completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
