#!/usr/bin/env python3
"""
SNIPER FRAMEWORK — CHART LAUNCHER
  1. Checks / installs missing dependencies
  2. Starts the Flask-SocketIO server (eventlet)
  3. Opens the chart in your default browser

Usage:
  python run_chart.py
"""

import sys
import subprocess
import os
import webbrowser
import threading
from importlib import import_module

REQUIRED = {
    # module_name  →  pip package
    "flask":          "flask",
    "flask_socketio": "flask-socketio",
    "flask_cors":     "flask-cors",
    "eventlet":       "eventlet",
    "yfinance":       "yfinance",
    "pandas":         "pandas",
    "numpy":          "numpy",
}


def check_and_install():
    """Return True if all dependencies are available (installing missing ones)."""
    missing = []
    print("\n  Checking dependencies …")
    for mod, pkg in REQUIRED.items():
        try:
            import_module(mod)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg}  (missing)")
            missing.append(pkg)

    if not missing:
        return True

    print(f"\n  Installing: {', '.join(missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
        )
        print("    ✓ installed")
        return True
    except subprocess.CalledProcessError:
        print("    ✗ pip install failed — install manually:")
        print(f"      pip install {' '.join(missing)}")
        return False


def check_files():
    here = os.path.dirname(os.path.abspath(__file__))
    needed = ["server.py", "sniper_chart.html", "config.py",
              "strategy/core_engine.py", "data/base_provider.py",
              "data/yfinance_provider.py"]
    ok = True
    for f in needed:
        p = os.path.join(here, f)
        if not os.path.exists(p):
            print(f"    ✗ missing  {f}")
            ok = False
    return ok


def main():
    print()
    print("=" * 60)
    print("  SNIPER TRADING FRAMEWORK — CHART LAUNCHER")
    print("=" * 60)

    if not check_and_install():
        sys.exit(1)

    if not check_files():
        print("\n  Some files are missing. Run from the project root.")
        sys.exit(1)

    # Open browser after a short delay so the server has time to bind
    def open_browser():
        import time; time.sleep(1.8)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser, daemon=True).start()

    # ── launch server ────────────────────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    server = os.path.join(here, "server.py")

    print("\n  Starting server …\n")
    try:
        subprocess.run([sys.executable, server], cwd=here)
    except KeyboardInterrupt:
        print("\n\n  Server stopped.")


if __name__ == "__main__":
    main()
