"""
Sniper Trading Framework - FastAPI Server Runner
Starts the FastAPI server with uvicorn
"""

import sys
import subprocess
import os


def check_dependencies():
    """
    Check if required dependencies are installed, install if missing
    """
    required = [
        "fastapi",
        "uvicorn",
        "websockets",
        "yfinance",
        "pandas",
        "numpy",
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing
        ])
        print("✓ Dependencies installed\n")


def main():
    """
    Start the FastAPI server
    """
    print("═" * 60)
    print("  SNIPER TRADING FRAMEWORK - API SERVER")
    print("═" * 60)
    print()

    # Check dependencies
    check_dependencies()

    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not found after installation attempt")
        sys.exit(1)

    # Print server info
    print("Starting Sniper API Server...")
    print()
    print("  API Server:   http://localhost:8000")
    print("  API Docs:     http://localhost:8000/docs")
    print("  WebSocket:    ws://localhost:8000/ws/{symbol}/{timeframe}")
    print()
    print("  Press Ctrl+C to stop the server")
    print("═" * 60)
    print()

    # Start uvicorn server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        sys.exit(1)
