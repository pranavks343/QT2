import subprocess
import sys

import uvicorn


def check_deps() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn as _uvicorn  # noqa: F401
        import yfinance  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "fastapi",
                "uvicorn[standard]",
                "yfinance",
            ]
        )


if __name__ == "__main__":
    check_deps()
    print("Sniper API running at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
