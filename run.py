#!/usr/bin/env python3
"""
One-command runner for the Customer Churn Prediction System.

    python run.py                 # train only if outputs/ is missing, then serve
    python run.py --retrain       # force retraining even if outputs/ exists
    python run.py --no-browser    # don't auto-open the dashboard
    python run.py --port 8080     # use a different API port
    python run.py --ui-port 8090  # use a different dashboard port
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
REQUIRED_ARTIFACTS = ["best_model.pkl", "feature_columns.pkl", "label_encoders.pkl", "scaler.pkl"]
REQUIRED_PACKAGES = ["pandas", "numpy", "sklearn", "xgboost", "shap", "fastapi", "uvicorn", "joblib"]


def check_dependencies():
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _can_import(pkg)]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install them first:  pip install -r requirements.txt")
        sys.exit(1)


def _can_import(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def artifacts_exist():
    return all((OUTPUTS_DIR / f).exists() for f in REQUIRED_ARTIFACTS)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the Customer Churn Prediction System")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if outputs/ already exists")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the dashboard")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8080, help="Dashboard port (default: 8080)")
    args = parser.parse_args()

    check_dependencies()

    if args.retrain or not artifacts_exist():
        print("Training pipeline (outputs/ missing artifacts, or --retrain passed)...")
        run([sys.executable, "eda.py"])
        run([sys.executable, "train_model.py"])
    else:
        print("Found a trained model in outputs/ — skipping training (use --retrain to force).")

    print(f"\nStarting API on http://localhost:{args.port}  (docs at /docs)")
    api_server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(args.port)],
        cwd=ROOT,
    )

    # The dashboard uses ES6 modules (web_ui/js/*.js loaded via <script type="module">),
    # which browsers refuse to import over file:// — it has to be served over HTTP,
    # even for local-only use. Python's built-in http.server handles that with no
    # extra dependency.
    print(f"Starting dashboard on http://localhost:{args.ui_port}")
    ui_server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(args.ui_port), "--directory", str(ROOT / "web_ui")],
        cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    dashboard_url = f"http://localhost:{args.ui_port}"
    if not args.no_browser:
        time.sleep(2)  # give both servers a moment to bind their ports
        import webbrowser
        webbrowser.open(dashboard_url)
    print(f"\nDashboard: {dashboard_url}   API: http://localhost:{args.port}/docs   — Ctrl+C to stop")

    try:
        api_server.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        api_server.terminate()
        ui_server.terminate()
        api_server.wait()
        ui_server.wait()


if __name__ == "__main__":
    main()
