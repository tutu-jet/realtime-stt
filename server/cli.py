"""
sfm — realtime-stt command-line interface

Usage:
    sfm serve              Start the STT server (tiny model, CPU)
    sfm serve --model medium --device cuda
    sfm test               Run the test suite
    sfm health             Check running server health
"""
import argparse
import os
import sys


def cmd_serve(args):
    os.environ.setdefault("MODEL_SIZE", args.model)
    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("COMPUTE_TYPE", args.compute_type)
    if args.model_cache:
        os.environ.setdefault("MODEL_CACHE_DIR", args.model_cache)

    if args.daemonize:
        import subprocess
        log_file = args.log or f"sfm-{args.port}.log"
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                [sys.argv[0], "serve",
                 "--model", args.model,
                 "--device", args.device,
                 "--host", args.host,
                 "--port", str(args.port),
                 "--compute-type", args.compute_type,
                 "--model-cache", args.model_cache],
                stdout=log, stderr=log,
                start_new_session=True,
            )
        print(f"Started in background (PID={proc.pid}), log: {log_file}")
        print(f"Listening on: http://{args.host}:{args.port}")
        print(f"Check status: sfm health --port {args.port}")
        return

    import uvicorn
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        app_dir=os.path.dirname(__file__),
        reload=args.reload,
    )


def cmd_test(_args):
    import subprocess
    project_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=project_root,
    )
    sys.exit(result.returncode)


def cmd_health(args):
    import urllib.request
    url = f"http://{args.host}:{args.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(resp.read().decode())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="sfm", description="realtime-stt CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # sfm serve
    p_serve = sub.add_parser("serve", help="Start the STT server")
    p_serve.add_argument("--model", default=os.environ.get("MODEL_SIZE", "tiny"),
                         help="Whisper model size (default: tiny)")
    p_serve.add_argument("--device", default=os.environ.get("DEVICE", "cpu"),
                         help="Device: cpu / cuda / auto (default: cpu)")
    p_serve.add_argument("--compute-type", default="int8", dest="compute_type")
    p_serve.add_argument("--model-cache", default=os.path.expanduser("~/.cache/huggingface/hub"),
                         dest="model_cache")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=9090)
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    p_serve.add_argument("-d", "--daemonize", action="store_true", help="Run in background")
    p_serve.add_argument("--log", default=None, help="Log file path (default: sfm-<port>.log)")

    # sfm test
    sub.add_parser("test", help="Run the test suite")

    # sfm health
    p_health = sub.add_parser("health", help="Check server health")
    p_health.add_argument("--host", default="127.0.0.1")
    p_health.add_argument("--port", type=int, default=9090)

    args = parser.parse_args()
    {"serve": cmd_serve, "test": cmd_test, "health": cmd_health}[args.command](args)


if __name__ == "__main__":
    main()
