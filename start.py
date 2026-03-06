"""
start.py - One command to launch the Roblox AI Dashboard
Usage:
  python start.py              # launch GUI on http://localhost:5000
  python start.py --port 8080  # custom port
  python start.py --install    # install dependencies first, then launch
"""

import os, sys, argparse, subprocess, webbrowser, time

BASE = os.path.dirname(os.path.abspath(__file__))
REQ  = os.path.join(BASE, "requirements.txt")

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQ])
    print("Done.\n")

def check_deps():
    try:
        import flask, torch, cv2, mss, pynput
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.install or not check_deps():
        install_deps()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        import threading
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    from gui import app
    print(f"\nRoblox AI Dashboard\nOpen: {url}\nStop: Ctrl+C\n")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
