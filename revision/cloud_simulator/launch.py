"""Start the loopback estimator and open it in Chrome when installed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[2] / "revision_outputs" / "cloud_simulator_v1"
URL = "http://127.0.0.1:8765/"


def ready():
    try:
        with urllib.request.urlopen(URL + "api/health", timeout=1) as stream:
            value = json.load(stream)
        return value.get("service") == "cloud-estimator" and value.get("status") == "ready"
    except (OSError, ValueError):
        return False


def main():
    if not ready():
        if not (ARTIFACTS / "manifest.json").exists():
            raise SystemExit("Train the local estimator before opening the interface.")
        with (ARTIFACTS / "server.stdout.log").open("ab") as out, (ARTIFACTS / "server.stderr.log").open("ab") as err:
            process = subprocess.Popen([sys.executable, "-u", str(HERE / "app.py")],
                cwd=HERE, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (ARTIFACTS / "server.pid").write_text(str(process.pid), encoding="ascii")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if ready():
                break
            if process.poll() is not None:
                raise SystemExit("Service failed. See " + str(ARTIFACTS / "server.stderr.log"))
            time.sleep(.25)
        else:
            raise SystemExit("Service is still starting. See the local server log.")
    candidates = [Path(os.environ.get(key, "")) / "Google/Chrome/Application/chrome.exe"
                  for key in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    browser = next((path for path in candidates if path.is_file()), None)
    if browser:
        subprocess.Popen([str(browser), URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(URL)
    print("Cloud estimator: " + URL)


if __name__ == "__main__":
    main()
