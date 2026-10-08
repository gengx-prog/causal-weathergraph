"""Local-only interactive cloud estimator; no remote services or source writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

# Keep small interactive requests from spawning a machine-wide OpenMP pool.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = ROOT.parent / "revision_outputs" / "cloud_simulator_v1"


def numeric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值。")
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值。")
    return float(value)


def normalize_payload(payload):
    """Convert meteorological *from* direction; never confuse RH and q."""
    if not isinstance(payload, dict):
        raise ValueError("请求必须是 JSON 对象。")
    result = dict(payload)
    if result.get("wind_input", "uv") == "speed_direction":
        speed = numeric(result.get("speed_mps"), "风速")
        direction = numeric(result.get("direction_deg"), "风向")
        if speed < 0 or not 0 <= direction <= 360:
            raise ValueError("风速须非负；气象风向须在 0–360°。")
        theta = math.radians(direction)
        result["u_mps"] = -speed * math.sin(theta)
        result["v_mps"] = -speed * math.cos(theta)
    elif result.get("wind_input", "uv") != "uv":
        raise ValueError("未知风输入方式。")
    for key in ("u_mps", "v_mps", "humidity_gkg"):
        result[key] = numeric(result.get(key), key)
    if result["humidity_gkg"] < 0:
        raise ValueError("比湿不可为负；单位为 g/kg，不是相对湿度百分比。")
    for key in ("horizon_hours", "region_id", "month"):
        if key in result and result[key] is not None:
            value = numeric(result[key], key)
            if value != int(value):
                raise ValueError(f"{key} 必须为整数。")
            result[key] = int(value)
    return result


def make_handler(estimator):
    prediction_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "CloudEstimator/1.0"

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self.send_bytes(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/metadata":
                self.send_json(200, estimator.metadata())
            elif path == "/api/health":
                self.send_json(200, {"service": "cloud-estimator", "status": "ready"})
            elif path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "页面不存在。"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/predict", "/api/sweep"):
                self.send_json(404, {"error": "接口不存在。"})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_json(403, {"error": "请从本地模拟器页面提交。"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("请求大小无效。")
                raw = json.loads(self.rfile.read(size))
                payload = normalize_payload(raw)
                with prediction_lock:
                    if path == "/api/predict":
                        value = estimator.predict(payload)
                        value["input"] = payload
                        value["generated_utc"] = datetime.now(timezone.utc).isoformat()
                    else:
                        lo = numeric(raw.get("sweep_min_gkg", 0), "扫描下限")
                        hi = numeric(raw.get("sweep_max_gkg", 20), "扫描上限")
                        if not 0 <= lo < hi <= 100:
                            raise ValueError("扫描范围须满足 0 ≤ 下限 < 上限 ≤ 100 g/kg。")
                        points = []
                        for i in range(25):
                            humidity = lo + (hi - lo) * i / 24
                            prediction = estimator.predict({**payload, "humidity_gkg": humidity})
                            points.append({"humidity_gkg": humidity, **prediction})
                        value = {"points": points, "input": payload,
                                 "interpretation": "固定其他输入的条件统计估计曲线，不是改变湿度的因果效应。"}
                self.send_json(200, value)
            except (ValueError, TypeError, KeyError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.send_json(500, {"error": "计算失败，请查看本地运行日志。"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    from revision.cloud_simulator.engine import CloudEstimator
    estimator = CloudEstimator(args.artifacts)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(estimator))
    print(f"Cloud estimator ready: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
