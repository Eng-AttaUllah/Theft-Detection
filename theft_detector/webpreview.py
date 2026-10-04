"""Browser preview: a tiny MJPEG server for the annotated live feed.

Why this exists: ``cv2.imshow`` needs a desktop OpenCV build
(``opencv-python``, not ``opencv-python-headless``) *and* a local display.
This preview needs neither - run ``run.py --web`` and open the printed URL in
any browser, on this machine or over SSH port-forwarding.

Only the Python standard library is used; the frames are JPEG-encoded with
OpenCV, which every build can do.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BOUNDARY = "--boundaryframe"
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>shop theft detector</title>
<style>
  body { background:#14161a; color:#dfe3e8; font:14px/1.5 system-ui,sans-serif;
         margin:0; padding:24px; text-align:center }
  img { max-width:100%; height:auto; border:1px solid #333a44; border-radius:6px;
        background:#000 }
  h1 { font-size:16px; font-weight:600; letter-spacing:.04em; margin:0 0 14px }
  p  { color:#8b93a1; font-size:13px }
</style>
</head>
<body>
<h1>shop theft detector &mdash; live</h1>
<img src="/stream.mjpg" alt="live feed"
     onerror="this.src='/snapshot.jpg?' + Date.now()">
<p>annotated feed &middot; alerts are printed on the terminal</p>
</body>
</html>
"""


class FrameBoard:
    """Latest JPEG frame + a condition variable, shared with the handler."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    def wait_for_new(self, seq: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._cond:
            if self._seq == seq:
                self._cond.wait(timeout)
            return self._seq, self._jpeg

    def snapshot(self) -> bytes | None:
        with self._cond:
            return self._jpeg


# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "theftdetector/1.0"
    board: FrameBoard                      # set on the server, see PreviewServer

    def log_message(self, fmt: str, *args) -> None:      # keep the console clean
        pass

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self._send(200, "text/html; charset=utf-8", body, cache=False)
            return
        if self.path == "/snapshot.jpg":
            jpeg = self.board.snapshot()
            if jpeg is None:
                self.send_error(503, "no frame yet")
                return
            self._send(200, "image/jpeg", jpeg, cache=False)
            return
        if self.path in ("/stream.mjpg", "/stream"):
            self._stream()
            return
        self.send_error(404)

    # ------------------------------------------------------------------ #
    def _send(self, code: int, ctype: str, body: bytes, cache: bool = True) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY[2:]}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        seq = -1
        try:
            while True:
                seq, jpeg = self.board.wait_for_new(seq)
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(
                    f"{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return                      # browser closed the tab


# --------------------------------------------------------------------------- #
class PreviewServer:
    """Serves the annotated frames on http://<host>:<port>/."""

    def __init__(self, port: int = 8080, host: str = "127.0.0.1") -> None:
        self.board = FrameBoard()
        handler = type("BoundHandler", (_Handler,), {"board": self.board})
        try:
            self._httpd = ThreadingHTTPServer((host, int(port)), handler)
        except OSError as exc:
            raise RuntimeError(
                f"cannot open the preview on {host}:{port} ({exc}) - "
                "is that port already in use? try --web 8090"
            ) from exc
        self._httpd.daemon_threads = True
        host, port = self._httpd.server_address[:2]
        self.url = f"http://{host}:{port}/"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="preview", daemon=True
        )

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread.start()

    def publish(self, frame: np.ndarray) -> None:
        self.board.publish(frame)

    def stop(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except OSError:
            pass
