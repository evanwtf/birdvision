"""WebSocket frame source for the Pi sidecar streaming mode.

Accepts JPEG frames from a remote client (phone browser or test script)
over WebSocket and yields them as (frame_number, bgr) tuples — same
iterator interface as V4L2FrameSource.

Also serves static files (the browser camera client) over HTTP from
the same port.

Protocol
--------
Client → Server:
    1. (optional) Text JSON: {"frame_id": N, "lat": ..., "lon": ...}
    2. Binary: JPEG blob

Server → Client:
    JSON: {"frame_id": N, "detections": [...], "fps": ...}

Bounding box coordinates in results are normalized [0, 1].
"""

import asyncio
import json
import logging
import queue
import ssl
import subprocess
import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.routing import Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def _ensure_self_signed_cert(cert_dir: str) -> tuple:
    """Generate a self-signed cert+key if they don't already exist.

    Returns (certfile_path, keyfile_path).
    """
    cert_path = Path(cert_dir) / "cert.pem"
    key_path = Path(cert_dir) / "key.pem"

    if cert_path.exists() and key_path.exists():
        logger.info("Reusing existing TLS cert at %s", cert_path)
        return str(cert_path), str(key_path)

    Path(cert_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Generating self-signed TLS certificate in %s", cert_dir)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "365", "-nodes",
            "-subj", "/CN=birdvision-sidecar",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert_path), str(key_path)


class WebSocketFrameSource:
    """Receives JPEG frames over WebSocket, yields (frame_no, bgr) pairs.

    Args:
        host:                 Bind address (default 0.0.0.0)
        port:                 Listen port (default 8765)
        static_dir:           Directory to serve as static files at /
        max_buffered_frames:  Max frames to buffer before dropping
        ssl_certfile:         Path to TLS certificate (PEM)
        ssl_keyfile:          Path to TLS private key (PEM)
        ssl_cert_dir:         Directory for auto-generated self-signed cert
                              (used when certfile/keyfile not provided)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        static_dir: Optional[str] = None,
        max_buffered_frames: int = 2,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        ssl_cert_dir: str = "/data/certs",
    ):
        self._host = host
        self._port = port
        self._static_dir = static_dir
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._ssl_cert_dir = ssl_cert_dir
        self._stop = False
        # Queue stores (bgr, metadata_dict) tuples
        self._frame_queue: queue.Queue = queue.Queue(maxsize=max_buffered_frames)
        self._result_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._pending_metadata: dict = {}
        self._metadata: dict = {}
        self._client_connected = threading.Event()

    @property
    def client_metadata(self) -> dict:
        """Latest metadata from the connected client (lat, lon, etc.)."""
        return self._metadata

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_number, bgr_frame) from WebSocket JPEG frames."""
        self._start_server()
        frame_no = 0

        try:
            while not self._stop:
                try:
                    bgr, meta = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Drain stale frames — keep only the latest
                while not self._frame_queue.empty():
                    try:
                        bgr, meta = self._frame_queue.get_nowait()
                        frame_no += 1
                    except queue.Empty:
                        break

                self._metadata = meta
                yield frame_no, bgr
                frame_no += 1
        finally:
            self._shutdown_server()
            logger.info("WebSocketFrameSource stopped after %d frames", frame_no)

    def send_result(self, result: dict) -> None:
        """Send a JSON result back to the connected client (thread-safe)."""
        if not self._client_connected.is_set():
            return
        if self._loop and self._result_queue and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._result_queue.put_nowait, result)

    def stop(self) -> None:
        """Signal the frames() iterator to exit cleanly."""
        self._stop = True

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        app = self._build_app()
        self._server_thread = threading.Thread(
            target=self._run_server, args=(app,), daemon=True,
        )
        self._server_thread.start()
        self._loop_ready.wait(timeout=10.0)

    def _run_server(self, app: Starlette) -> None:
        import uvicorn

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._result_queue = asyncio.Queue()
        self._loop_ready.set()

        certfile = self._ssl_certfile
        keyfile = self._ssl_keyfile
        if not certfile or not keyfile:
            try:
                certfile, keyfile = _ensure_self_signed_cert(self._ssl_cert_dir)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.warning(
                    "Could not generate TLS cert (%s) — serving plain HTTP. "
                    "Camera will not work on phones over LAN.",
                    exc,
                )
                certfile = keyfile = None

        config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning",
            ssl_certfile=certfile, ssl_keyfile=keyfile,
        )
        server = uvicorn.Server(config)
        proto = "https" if certfile else "http"
        logger.info("WebSocket server listening on %s://%s:%d", proto, self._host, self._port)
        self._loop.run_until_complete(server.serve())

    def _shutdown_server(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    def _build_app(self) -> Starlette:
        routes = [WebSocketRoute("/ws", self._ws_endpoint)]

        if self._static_dir:
            static_path = Path(self._static_dir)
            if static_path.is_dir():
                routes.append(
                    Mount("/", app=StaticFiles(directory=str(static_path), html=True)),
                )
                logger.info("Serving static files from %s", static_path)

        return Starlette(routes=routes)

    async def _ws_endpoint(self, ws: WebSocket) -> None:
        await ws.accept()
        logger.info("Client connected: %s", ws.client)

        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._client_connected.set()

        recv_task = asyncio.create_task(self._receive_loop(ws))
        send_task = asyncio.create_task(self._send_loop(ws))

        try:
            done, pending = await asyncio.wait(
                {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            self._client_connected.clear()
            logger.info("Client disconnected: %s", ws.client)

    async def _receive_loop(self, ws: WebSocket) -> None:
        try:
            while not self._stop:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if "text" in msg:
                    try:
                        self._pending_metadata = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        pass

                elif "bytes" in msg:
                    buf = np.frombuffer(msg["bytes"], dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        meta_copy = self._pending_metadata.copy()
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._frame_queue.put_nowait((bgr, meta_copy))
                        except queue.Full:
                            pass
                        self._pending_metadata = {}
        except WebSocketDisconnect:
            pass

    async def _send_loop(self, ws: WebSocket) -> None:
        try:
            while not self._stop:
                result = await self._result_queue.get()
                await ws.send_json(result)
        except (WebSocketDisconnect, RuntimeError):
            pass
