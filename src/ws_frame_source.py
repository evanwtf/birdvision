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
import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketFrameSource:
    """Receives JPEG frames over WebSocket, yields (frame_no, bgr) pairs.

    Args:
        host:                 Bind address (default 0.0.0.0)
        port:                 Listen port (default 8765)
        static_dir:           Directory to serve as static files at /
        max_buffered_frames:  Max frames to buffer before dropping
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        static_dir: Optional[str] = None,
        max_buffered_frames: int = 2,
        upload_handler=None,
        client_idle_timeout_seconds: float = 30.0,
    ):
        self._host = host
        self._port = port
        self._static_dir = static_dir
        self._upload_handler = upload_handler
        self._client_idle_timeout = client_idle_timeout_seconds
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
        self._client_lock = threading.Lock()

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

    def _claim_client(self) -> bool:
        """Return True if this connection may become the active stream owner."""
        with self._client_lock:
            if self._client_connected.is_set():
                return False
            self._client_connected.set()
            return True

    def _release_client(self) -> None:
        """Release the active stream owner slot."""
        with self._client_lock:
            self._client_connected.clear()

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

        config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.info("WebSocket server listening on http://%s:%d", self._host, self._port)
        self._loop.run_until_complete(server.serve())

    def _shutdown_server(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    def _build_app(self) -> Starlette:
        routes = [WebSocketRoute("/ws", self._ws_endpoint)]

        if self._upload_handler:
            routes.append(Route("/upload", self._upload_endpoint, methods=["POST"]))

        if self._static_dir:
            static_path = Path(self._static_dir)
            if static_path.is_dir():
                routes.append(
                    Mount("/", app=StaticFiles(directory=str(static_path), html=True)),
                )
                logger.info("Serving static files from %s", static_path)

        return Starlette(routes=routes)

    async def _upload_endpoint(self, request: Request) -> JSONResponse:
        form = await request.form()
        upload = form.get("file")
        if not upload:
            return JSONResponse({"error": "No file provided"}, status_code=400)

        content = await upload.read()
        filename = upload.filename or "upload"
        content_type = upload.content_type or ""
        logger.info("Upload received: %s  %s  %d bytes", filename, content_type, len(content))

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self._upload_handler, content, filename, content_type,
            )
        except Exception as exc:
            logger.exception("Upload processing failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

        return JSONResponse(result)

    async def _ws_endpoint(self, ws: WebSocket) -> None:
        if not self._claim_client():
            await ws.accept()
            logger.info("Rejecting extra client while stream is active: %s", ws.client)
            await ws.send_json({
                "error": (
                    "BirdVision is already in use by another camera. "
                    "Try again when that session disconnects."
                ),
            })
            await asyncio.sleep(0.25)
            await ws.close(code=1013, reason="BirdVision is already in use")
            return

        await ws.accept()
        logger.info("Client connected: %s", ws.client)

        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        recv_task = asyncio.create_task(self._receive_loop(ws))
        send_task = asyncio.create_task(self._send_loop(ws))

        try:
            done, pending = await asyncio.wait(
                {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            self._release_client()
            logger.info("Client disconnected: %s", ws.client)

    async def _receive_loop(self, ws: WebSocket) -> None:
        try:
            while not self._stop:
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(),
                        timeout=self._client_idle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "Closing idle client after %.0fs without frames: %s",
                        self._client_idle_timeout,
                        ws.client,
                    )
                    await ws.close(code=1001)
                    break

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
