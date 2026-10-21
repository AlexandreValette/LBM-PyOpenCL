# -*- coding: utf-8 -*-
"""Web interface of the LBM solver : python app.py  (then open http://localhost:8000).

Protocol of the WebSocket /ws (so another client, e.g. Unity, can use it) :
  client -> server (text, JSON)
    {"type": "start", "config": {...}}   start (or restart) a simulation, see simulation_session.DEFAULT_CONFIG
    {"type": "set", "params": {...}}     change live parameters (re, display, obstacle_*, ...)
    {"type": "stop"}
    {"type": "ack"}                      the last frame has been displayed : the server may send the next one
  server -> client
    binary message : one JPEG frame (origin top left, y axis already pointing up)
    text message   : {"type": "stats", "state", "error", "fps", "mlups", "iteration", "device", "grid"}
"""
import asyncio
import json
import os
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

import ocl_driver
import simulation_session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8000

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "web", "index.html"))


@app.get("/api/devices")
def devices():
    return JSONResponse(ocl_driver.list_devices())


@app.get("/api/config")
def config():
    return JSONResponse({"defaults": simulation_session.DEFAULT_CONFIG,
                         "obstacles": simulation_session.list_obstacles()})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    session = None
    ack = asyncio.Event()
    ack.set()

    async def pump():
        """Send the stats twice a second, and a new frame each time the client acknowledged the last one."""
        nonlocal session
        last_stats, last_frame_id = 0., 0
        while True:
            await asyncio.sleep(0.004)
            s = session
            if s is None:
                continue
            now = loop.time()
            if now - last_stats > 0.5:
                last_stats = now
                await ws.send_text(json.dumps(s.stats()))
            if ack.is_set() and s.state == "running":
                frame_id, jpeg = s.get_frame()
                if frame_id != last_frame_id and jpeg:
                    last_frame_id = frame_id
                    ack.clear()
                    await ws.send_bytes(jpeg)
                    loop.call_later(1.0, ack.set)     # safety if the client never answers
                else:
                    s.request_frame()

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "start":
                if session is not None:
                    await loop.run_in_executor(None, session.stop)
                session = simulation_session.SimulationSession(msg.get("config"))
                ack.set()
                session.start()
            elif kind == "set" and session is not None:
                session.set_params(msg.get("params", {}))
            elif kind == "stop" and session is not None:
                await loop.run_in_executor(None, session.stop)
                await ws.send_text(json.dumps(session.stats()))
            elif kind == "ack":
                ack.set()
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        if session is not None:
            await loop.run_in_executor(None, session.stop)


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:%d" % PORT)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
