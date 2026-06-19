import os
import asyncio
from typing import Set, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.database import (
    init_db,
    add_queue_item,
    get_queue,
    pop_queue_item,
    clear_queues,
    save_vehicle_run,
    get_vehicle_runs
)

app = FastAPI(title="Leaf Spring Assembly Verification API")

# Allow CORS for dashboard UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve DB path
DB_PATH = os.getenv("DATABASE_PATH", "mydata/system.db")

def init_server_db():
    init_db(DB_PATH)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init_server_db()

# Trackers and clients connections
active_client_connections: Set[WebSocket] = set()
active_tracker_connections: Set[WebSocket] = set()
# Track tracker active state transition
tracker_was_active: bool = False

# REST Models
class SideEvent(BaseModel):
    side: str
    model: str

class PopRequest(BaseModel):
    side: str

class VehicleRunPayload(BaseModel):
    fl_model: str | None = None
    fr_model: str | None = None
    rl_model: str | None = None
    rr_model: str | None = None
    status: str

# Helper to broadcast queue updates to clients
async def broadcast_queue_update():
    left_q = get_queue("left", DB_PATH)
    right_q = get_queue("right", DB_PATH)
    payload = {
        "type": "queue",
        "left": left_q,
        "right": right_q
    }
    await broadcast_to_clients(payload)

# Helper to broadcast to all dashboard clients
async def broadcast_to_clients(message: dict):
    disconnected = set()
    for websocket in active_client_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.add(websocket)
    for ws in disconnected:
        active_client_connections.remove(ws)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    try:
        with open("src/static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)

# REST API Endpoints
@app.post("/api/event")
async def receive_event(event: SideEvent):
    if event.side not in ("left", "right"):
        raise HTTPException(status_code=400, detail="Invalid side. Must be 'left' or 'right'")
    add_queue_item(event.side, event.model, DB_PATH)
    await broadcast_queue_update()
    return {"status": "success"}

@app.get("/api/queue")
async def read_queues():
    left_q = get_queue("left", DB_PATH)
    right_q = get_queue("right", DB_PATH)
    return {"left": left_q, "right": right_q}

@app.post("/api/queue/pop")
async def pop_queue(req: PopRequest):
    if req.side not in ("left", "right"):
        raise HTTPException(status_code=400, detail="Invalid side. Must be 'left' or 'right'")
    model = pop_queue_item(req.side, DB_PATH)
    await broadcast_queue_update()
    return {"model": model}

@app.post("/api/queue/clear")
async def clear_all_queues():
    clear_queues(DB_PATH)
    await broadcast_queue_update()
    return {"status": "success"}

@app.get("/api/history")
async def read_history():
    return get_vehicle_runs(DB_PATH)

@app.post("/api/history")
async def create_history_record(run: VehicleRunPayload):
    save_vehicle_run(
        run.fl_model,
        run.fr_model,
        run.rl_model,
        run.rr_model,
        run.status,
        DB_PATH
    )
    return {"status": "success"}

@app.post("/api/tracker/reset")
async def reset_tracker():
    clear_queues(DB_PATH)
    await broadcast_queue_update()
    for ws in active_tracker_connections:
        try:
            await ws.send_json({"command": "reset"})
        except Exception:
            pass
    return {"status": "success"}

# WebSocket Endpoints
@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    active_client_connections.add(websocket)
    try:
        # Send initial queues on connect
        left_q = get_queue("left", DB_PATH)
        right_q = get_queue("right", DB_PATH)
        await websocket.send_json({
            "type": "queue",
            "left": left_q,
            "right": right_q
        })
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_client_connections.remove(websocket)
    except Exception:
        if websocket in active_client_connections:
            active_client_connections.remove(websocket)

@app.websocket("/api/ws/tracker")
async def websocket_tracker(websocket: WebSocket):
    global tracker_was_active
    await websocket.accept()
    active_tracker_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Broadcast state & frame directly to clients
            await broadcast_to_clients(data)

            # Check for active state transition to log vehicle run
            state = data.get("state", {})
            is_active = state.get("active", False)

            if tracker_was_active and not is_active:
                # Active -> Inactive transition. Save run to DB.
                slots = state.get("slots", {})
                fl = slots.get("FL", {}).get("model")
                fr = slots.get("FR", {}).get("model")
                rl = slots.get("RL", {}).get("model")
                rr = slots.get("RR", {}).get("model")
                status = state.get("status", "UNKNOWN")
                
                save_vehicle_run(fl, fr, rl, rr, status, DB_PATH)
            
            tracker_was_active = is_active
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in active_tracker_connections:
            active_tracker_connections.remove(websocket)
