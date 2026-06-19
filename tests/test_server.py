import os
import pytest
from fastapi.testclient import TestClient

# Set environment variable to use test database
os.environ["DATABASE_PATH"] = "mydata/test_system.db"

# Now import app after setting environment variable
from src.server import app, init_server_db
from src.database import init_db

@pytest.fixture(autouse=True)
def setup_server_db():
    db_path = "mydata/test_system.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(db_path)
    yield
    if os.path.exists(db_path):
        os.remove(db_path)

def test_dashboard_route():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "LEAF SPRING VERIFICATION SYSTEM" in response.text

def test_rest_api():
    client = TestClient(app)

    # 1. Check initial queues
    response = client.get("/api/queue")
    assert response.status_code == 200
    assert response.json() == {"left": [], "right": []}

    # 2. Add left and right events
    response = client.post("/api/event", json={"side": "left", "model": "GREEN_A"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    response = client.post("/api/event", json={"side": "right", "model": "GREEN_A"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify queue contains them
    response = client.get("/api/queue")
    assert response.status_code == 200
    assert response.json() == {"left": ["GREEN_A"], "right": ["GREEN_A"]}

    # 3. Pop queue
    response = client.post("/api/queue/pop", json={"side": "left"})
    assert response.status_code == 200
    assert response.json() == {"model": "GREEN_A"}

    # Verify popped
    response = client.get("/api/queue")
    assert response.json() == {"left": [], "right": ["GREEN_A"]}

    # 4. Clear queue
    client.post("/api/queue/clear")
    response = client.get("/api/queue")
    assert response.json() == {"left": [], "right": []}

def test_history_api():
    client = TestClient(app)

    # Check empty history
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.json() == []

    # Insert a run
    payload = {
        "fl_model": "RED_A",
        "fr_model": "RED_A",
        "rl_model": "BLUE_B",
        "rr_model": "BLUE_B",
        "status": "PASS"
    }
    response = client.post("/api/history", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify history
    response = client.get("/api/history")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["fl_model"] == "RED_A"
    assert runs[0]["status"] == "PASS"

def test_websockets():
    client = TestClient(app)

    # Connect client (dashboard) first
    with client.websocket_connect("/ws/client") as client_ws:
        # First message is the initial queue sync
        init_msg = client_ws.receive_json()
        assert init_msg["type"] == "queue"

        # Connect tracker
        with client.websocket_connect("/api/ws/tracker") as tracker_ws:
            # Send state update from tracker
            test_state = {
                "active": True,
                "ref_L": [100, 200],
                "ref_R": [400, 200],
                "slots": {},
                "left_queue": [],
                "right_queue": [],
                "front_match": "PENDING",
                "rear_match": "PENDING",
                "status": "PENDING"
            }
            payload = {
                "state": test_state,
                "frame": "data:image/jpeg;base64,abc"
            }
            tracker_ws.send_json(payload)

            # Dashboard client should receive the update
            received = client_ws.receive_json()
            assert received["state"]["active"] is True
            assert received["frame"] == "data:image/jpeg;base64,abc"

            # Send inactive state update to trigger automatic run logging
            test_state_inactive = {
                "active": False,
                "ref_L": None,
                "ref_R": None,
                "slots": {
                    "FL": {"occupied": True, "model": "RED_A"},
                    "FR": {"occupied": True, "model": "RED_A"},
                    "RL": {"occupied": True, "model": "BLUE_B"},
                    "RR": {"occupied": True, "model": "BLUE_B"}
                },
                "left_queue": [],
                "right_queue": [],
                "front_match": "PASS",
                "rear_match": "PASS",
                "status": "PASS"
            }
            payload_inactive = {
                "state": test_state_inactive,
                "frame": "data:image/jpeg;base64,xyz"
            }
            tracker_ws.send_json(payload_inactive)

            # Receive the update on client dashboard
            received_inactive = client_ws.receive_json()
            assert received_inactive["state"]["active"] is False

    # Check that database has recorded the vehicle run
    response = client.get("/api/history")
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["fl_model"] == "RED_A"
    assert runs[0]["status"] == "PASS"
