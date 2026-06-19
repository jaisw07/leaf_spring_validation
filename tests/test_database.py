import os
import sqlite3
import pytest
from src.database import init_db, add_queue_item, get_queue, pop_queue_item, save_vehicle_run, get_vehicle_runs

DB_PATH = "mydata/test_system.db"

@pytest.fixture(autouse=True)
def setup_database():
    # Remove existing test DB if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    # Initialize DB
    init_db(DB_PATH)
    yield
    
    # Clean up after test
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_queue_operations():
    # Verify queues initially empty
    assert get_queue("left", DB_PATH) == []
    assert get_queue("right", DB_PATH) == []

    # Add items
    add_queue_item("left", "RED_A", DB_PATH)
    add_queue_item("left", "GREEN_B", DB_PATH)
    add_queue_item("right", "BLUE_C", DB_PATH)

    # Get queues
    assert get_queue("left", DB_PATH) == ["RED_A", "GREEN_B"]
    assert get_queue("right", DB_PATH) == ["BLUE_C"]

    # Pop item
    popped = pop_queue_item("left", DB_PATH)
    assert popped == "RED_A"
    assert get_queue("left", DB_PATH) == ["GREEN_B"]

    # Pop another
    popped = pop_queue_item("left", DB_PATH)
    assert popped == "GREEN_B"
    assert get_queue("left", DB_PATH) == []

    # Pop empty queue
    assert pop_queue_item("left", DB_PATH) is None

def test_vehicle_runs_operations():
    # Verify runs initially empty
    assert get_vehicle_runs(DB_PATH) == []

    # Save a vehicle run
    save_vehicle_run("RED_A", "RED_A", "BLUE_B", "BLUE_B", "PASS", DB_PATH)
    save_vehicle_run("GREEN_C", "RED_A", "BLUE_B", "BLUE_B", "FAIL", DB_PATH)

    # Get history
    runs = get_vehicle_runs(DB_PATH)
    assert len(runs) == 2

    # Should be sorted desc by id (newest first)
    assert runs[0]["fl_model"] == "GREEN_C"
    assert runs[0]["status"] == "FAIL"
    assert runs[0]["timestamp"] is not None

    assert runs[1]["fl_model"] == "RED_A"
    assert runs[1]["status"] == "PASS"

