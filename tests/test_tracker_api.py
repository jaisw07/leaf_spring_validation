import pytest
import requests_mock
from src.tracker import GeometryTracker

API_URL = "http://localhost:8000"

def test_tracker_sync_queues():
    tracker = GeometryTracker(api_url=API_URL)
    
    with requests_mock.Mocker() as m:
        m.get(f"{API_URL}/api/queue", json={"left": ["RED_A"], "right": ["BLUE_B"]})
        
        tracker.sync_queues()
        
        assert tracker.left_queue == ["RED_A"]
        assert tracker.right_queue == ["BLUE_B"]

def test_tracker_add_events():
    tracker = GeometryTracker(api_url=API_URL)
    
    with requests_mock.Mocker() as m:
        m.post(f"{API_URL}/api/event", json={"status": "success"})
        
        tracker.add_left_event("RED_A")
        tracker.add_right_event("BLUE_B")
        
        assert m.call_count == 2
        assert m.request_history[0].json() == {"side": "left", "model": "RED_A"}
        assert m.request_history[1].json() == {"side": "right", "model": "BLUE_B"}

def test_tracker_pop_from_api_success():
    tracker = GeometryTracker(api_url=API_URL)
    # Mock slot and detections to trigger occupancy
    tracker.slots["FL"]["center"] = (100, 100)
    tracker.slots["FL"]["occupied"] = False
    
    with requests_mock.Mocker() as m:
        m.post(f"{API_URL}/api/queue/pop", json={"model": "RED_A"})
        m.get(f"{API_URL}/api/queue", json={"left": [], "right": []})
        
        detections = [{"center": (100, 100)}]
        tracker._check_slot_occupancy("FL", detections)
        
        assert tracker.slots["FL"]["occupied"] is True
        assert tracker.slots["FL"]["model"] == "RED_A"
        assert m.call_count == 2 # Pop + Sync

def test_tracker_pop_from_api_fallback():
    tracker = GeometryTracker(api_url=API_URL)
    tracker.slots["FL"]["center"] = (100, 100)
    tracker.slots["FL"]["occupied"] = False
    tracker.left_queue = ["FALLBACK_MODEL"]
    
    # Mock connection failure
    with requests_mock.Mocker() as m:
        m.post(f"{API_URL}/api/queue/pop", exc=Exception("Connection refused"))
        
        detections = [{"center": (100, 100)}]
        tracker._check_slot_occupancy("FL", detections)
        
        # Should fallback to local queue and pop FALLBACK_MODEL
        assert tracker.slots["FL"]["occupied"] is True
        assert tracker.slots["FL"]["model"] == "FALLBACK_MODEL"
        assert tracker.left_queue == []
