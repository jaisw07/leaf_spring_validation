import os
import json
import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Set environment variables for config
os.environ["VIDEO_SOURCE"] = "rtsp://mock_video"
os.environ["WS_URL"] = "ws://localhost:8001/api/ws/tracker"
os.environ["API_URL"] = "http://localhost:8001"

from src.run_tracker import stream_tracker

@pytest.mark.asyncio
@patch("src.run_tracker.YOLO")
@patch("src.run_tracker.cv2.VideoCapture")
@patch("src.run_tracker.websockets.connect")
@patch("src.run_tracker.np.load")
@patch("builtins.open")
async def test_stream_tracker_loop(mock_open, mock_np_load, mock_ws_connect, mock_video_capture, mock_yolo_class):
    # Mock homography load and warp config JSON
    mock_np_load.return_value = np.eye(3)
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps({"lane_width": 576, "lane_height": 1080})
    mock_open.return_value.__enter__.return_value = mock_file

    # Mock YOLO model
    mock_model = MagicMock()
    # Mock predict result with empty boxes
    mock_pred = MagicMock()
    mock_pred.boxes = []
    mock_model.predict.return_value = [mock_pred]
    mock_yolo_class.return_value = mock_model

    # Mock cv2 VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 25.0 # Native FPS
    
    # Return dummy frame on every read
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap

    # Mock Websocket connection and send method
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws_connect.return_value = mock_ws

    # Mock tracker API requests (sync_queues, event popping)
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"left": [], "right": []}

        # Run stream_tracker with a timeout (since RTSP stream retries infinitely)
        try:
            await asyncio.wait_for(stream_tracker(), timeout=1.0)
        except (asyncio.TimeoutError, SystemExit):
            pass

        # Verify websockets.connect was called with the correct WS URL
        mock_ws_connect.assert_called_with("ws://localhost:8001/api/ws/tracker")

        # Verify ws.send was called with the tracking state payload
        mock_ws_entered = mock_ws.__aenter__.return_value
        assert mock_ws_entered.send.call_count >= 1
        sent_data = json.loads(mock_ws_entered.send.call_args[0][0])
        assert "state" in sent_data
        assert "frame" in sent_data
        assert sent_data["state"]["active"] is False # because no detections, so inactive
