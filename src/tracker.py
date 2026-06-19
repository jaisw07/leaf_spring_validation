import numpy as np
import requests

class GeometryTracker:
    def __init__(self, dx_L=6.7325, dy_L=-216.4267, dx_R=-4.2520, dy_R=-206.6953, api_url=None):
        # Calibration offsets (Front relative to Rear)
        self.dx_L = dx_L
        self.dy_L = dy_L
        self.dx_R = dx_R
        self.dy_R = dy_R
        self.api_url = api_url

        # State queues for side cameras (FIFO)
        self.left_queue = []
        self.right_queue = []


        # Tracking state
        self.active = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self.cooldown_frames = 0
        self.cooldown_after_reset = 5  # Frames to skip after lost-frames reset

        # Reference points (Rear-Left, Rear-Right)
        self.ref_L = None
        self.ref_R = None

        # Smoothing factor (EMA)
        self.alpha = 0.8

        # Estimated velocities (pixels per frame)
        self.vx_L, self.vy_L = 0.0, 0.0
        self.vx_R, self.vy_R = 0.0, 0.0

        # Slot states
        # Each slot: {"occupied": bool, "model": str, "center": (x, y)}
        self.slots = {
            "FL": {"occupied": False, "model": None, "center": None},
            "FR": {"occupied": False, "model": None, "center": None},
            "RL": {"occupied": False, "model": None, "center": None},
            "RR": {"occupied": False, "model": None, "center": None}
        }

        # ROI half-sizes (width, height)
        self.roi_w = 50
        self.roi_h = 100

    def sync_queues(self):
        if not self.api_url:
            return
        try:
            response = requests.get(f"{self.api_url}/api/queue", timeout=0.5)
            if response.status_code == 200:
                data = response.json()
                self.left_queue = data.get("left", [])
                self.right_queue = data.get("right", [])
        except Exception:
            pass

    def add_left_event(self, model_name):
        self.left_queue.append(model_name)
        if self.api_url:
            try:
                requests.post(f"{self.api_url}/api/event", json={"side": "left", "model": model_name}, timeout=1.0)
            except Exception:
                pass

    def add_right_event(self, model_name):
        self.right_queue.append(model_name)
        if self.api_url:
            try:
                requests.post(f"{self.api_url}/api/event", json={"side": "right", "model": model_name}, timeout=1.0)
            except Exception:
                pass

    def reset(self, clear_queues=True):
        self.active = False
        self.lost_frames = 0
        self.ref_L = None
        self.ref_R = None
        self.vx_L, self.vy_L = 0.0, 0.0
        self.vx_R, self.vy_R = 0.0, 0.0
        for slot in self.slots.values():
            slot["occupied"] = False
            slot["model"] = None
            slot["center"] = None
        
        if clear_queues:
            # Clear server queues only on explicit user reset,
            # not on video loop or lost-frames timeout.
            if self.api_url:
                try:
                    requests.post(f"{self.api_url}/api/queue/clear", timeout=1.0)
                except Exception:
                    pass
            self.left_queue = []
            self.right_queue = []

    def update(self, detections, lane_width=576, lane_height=1080):
        """
        Update tracker with detections from current frame.
        detections: list of dicts {"class_id": int, "bbox": [x1, y1, x2, y2], "conf": float}
        """
        # After a lost-frames reset, skip processing for a few frames
        # to prevent rapid active/inactive oscillation.
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return self.get_state()

        # Separate detections into left/right and class
        left_rods = []
        right_rods = []
        left_springs = []
        right_springs = []

        mid_x = lane_width / 2

        for det in detections:
            cls = det["class_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            det_obj = {"center": (cx, cy), "bbox": det["bbox"], "conf": det["conf"]}

            if cx < mid_x:
                if cls == 0:
                    left_rods.append(det_obj)
                elif cls == 1:
                    left_springs.append(det_obj)
            else:
                if cls == 0:
                    right_rods.append(det_obj)
                elif cls == 1:
                    right_springs.append(det_obj)

        # 1. Update/Track reference points
        new_ref_L = self._find_reference(left_rods, left_springs, "RL")
        new_ref_R = self._find_reference(right_rods, right_springs, "RR")

        # Handle active state transition
        if new_ref_L is not None or new_ref_R is not None:
            self.lost_frames = 0
            if not self.active:
                self.active = True
            
            # Update Left Reference Point
            if new_ref_L is not None:
                if self.ref_L is not None:
                    # Update velocity
                    dx = new_ref_L[0] - self.ref_L[0]
                    dy = new_ref_L[1] - self.ref_L[1]
                    self.vx_L = self.alpha * dx + (1 - self.alpha) * self.vx_L
                    self.vy_L = self.alpha * dy + (1 - self.alpha) * self.vy_L
                    # EMA smoothing
                    self.ref_L = (
                        self.alpha * new_ref_L[0] + (1 - self.alpha) * self.ref_L[0],
                        self.alpha * new_ref_L[1] + (1 - self.alpha) * self.ref_L[1]
                    )
                else:
                    self.ref_L = new_ref_L
            else:
                # Predict Left using velocity
                if self.ref_L is not None:
                    self.ref_L = (self.ref_L[0] + self.vx_L, self.ref_L[1] + self.vy_L)

            # Update Right Reference Point
            if new_ref_R is not None:
                if self.ref_R is not None:
                    # Update velocity
                    dx = new_ref_R[0] - self.ref_R[0]
                    dy = new_ref_R[1] - self.ref_R[1]
                    self.vx_R = self.alpha * dx + (1 - self.alpha) * self.vx_R
                    self.vy_R = self.alpha * dy + (1 - self.alpha) * self.vy_R
                    # EMA smoothing
                    self.ref_R = (
                        self.alpha * new_ref_R[0] + (1 - self.alpha) * self.ref_R[0],
                        self.alpha * new_ref_R[1] + (1 - self.alpha) * self.ref_R[1]
                    )
                else:
                    self.ref_R = new_ref_R
            else:
                # Predict Right using velocity
                if self.ref_R is not None:
                    self.ref_R = (self.ref_R[0] + self.vx_R, self.ref_R[1] + self.vy_R)

        else:
            if self.active:
                self.lost_frames += 1
                if self.lost_frames >= self.max_lost_frames:
                    self.reset(clear_queues=False)
                    self.cooldown_frames = self.cooldown_after_reset
                    return self.get_state()
                # Predict references using last known velocities
                if self.ref_L is not None:
                    self.ref_L = (self.ref_L[0] + self.vx_L, self.ref_L[1] + self.vy_L)
                if self.ref_R is not None:
                    self.ref_R = (self.ref_R[0] + self.vx_R, self.ref_R[1] + self.vy_R)

        if not self.active:
            return self.get_state()

        # Cross-predict references if one is missing but the other is present
        # This keeps the geometry synchronized if one side is temporarily lost
        if self.ref_L is None and self.ref_R is not None:
            # Shift right reference left by approximate lane half-width
            self.ref_L = (self.ref_R[0] - 250, self.ref_R[1])
        elif self.ref_R is None and self.ref_L is not None:
            # Shift left reference right by approximate lane half-width
            self.ref_R = (self.ref_L[0] + 250, self.ref_L[1])

        # 2. Update Slot Centers
        self.slots["RL"]["center"] = self.ref_L
        self.slots["RR"]["center"] = self.ref_R
        self.slots["FL"]["center"] = (self.ref_L[0] + self.dx_L, self.ref_L[1] + self.dy_L)
        self.slots["FR"]["center"] = (self.ref_R[0] + self.dx_R, self.ref_R[1] + self.dy_R)

        # 3. Process Slot Occupancy Transitions
        self._check_slot_occupancy("FL", left_springs)
        self._check_slot_occupancy("RL", left_springs)
        self._check_slot_occupancy("FR", right_springs)
        self._check_slot_occupancy("RR", right_springs)

        return self.get_state()

    def _find_reference(self, rods, springs, rear_slot_key):
        """
        Resolve the reference point coordinate for a side.
        Prefer rod if visible. If rod is missing but rear spring is detected/installed,
        use the rear spring center.
        """
        # If rod is detected, use the highest confidence rod
        if rods:
            best_rod = max(rods, key=lambda x: x["conf"])
            return best_rod["center"]

        # If tracker is active, match springs to slots
        if self.active:
            # First preference: check if we detect a spring matching the rear slot (RL/RR)
            if self.slots[rear_slot_key]["center"] is not None:
                cx_ref, cy_ref = self.slots[rear_slot_key]["center"]
                for spring in springs:
                    cx, cy = spring["center"]
                    if abs(cx - cx_ref) < self.roi_w and abs(cy - cy_ref) < self.roi_h:
                        return spring["center"]
            
            # Second preference: check if we detect a spring matching the front slot (FL/FR)
            # and project it backward to estimate the rear reference coordinate
            front_slot_key = "FL" if rear_slot_key == "RL" else "FR"
            if self.slots[front_slot_key]["center"] is not None:
                cx_front, cy_front = self.slots[front_slot_key]["center"]
                for spring in springs:
                    cx, cy = spring["center"]
                    if abs(cx - cx_front) < self.roi_w and abs(cy - cy_front) < self.roi_h:
                        is_left = rear_slot_key.endswith("L")
                        dx = self.dx_L if is_left else self.dx_R
                        dy = self.dy_L if is_left else self.dy_R
                        return (cx - dx, cy - dy)
        else:
            # Tracker is NOT active. The vehicle is entering the frame.
            if springs:
                # Sort springs by y-coordinate descending (bottom/entry of frame to top)
                sorted_springs = sorted(springs, key=lambda x: x["center"][1], reverse=True)
                
                # Check if we have two springs representing front and rear
                matched_pair = False
                if len(sorted_springs) >= 2:
                    for i in range(len(sorted_springs)):
                        for j in range(i + 1, len(sorted_springs)):
                            sy_rear = sorted_springs[i]["center"][1]
                            sy_front = sorted_springs[j]["center"][1]
                            dy_actual = sy_rear - sy_front
                            # Check if actual dy is close to target dy (approx 210 pixels)
                            if 180 < dy_actual < 250:
                                best_spring = sorted_springs[i]
                                matched_pair = True
                                return best_spring["center"]
                
                if not matched_pair:
                    # Single spring (or no pair found). Assume it's the front spring entering first.
                    best_spring = sorted_springs[0]
                    is_left = rear_slot_key.endswith("L")
                    dx = self.dx_L if is_left else self.dx_R
                    dy = self.dy_L if is_left else self.dy_R
                    
                    sx, sy = best_spring["center"]
                    rx = sx - dx
                    ry = sy - dy
                    return (rx, ry)

        return None

    def _check_slot_occupancy(self, slot_key, detected_springs):
        slot = self.slots[slot_key]
        if slot["center"] is None:
            return

        cx_slot, cy_slot = slot["center"]

        # Look for a detected spring center inside the slot ROI
        occupied_now = False
        for spring in detected_springs:
            cx, cy = spring["center"]
            if abs(cx - cx_slot) < self.roi_w and abs(cy - cy_slot) < self.roi_h:
                occupied_now = True
                break

        # Transition empty -> occupied
        if occupied_now and not slot["occupied"]:
            slot["occupied"] = True
            
            # Pop event from corresponding side queue
            is_left = slot_key.endswith("L")
            side = "left" if is_left else "right"
            
            model = None
            if self.api_url:
                try:
                    response = requests.post(f"{self.api_url}/api/queue/pop", json={"side": side}, timeout=1.0)
                    if response.status_code == 200:
                        model = response.json().get("model")
                except Exception as e:
                    print(f"Error popping queue from API: {e}")
            
            if model is not None:
                slot["model"] = model
                self.sync_queues()
            else:
                # Fallback to local queue
                queue = self.left_queue if is_left else self.right_queue
                if queue:
                    slot["model"] = queue.pop(0)
                else:
                    slot["model"] = "UNKNOWN"

    def get_state(self):
        # Determine verification results
        fl_model = self.slots["FL"]["model"]
        fr_model = self.slots["FR"]["model"]
        rl_model = self.slots["RL"]["model"]
        rr_model = self.slots["RR"]["model"]

        # Front Match
        if not self.slots["FL"]["occupied"] or not self.slots["FR"]["occupied"]:
            front_match = "PENDING"
        elif fl_model == "UNKNOWN" or fr_model == "UNKNOWN":
            front_match = "FAIL"
        elif fl_model == fr_model:
            front_match = "PASS"
        else:
            front_match = "FAIL"

        # Rear Match
        if not self.slots["RL"]["occupied"] or not self.slots["RR"]["occupied"]:
            rear_match = "PENDING"
        elif rl_model == "UNKNOWN" or rr_model == "UNKNOWN":
            rear_match = "FAIL"
        elif rl_model == rr_model:
            rear_match = "PASS"
        else:
            rear_match = "FAIL"

        # Overall Status
        if front_match == "FAIL" or rear_match == "FAIL":
            overall_status = "FAIL"
        elif front_match == "PASS" and rear_match == "PASS":
            overall_status = "PASS"
        else:
            overall_status = "PENDING"

        return {
            "active": self.active,
            "ref_L": self.ref_L,
            "ref_R": self.ref_R,
            "slots": {k: {"occupied": v["occupied"], "model": v["model"], "center": v["center"]} 
                      for k, v in self.slots.items()},
            "left_queue": list(self.left_queue),
            "right_queue": list(self.right_queue),
            "front_match": front_match,
            "rear_match": rear_match,
            "status": overall_status
        }
