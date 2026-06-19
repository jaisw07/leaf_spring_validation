# Leaf Spring Verification System - API Integration & External Access Guide

This guide is designed for developers integrating external hardware modules (such as side/rear camera modules, RFID readers, barcode scanners, or PLC line managers) with the leaf spring verification system. It details the REST API specifications, network configurations, external IP access setup, and dashboard access steps.

---

## 1. Network & External IP Configuration

To allow external device modules and remote dashboards to communicate with the FastAPI server, you must configure the server's network binding and ensure the correct port is open.

### Step 1: Bind Server to All Network Interfaces (`0.0.0.0`)
By default, servers bound to local loopback (`127.0.0.1`) only accept connections from the local machine. To receive requests from external IPs on the local area network (LAN), start the server with the `--host 0.0.0.0` argument.

```powershell
# Run backend server listening on all network interfaces of port 8000
conda run -n dump uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Step 2: Determine Server Host IP Address
To find the IP address of your Windows host server machine on the local network:
1. Open PowerShell or Command Prompt.
2. Run the command: `ipconfig`
3. Locate your active network adapter (e.g., *Ethernet adapter* or *Wireless LAN adapter Wi-Fi*).
4. Note the **IPv4 Address** (typically in the range `192.168.X.X` or `10.X.X.X`). 

*Assume for this guide that the Server IP is **`192.168.1.50`**.*

### Step 3: Open Port 8000 in Windows Defender Firewall
External devices will fail to connect if Windows Firewall blocks incoming TCP traffic on port `8000`. 

To open the port, run PowerShell as **Administrator** and execute:
```powershell
New-NetFirewallRule -DisplayName "FastAPI Web Server (Port 8000)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000
```
Alternatively, configure it via GUI:
1. Open **Windows Defender Firewall with Advanced Security**.
2. Click **Inbound Rules** > **New Rule...**
3. Choose **Port** > Click Next.
4. Select **TCP** and enter **`8000`** in **Specific local ports** > Click Next.
5. Choose **Allow the connection** > Click Next.
6. Check all profiles (Domain, Private, Public) > Click Next.
7. Name it `FastAPI Web Server (Port 8000)` and click Finish.

---

## 2. API Endpoint Reference for Camera & Hardware Modules

External modules update the left and right assembly queues by hitting the FastAPI server's HTTP REST endpoints.

```
+-----------------------------------+
|  Rear/Side Camera / RFID / Barcode|
+-----------------------------------+
                  │
                  ▼ (POST /api/event)
        ┌───────────────────┐
        │  FastAPI Backend  │
        │  (Port 8000)      │
        └─────────┬─────────┘
                  │ (Save queue to SQLite & Broadcast WS)
                  ▼
        ┌───────────────────┐
        │ Vision Tracker    │
        │ (pops queue item) │
        └───────────────────┘
```

### 1. Add Pickup Event (Enqueue Item)
* **URL**: `http://<server-ip-address>:8000/api/event`
* **Method**: `POST`
* **Content-Type**: `application/json`
* **Description**: Invoked by a side/rear camera or scan sensor whenever a leaf spring model is picked up or assigned to a side. The item is appended to that side's FIFO queue in the database and broadcasted.

#### Request Payload
```json
{
  "side": "left",
  "model": "GREEN_TRIANGLE"
}
```
* **Parameters**:
  * `side` (string, Required): Must be exactly `"left"` or `"right"`. Matches the physical side of the vehicle assembly line.
  * `model` (string, Required): The identified leaf spring model designation (e.g., `RED_A`, `GREEN_TRIANGLE`, `YELLOW_B`).

#### Response Examples
* **Success (`200 OK`)**:
  ```json
  {
    "status": "success"
  }
  ```
* **Validation Error (`400 Bad Request`)**:
  ```json
  {
    "detail": "Invalid side. Must be 'left' or 'right'"
  }
  ```

---

### 2. Retrieve Current Queue States
* **URL**: `http://<server-ip-address>:8000/api/queue`
* **Method**: `GET`
* **Description**: Retrieves lists of currently enqueued models for both the left and right sides.

#### Response Example (`200 OK`)
```json
{
  "left": [
    "RED_A",
    "GREEN_TRIANGLE"
  ],
  "right": [
    "RED_A"
  ]
}
```

---

### 3. Pop Queue Item
* **URL**: `http://<server-ip-address>:8000/api/queue/pop`
* **Method**: `POST`
* **Content-Type**: `application/json`
* **Description**: Dequeues and returns the oldest item for the specified side. Usually called automatically by the vision tracker upon chassis region occupancy.

#### Request Payload
```json
{
  "side": "left"
}
```

#### Response Example (`200 OK`)
```json
{
  "model": "RED_A"
}
```
*(Returns `{"model": null}` if the specified queue is empty).*

---

### 4. Clear All Queues
* **URL**: `http://<server-ip-address>:8000/api/queue/clear`
* **Method**: `POST`
* **Description**: Empties both Left and Right FIFO queues.

#### Response Example (`200 OK`)
```json
{
  "status": "success"
}
```

---

### 5. Clear Queues and Reset Vision Tracker
* **URL**: `http://<server-ip-address>:8000/api/tracker/reset`
* **Method**: `POST`
* **Description**: Performs a database-wide queue reset and signals the main Top-Camera Vision Tracker client (via WebSockets) to reset its spatial geometry trackers and rewind the video (if using local MP4 stream).

#### Response Example (`200 OK`)
```json
{
  "status": "success"
}
```

---

## 3. Remote Dashboard Access

Once the server host IP is configured and port `8000` is allowed through the firewall:

1. **Dashboard Browser URL**: Open any modern web browser on an external machine connected to the same LAN and navigate to:
   ```text
   http://<server-ip-address>:8000/
   ```
   *(e.g., `http://192.168.1.50:8000/`)*
2. **WebSocket Communication**: The dashboard communicates dynamically with the backend. It connects to the client websocket:
   ```text
   ws://<server-ip-address>:8000/ws/client
   ```
   This WebSocket streams live processed warping overlays, region bounding boxes, active tracking status, and current queues.

---

## 4. Code Integration Recipes for Camera Modules

Below are practical code implementation recipes to hit the event queue endpoint (`/api/event`) from external modules using different languages.

### curl (Shell/Command Line)
```bash
curl -X POST "http://192.168.1.50:8000/api/event" \
     -H "Content-Type: application/json" \
     -d '{"side": "left", "model": "GREEN_TRIANGLE"}'
```

### Python (Camera/RFID Scripts)
Ensure you have the `requests` library installed (`pip install requests`):
```python
import requests

def send_pickup_event(server_ip: str, side: str, model_id: str):
    url = f"http://{server_ip}:8000/api/event"
    payload = {
        "side": side,
        "model": model_id
    }
    
    try:
        response = requests.post(url, json=payload, timeout=3.0)
        if response.status_code == 200:
            print(f"Successfully enqueued {model_id} on {side} side.")
        else:
            print(f"Error {response.status_code}: {response.json().get('detail')}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to verification server: {e}")

# Example Usage
send_pickup_event("192.168.1.50", "left", "RED_A")
```

### Node.js (Javascript/Typescript)
```javascript
const axios = require('axios'); // or use native fetch

async function sendPickupEvent(serverIp, side, model) {
    const url = `http://${serverIp}:8000/api/event`;
    
    try {
        const response = await axios.post(url, { side, model });
        console.log('Event enqueued successfully:', response.data);
    } catch (error) {
        if (error.response) {
            console.error(`API Error (${error.response.status}):`, error.response.data.detail);
        } else {
            console.error('Connection failed:', error.message);
        }
    }
}

// Example Usage
sendPickupEvent('192.168.1.50', 'right', 'BLUE_C');
```

### Windows PowerShell
```powershell
$headers = @{
    "Content-Type" = "application/json"
}
$body = @{
    side = "left"
    model = "GREEN_TRIANGLE"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://192.168.1.50:8000/api/event" -Method Post -Headers $headers -Body $body
```
