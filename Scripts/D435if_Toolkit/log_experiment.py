"""
Experiment Logger for IEEE RA-L MAC-VIO Research.
Logs experiment metadata, run parameters, and performance metrics to the Google Sheet webhook.
"""

import sys
import json
import argparse
import datetime
import httpx

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzgisdNcrAwaYt5tUR1lYBh2xC_5lt9QA7afc-KFlPkx6MsMdnDrAyM7wJdzEDAjvk/exec"

def log_to_sheet(exp_id: str, title: str, emitter: str, duration: str, frames: int, imu_samples: int, result: str, notes: str, color: str | None = None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_values = [
        timestamp,
        exp_id,
        title,
        emitter,
        duration,
        frames,
        imu_samples,
        result,
        notes,
        "Intel RealSense D435if (USB 3.2)",
        "MAC-VO (Fast/Performant)"
    ]
    payload = {
        "row": row_values,
        "timestamp": timestamp,
        "exp_id": exp_id,
        "title": title,
        "emitter": emitter,
        "duration": duration,
        "frames": frames,
        "imu_samples": imu_samples,
        "result": result,
        "notes": notes,
        "device": "Intel RealSense D435if (USB 3.2)",
        "model": "MAC-VO (Fast/Performant)"
    }
    if color:
        payload["color"] = color
    
    print(f"Logging experiment [{exp_id}] to Google Sheet...")
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0) as client:
            resp = client.post(WEBHOOK_URL, json=payload)
            if resp.status_code == 200 and "Error" not in resp.text and "error" not in resp.text.lower():
                print(f"[SUCCESS] Successfully logged {exp_id} to Google Sheet: {resp.text.strip()}")
                return True
            else:
                print(f"[ERROR] Webhook rejected payload (status {resp.status_code}): {resp.text[:300]}")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to send log to webhook: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log experiment to Google Sheet")
    parser.add_argument("--exp_id", type=str, required=True, help="e.g. EXP-001, EXP-002, EXP-003")
    parser.add_argument("--title", type=str, required=True, help="Experiment title/description")
    parser.add_argument("--emitter", type=str, default="OFF", help="ON or OFF")
    parser.add_argument("--duration", type=str, default="60s", help="Duration")
    parser.add_argument("--frames", type=int, default=0, help="Total frames")
    parser.add_argument("--imu_samples", type=int, default=0, help="IMU samples")
    parser.add_argument("--result", type=str, default="Completed", help="Outcome / Drift / Metrics")
    parser.add_argument("--notes", type=str, default="", help="Observations and notes")
    parser.add_argument("--color", type=str, default=None, help="Hex color code for row highlight, e.g. #d9ead3")
    
    args = parser.parse_args()
    log_to_sheet(args.exp_id, args.title, args.emitter, args.duration, args.frames, args.imu_samples, args.result, args.notes, args.color)
