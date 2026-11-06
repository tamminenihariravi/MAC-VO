#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.error

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzgisdNcrAwaYt5tUR1lYBh2xC_5lt9QA7afc-KFlPkx6MsMdnDrAyM7wJdzEDAjvk/exec"

def send_row(row_data, color="#d9ead3"):
    """
    Appends a single row (list of strings/values) directly to the Google Sheet.
    """
    payload_dict = {"row": row_data}
    if color:
        payload_dict["color"] = color
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            print(f"[Google Sheet] Appended row '{row_data[0]}': {res_body}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[Google Sheet] HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"[Google Sheet] Network Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User passed a custom row via CLI arguments
        # Example: ./sync_to_google_sheet.py EXP-007 2026-10-25 Laptop commit1 "Idea" "Config" "Lighting" "Algo" "Cov" "0.04" "1.2" "Notes" "Next"
        custom_row = sys.argv[1:]
        send_row(custom_row)
    else:
        print("Usage: ./sync_to_google_sheet.py <Exp_ID> <Date> <Machine> <Git_Commit> <Idea> <Sensor_Config> <Motion> <Algo> <Cov_Type> <ATE_m> <RPE_deg_m> <Observations> <Next_Steps>")
