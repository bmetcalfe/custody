"""Minimal GFW API smoke test — reads GFW_API_TOKEN from .env, pings a small AOI."""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GFW_API_TOKEN")
if not TOKEN or TOKEN == "your_token_here":
    sys.exit("ERR: GFW_API_TOKEN not set in .env (see .env.example)")

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
# 24 h window ending now, 0.1° box around the Cuarteron hero feature
params = {
    "datasets[0]": "public-global-fishing-effort:latest",
    "date-range": "2023-08-06,2023-08-07",
    "format": "JSON",
    "spatial-resolution": "HIGH",
    "temporal-resolution": "DAILY",
    "group-by": "FLAG",
}
body = {
    "geojson": {
        "type": "Polygon",
        "coordinates": [[
            [114.615, 8.806], [114.715, 8.806],
            [114.715, 8.906], [114.615, 8.906],
            [114.615, 8.806],
        ]],
    },
}
r = requests.post(URL, headers={"Authorization": f"Bearer {TOKEN}"}, params=params, json=body, timeout=20)
if r.ok:
    dataset = params["datasets[0]"].split(":")[0]
    print(f"OK — {dataset} reachable (HTTP {r.status_code})")
else:
    print(f"ERR {r.status_code}: {r.text[:200]}")
