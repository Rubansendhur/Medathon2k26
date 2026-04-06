#!/bin/bash
echo "PhysioTracker v2 — Starting..."
cd "$(dirname "$0")/backend"
pip install fastapi uvicorn[standard] anthropic numpy --break-system-packages -q
echo ""
echo "  Dashboard: http://localhost:8000"
echo "  Phone IMU: http://<YOUR_IP>:8000/phone"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
