@echo off
title FLARE Demo

cd /d "%~dp0..\Anti-Interference-2D-Vision"

echo Starting FLARE Demo Server...
echo Please visit: http://localhost:8501
echo Press Ctrl+C to stop
echo.

python -m streamlit run demo_streamlit.py --server.port 8501
