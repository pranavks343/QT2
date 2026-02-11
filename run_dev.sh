#!/bin/bash
echo "Starting Sniper Framework..."
echo ""
echo "Starting FastAPI backend on port 8000..."
cd sniper_framework && python run_server.py &
BACKEND_PID=$!
echo ""
echo "Starting Next.js frontend on port 3000..."
cd ../sniper-ui && npm run dev &
FRONTEND_PID=$!
echo ""
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
wait
