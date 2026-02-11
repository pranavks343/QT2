#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "  SNIPER TRADING FRAMEWORK - Starting Development Servers"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Start FastAPI backend
echo -e "${BLUE}[1/2] Starting FastAPI backend...${NC}"
cd sniper_framework && python run_server.py &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo ""

# Wait for backend to initialize
sleep 3

# Start Next.js frontend
echo -e "${BLUE}[2/2] Starting Next.js frontend...${NC}"
cd ../sniper-ui && npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✓ Both servers are running!${NC}"
echo ""
echo "  Backend API:     http://localhost:8000"
echo "  API Docs:        http://localhost:8000/docs"
echo "  Frontend UI:     http://localhost:3000"
echo "  WebSocket:       ws://localhost:8000/ws/{symbol}/{timeframe}"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo ""
echo "═══════════════════════════════════════════════════════════"

# Function to kill both processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for both processes
wait
