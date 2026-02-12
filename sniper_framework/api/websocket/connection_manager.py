"""
WebSocket Connection Manager
Manages active WebSocket connections and broadcasts messages
"""

from typing import Dict, List
from fastapi import WebSocket


class ConnectionManager:
    """
    Manages WebSocket connections grouped by symbol and timeframe
    """

    def __init__(self):
        # Key format: "SYMBOL_TIMEFRAME" (e.g., "NIFTY_5m")
        self.active_connections: Dict[str, List[WebSocket]] = {}

    def _get_key(self, symbol: str, timeframe: str) -> str:
        """Generate connection key from symbol and timeframe"""
        return f"{symbol}_{timeframe}"

    async def connect(self, websocket: WebSocket, symbol: str, timeframe: str):
        """
        Accept and register a new WebSocket connection
        """
        await websocket.accept()
        key = self._get_key(symbol, timeframe)

        if key not in self.active_connections:
            self.active_connections[key] = []

        self.active_connections[key].append(websocket)
        print(f"[WS] Client connected to {key}. Total connections: {len(self.active_connections[key])}")

    async def disconnect(self, websocket: WebSocket, symbol: str, timeframe: str):
        """
        Remove a WebSocket connection
        """
        key = self._get_key(symbol, timeframe)

        if key in self.active_connections:
            if websocket in self.active_connections[key]:
                self.active_connections[key].remove(websocket)
                print(f"[WS] Client disconnected from {key}. Remaining: {len(self.active_connections[key])}")

            # Clean up empty lists
            if not self.active_connections[key]:
                del self.active_connections[key]

    async def broadcast(self, symbol: str, timeframe: str, message: dict):
        """
        Broadcast a message to all connections for a specific symbol/timeframe
        """
        key = self._get_key(symbol, timeframe)

        if key not in self.active_connections:
            return

        # Keep track of dead connections to remove
        dead_connections = []

        for connection in self.active_connections[key]:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"[WS] Error sending to client: {e}")
                dead_connections.append(connection)

        # Remove dead connections
        for dead in dead_connections:
            await self.disconnect(dead, symbol, timeframe)

    def get_active_symbols(self) -> List[str]:
        """
        Get list of all active symbol_timeframe combinations
        """
        return list(self.active_connections.keys())

    def get_connection_count(self) -> int:
        """
        Get total number of active connections across all symbols
        """
        return sum(len(conns) for conns in self.active_connections.values())
