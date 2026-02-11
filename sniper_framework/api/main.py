from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.market import router as market_router
from api.websocket.stream import router as ws_router

app = FastAPI(title="Sniper Trading API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router, prefix="/api")
app.include_router(ws_router)
