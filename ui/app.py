"""
ui/app.py
─────────────────────────────────────────────────────────────────────────────
FastAPI web server for the SpotifyCares Support Agent demo.

Endpoints:
  GET  /           → serves the interactive UI
  POST /api/agent  → runs the agent pipeline
  GET  /api/health → health check
  GET  /api/stats  → processed data statistics

Usage:
  uvicorn ui.app:app --reload --port 8000
  python ui/app.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpotifyCares Support Agent",
    description="AI-powered customer support agent demo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Request/Response models ───────────────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str
    conversation_context: str = ""


class AgentResponse(BaseModel):
    intent: str
    intent_confidence: float
    retrieved_evidence: list[dict]
    draft_reply: str
    grounding_score: float
    decision: str
    decision_reason: str
    risk_flags: list[str]
    latency_ms: float


# ── Lazy agent init ───────────────────────────────────────────────────────────

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from src.agent import SpotifySupportAgent
        config_path = ROOT / "config.yaml"
        _agent = SpotifySupportAgent(config_path=config_path)
    return _agent


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "SpotifyCares Support Agent API. See /docs for endpoints."})


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/stats")
async def stats():
    stats_path = ROOT / "data" / "processed" / "data_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {"error": "Run prepare_data.py first to generate statistics."}


@app.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        agent = get_agent()
        result = agent.run(req.message, req.conversation_context)
        return AgentResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not ready. Run prepare_data.py and build_index.py first. ({e})"
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/examples")
async def get_examples():
    """Return example customer messages for the demo UI."""
    return [
        {"label": "Login issue", "message": "I can't log into my Spotify account. My password isn't working."},
        {"label": "Double charge", "message": "I was charged twice for Spotify Premium this month. Please refund!"},
        {"label": "App crash", "message": "The Spotify app keeps crashing every time I try to play a song on my iPhone."},
        {"label": "Song missing", "message": "A song I liked yesterday disappeared from my playlist. Where did it go?"},
        {"label": "Ads on Premium", "message": "Why am I getting ads? I'm paying for Premium!"},
        {"label": "Playback issue", "message": "Spotify stops playing after every song. Shuffle is broken."},
        {"label": "Family plan", "message": "How do I add someone to my family plan?"},
        {"label": "Crossfade", "message": "How do I enable crossfade between songs on desktop?"},
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.app:app", host="0.0.0.0", port=8000, reload=True)
