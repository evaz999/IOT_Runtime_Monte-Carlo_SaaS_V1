from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Battery Feasibility V1 API")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# TODO: add endpoints that accept JSON configs and call run_coin_cell_feasibility
