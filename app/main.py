import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile

from .config import Settings
from .matcher import load_inventory
from .pipeline import analyze
from .schemas import AnalyzeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
settings = Settings()
inventory = load_inventory(settings.inventory_path)
app = FastAPI(title="Brief to Priced Estimate", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": settings.llm_enabled, "provider": settings.provider,
            "model": settings.model if settings.llm_enabled else None, "inventory_rows": len(inventory)}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(brief: str = Form(""), image: Optional[UploadFile] = File(None), use_vlm: bool = Form(False)):
    img = None
    if image is not None and image.filename:
        # read one byte past the limit so an oversized upload is detected without buffering it all
        data = await image.read(settings.max_image_bytes + 1)
        img = (data, image.content_type or "image/jpeg")
    return analyze(brief, img, settings, inventory, use_vlm=use_vlm)
