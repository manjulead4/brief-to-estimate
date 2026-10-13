from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ProductType = Literal["ring", "pendant", "earrings", "bracelet", "necklace"]
Metal = Literal["gold", "white_gold", "rose_gold", "silver", "platinum"]
StoneType = Literal["diamond", "ruby", "emerald", "sapphire", "cubic_zirconia", "none"]
Finish = Literal["polished", "matte", "hammered", "brushed"]
Purity = Literal["9k", "10k", "14k", "18k", "22k", "24k", "925", "950"]
Confidence = Literal["high", "medium", "low"]

FIELDS = ["product_type", "metal", "purity", "stone_type", "stone_count", "dimension_mm", "finish"]


class Spec(BaseModel):
    """Structured spec. None means 'not stated / unknown' -- never a guess."""
    product_type: Optional[ProductType] = None
    metal: Optional[Metal] = None
    purity: Optional[Purity] = None
    stone_type: Optional[StoneType] = None
    stone_count: Optional[int] = Field(default=None, ge=0, le=1000)
    dimension_mm: Optional[float] = Field(default=None, gt=0, le=2000)
    finish: Optional[Finish] = None


class FieldOut(BaseModel):
    value: Any = None
    confidence: Confidence
    reason: str
    source: str  # brief_keywords | llm | llm+brief_keywords | image_cv | brief+image | none


class Flag(BaseModel):
    field: str
    severity: Literal["info", "warning", "critical"]
    message: str


class Candidate(BaseModel):
    sku: str
    name: str
    score: float
    breakdown: Dict[str, float]
    price_usd: float


class MatchResult(BaseModel):
    auto_selected: Optional[str]
    reason: str
    min_score: float
    coverage: float
    candidates: List[Candidate]


class ImageEstimate(BaseModel):
    status: Literal["ok", "error"]
    count: Optional[int] = None
    confidence: Optional[Confidence] = None
    reason: str = ""
    methods: Dict[str, Any] = {}
    vlm_count: Optional[int] = None


class ExtractionMeta(BaseModel):
    mode: Literal["llm", "llm_repaired", "llm_retry", "rules_fallback", "empty_default"]
    notes: List[str] = []
    provider: Optional[str] = None
    model: Optional[str] = None
    attempts: int = 0
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    extraction: ExtractionMeta
    spec: Dict[str, FieldOut]
    matches: MatchResult
    image_estimate: Optional[ImageEstimate] = None
    needs_review: bool
    flags: List[Flag]
    timings_ms: Dict[str, float]
