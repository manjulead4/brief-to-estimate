"""All configuration comes from environment variables. Nothing secret lives in the repo."""
import os

DEFAULT_MODELS = {
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "gemini": ("gemini-3.5-flash", "gemini-3.1-flash-lite"),
}
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class Settings:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
        default_primary, default_fallback = DEFAULT_MODELS.get(self.provider, ("", ""))
        self.model = os.getenv("LLM_MODEL") or default_primary
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL") or default_fallback
        self.api_key = os.getenv(KEY_VARS.get(self.provider, ""), "")
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_S", "30"))
        self.price_in = float(os.getenv("LLM_PRICE_IN_PER_MTOK", "0"))
        self.price_out = float(os.getenv("LLM_PRICE_OUT_PER_MTOK", "0"))
        self.max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
        self.inventory_path = os.getenv("INVENTORY_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "inventory.csv"))

    @property
    def llm_enabled(self) -> bool:
        return self.provider in KEY_VARS and bool(self.api_key) and bool(self.model)
