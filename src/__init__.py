"""Package init — pins every model cache to E:\\cache before any library loads.

Another project on this machine exports TRANSFORMERS_CACHE=E:\\eraya_cache, and
that variable outranks HF_HOME inside transformers. Without this, downloads land
in the wrong project's cache: a 8GB VLM went there before it was caught. Set here
because it must happen before transformers/sentence-transformers are imported,
and every entry point imports `src` first.
"""
import os
from pathlib import Path

CACHE_ROOT = Path(r"E:\cache")

os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "huggingface" / "hub")
os.environ.pop("TRANSFORMERS_CACHE", None)  # deprecated and hijacks HF_HOME
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
