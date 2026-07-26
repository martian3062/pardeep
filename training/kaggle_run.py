"""Drive Kaggle's free GPU from here: upload data, push the notebook, fetch the adapter.

Kaggle gives 30 GPU-hours a week at no cost, which is enough to keep the laptop
twin (Qwen3-4B) trained without touching the paid/shared L4 that the 24B needs.

    python -m training.kaggle_run push     # upload dataset + start training
    python -m training.kaggle_run status   # how is it going
    python -m training.kaggle_run fetch    # download the trained adapter

Credentials come from .env (KAGGLE_USERNAME / KAGGLE_KEY). The dataset is
created PRIVATE — it contains his own conversations.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATASET_SLUG = "pardeep-self-sft"
# one kernel per candidate base model so the comparison runs can coexist
BASE_MODEL = os.environ.get("TWIN_BASE_MODEL", "qwen4b")
# Kaggle rejects a push while the previous kernel of that slug is still running
# (409). A suffix gives a fresh slug instead of waiting for it to drain.
_SUFFIX = os.environ.get("KERNEL_SUFFIX", "")
KERNEL_SLUG = f"pardeep-self-{BASE_MODEL}{'-' + _SUFFIX if _SUFFIX else ''}"
UPLOAD_DIR = REPO_ROOT / "data" / "kaggle_upload"
KERNEL_DIR = REPO_ROOT / "data" / "kaggle_kernel"
OUTPUT_DIR = REPO_ROOT / "data" / "models" / "qwen4b"


def _load_credentials(account: str = "") -> None:
    """Kaggle's current auth is a single KGAT_ token in KAGGLE_API_TOKEN (or
    ~/.kaggle/access_token). The legacy username+key pair still works but a
    guessed username fails with a misleading 401, so prefer the token.

    `account="alt"` selects the second Kaggle account, which doubles the free
    GPU allowance to 60 h/week and gives somewhere to retry when one errors.
    """
    if account:
        suffix = f"_{account.upper()}"
        token = os.environ.get(f"KAGGLE_API_TOKEN{suffix}")
        if not token:
            raise SystemExit(f"KAGGLE_API_TOKEN{suffix} missing from .env")
        os.environ["KAGGLE_API_TOKEN"] = token
        os.environ["KAGGLE_USERNAME"] = os.environ.get(f"KAGGLE_USERNAME{suffix}", "")
        print(f"using account: {os.environ['KAGGLE_USERNAME']}")
        return
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text(encoding="utf-8").strip()
        return
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.is_file():
        creds = json.loads(legacy.read_text(encoding="utf-8"))
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        return
    raise SystemExit(
        "No Kaggle credentials.\n"
        "  Kaggle -> Settings -> API -> Create New Token, then either set\n"
        "  KAGGLE_API_TOKEN in .env or save it to ~/.kaggle/access_token"
    )


def api(account: str = ""):
    _load_credentials(account)
    from kaggle.api.kaggle_api_extended import KaggleApi

    a = KaggleApi()
    a.authenticate()
    return a


_DETECTED: dict[str, str] = {}


def user(a=None) -> str:
    """The account the token actually belongs to.

    A hand-typed username silently breaks reads: pushes still succeed because
    Kaggle files them under the token's real owner, but every status/fetch then
    queries a namespace that does not exist and reports a permission error.
    Ask the API who we are instead.
    """
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if token in _DETECTED:
        return _DETECTED[token]
    if a is not None:
        try:
            owned = a.kernels_list(mine=True, page_size=5)
            refs = [k.ref for k in owned]
            if refs:
                name = refs[0].split("/")[0]
                _DETECTED[token] = name
                if name != os.environ.get("KAGGLE_USERNAME"):
                    print(f"note: token belongs to '{name}', not "
                          f"'{os.environ.get('KAGGLE_USERNAME')}' — using '{name}'")
                return name
        except Exception:
            pass  # brand-new account with no kernels yet
    return os.environ.get("KAGGLE_USERNAME", "")


def push_dataset(a) -> str:
    """Create the dataset, or add a version if it already exists."""
    ref = f"{user(a)}/{DATASET_SLUG}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("sft_train.jsonl", "sft_eval.jsonl"):
        src = REPO_ROOT / "data" / "datasets" / name
        if src.exists():
            shutil.copy2(src, UPLOAD_DIR / name)

    (UPLOAD_DIR / "dataset-metadata.json").write_text(
        json.dumps({"title": "pardeep self sft", "id": ref, "licenses": [{"name": "other"}]}),
        encoding="utf-8",
    )
    existing = {d.ref for d in a.dataset_list(user=user(a), search=DATASET_SLUG)}
    if ref in existing:
        a.dataset_create_version(str(UPLOAD_DIR), version_notes="refresh", dir_mode="zip")
        print(f"dataset updated: {ref}")
    else:
        a.dataset_create_new(str(UPLOAD_DIR), public=False, dir_mode="zip")
        print(f"dataset created (private): {ref}")
    return ref


def push_kernel(a, dataset_ref: str) -> str:
    ref = f"{user(a)}/{KERNEL_SLUG}"
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    # The kernel runs in a fresh environment, so TWIN_BASE_MODEL set here never
    # reaches it — bake the choice into the uploaded script instead. Without
    # this every kernel silently trained the default model regardless of slug.
    source = (REPO_ROOT / "training" / "kaggle_train_qwen4b.py").read_text(encoding="utf-8")
    header = f'import os\nos.environ["TWIN_BASE_MODEL"] = "{BASE_MODEL}"\n'
    (KERNEL_DIR / "script.py").write_text(header + source, encoding="utf-8")
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": ref,
                # title must slugify to the id or Kaggle warns and may misroute
                "title": KERNEL_SLUG.replace("-", " "),
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                # Kaggle offers "T4 x2" and "P100"; there is no single-T4 option,
                # so "nvidiaTeslaT4" is not a valid value and is silently ignored
                # — which lands the run on a P100 (compute 6.0) that modern
                # PyTorch has no kernels for. The script uses one GPU regardless
                # (CUDA_VISIBLE_DEVICES=0); this is purely about getting Turing.
                "accelerator": "nvidiaTeslaT4x2",
                "enable_internet": True,  # needs to pull the base model
                "dataset_sources": [dataset_ref],
                "competition_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )
    a.kernels_push(str(KERNEL_DIR))
    print(f"notebook pushed and queued on GPU: https://kaggle.com/code/{ref}")
    return ref


def status(a) -> str:
    ref = f"{user(a)}/{KERNEL_SLUG}"
    st = a.kernels_status(ref)
    state = getattr(st, "status", st)
    print(f"{ref}: {state}")
    if getattr(st, "failureMessage", None):
        print(f"  failure: {st.failureMessage}")
    return str(state)


def fetch(a) -> None:
    ref = f"{user(a)}/{KERNEL_SLUG}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a.kernels_output(ref, path=str(OUTPUT_DIR))
    print(f"artifacts -> {OUTPUT_DIR}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    account = sys.argv[2] if len(sys.argv) > 2 else ""  # "" = primary, "alt" = second
    a = api(account)
    if cmd == "push":
        push_kernel(a, push_dataset(a))
        print("watch with: python -m training.kaggle_run status")
    elif cmd == "status":
        status(a)
    elif cmd == "wait":
        while (s := status(a)).lower() in {"running", "queued"}:
            time.sleep(60)
    elif cmd == "fetch":
        fetch(a)
    else:
        raise SystemExit(f"unknown command {cmd!r} (push | status | wait | fetch)")


if __name__ == "__main__":
    main()
