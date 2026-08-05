#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 1 — lncRNA–RBP inference

Command-line example:
python skill1_lncRNA_RBP_inference.py \
  --gene_of_interest "PRKAG2-AS1" \
  --user_query "I want to know the binding protein of PRKAG2-AS1 to inference its potential function in liver metabolism" \
  --reasoning_function "Liver metabolism (Fasting & Re-feeding)" \
  --binding_database "NPInter,starBase,RNAInter" \
  --literature_searching "Similarity,PubMed" \
  --external_information "Kaiyuan_RNApulldown_M11_Reasoning.txt (description: RNA-pulldown result for PRKAG2-AS1 mouse conserved LncRNA )" \
  --output_file_name "PRKAG2_AS1_skill1_01" \
  --quotauser "quota001"
"""

# ============================================================================
# 1) PARAMETERS & CONSTANTS (SET FIRST)
#    NOTE: Clients and derived paths are instantiated after imports.
# ============================================================================
from pathlib import Path
import os, json
def _load_profile(profile_path: str | None = None) -> dict:
    path = Path(profile_path or os.getenv("PROFILE_JSON", Path(__file__).with_name("profile.json")))
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[Config] 未找到配置文件: {path}\n可设置环境变量 PROFILE_JSON 指定路径。")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[Config] 解析 {path} 失败：{e}")

    # 环境变量覆盖（方便在服务器/HPC上注入）
    for k in list(cfg.keys()):
        if os.getenv(k):
            cfg[k] = os.getenv(k)

    return cfg

_CFG = _load_profile()

# --- LLM provider selection -------------------------------------------------
# A non-empty OPENAI_API_KEY takes priority. If it is absent, the original
# Azure configuration is used unchanged.
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or _CFG.get("OPENAI_API_KEY") or "").strip()
OPENAI_GPT_MODEL = (os.getenv("OPENAI_GPT_MODEL") or _CFG.get("OPENAI_GPT_MODEL") or "gpt-4o").strip()
OPENAI_REASONING_MODEL = (
    os.getenv("OPENAI_REASONING_MODEL")
    or _CFG.get("OPENAI_REASONING_MODEL")
    or "o4-mini"
).strip()
USE_OPENAI = bool(OPENAI_API_KEY)

# --- Original Azure OpenAI / Inference endpoints & keys (fallback) ---------
AZURE_GPT4O_ENDPOINT = (_CFG.get("AZURE_GPT4O_ENDPOINT") or "").strip()
AZURE_GPT4O_API_KEY = (_CFG.get("AZURE_GPT4O_API_KEY") or "").strip()
AZURE_GPT4O_API_VER  = "2024-08-01-preview"

AZURE_O4MINI_ENDPOINT = (_CFG.get("AZURE_O4MINI_ENDPOINT") or "").strip()
AZURE_O4MINI_API_KEY = (_CFG.get("AZURE_O4MINI_API_KEY") or "").strip()
AZURE_O4MINI_API_VERSION = "2025-01-01-preview"

# --- Google Web Search via Vertex AI Discovery Engine (searchLite; replaces Google CSE) ---
# Read from profile.json (or env var override via _load_profile()).
VERTEX_PROJECT = (_CFG.get("VERTEX_PROJECT") or "").strip()

# Backward compatible: allow either VERTEX_* or legacy GOOGLE_* keys in profile.json
VERTEX_ENGINE_ID = (_CFG.get("VERTEX_ENGINE_ID") or "").strip()
VERTEX_API_KEY   = (_CFG.get("VERTEX_API_KEY") or "").strip()

# Optional (defaults match your sample)
VERTEX_LOCATION       = (_CFG.get("VERTEX_LOCATION") or "global").strip()
VERTEX_COLLECTION     = (_CFG.get("VERTEX_COLLECTION") or "default_collection").strip()
VERTEX_SERVING_CONFIG = (_CFG.get("VERTEX_SERVING_CONFIG") or "default_search").strip()

# Keep original variable names used in the rest of the script
GOOGLE_API_KEY = VERTEX_API_KEY
GOOGLE_CX      = VERTEX_ENGINE_ID


# --- I/O and external info ---
EXTERNAL_BASE_DIR = "./temp"

# --- File/Folder naming (derived absolute paths are computed after imports) ---
DATA_SUBDIR_NAME   = "database"
INDEX_FILENAME     = "faiss_index.bin"
META_FILENAME      = "meta.pkl"

# --- Verbosity flags (declared twice in original; preserved; final value=True) ---
VERBOSE = False   # original first declaration

# ============================================================================
# 2) IMPORTS (STANDARD → THIRD-PARTY → LOCAL)
# ============================================================================

# ----- Standard Library -----
import sys
import re
import gc
import csv
import json
import argparse
import datetime
import random
import os
import time
import pickle
import hashlib
import gzip
from pathlib import Path
import threading
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

# Aliases used by AgentProgress utility
import json as _json
import sys as _sys
import time as _time

# ----- Third‑Party Packages -----
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI, OpenAIError, AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# NEW: pandas + Excel styling helpers (with graceful fallbacks)
import pandas as pd
try:
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    _HAS_OPENPYXL = True
except Exception:
    _HAS_OPENPYXL = False

# ----- Local Modules -----
from local_db_handler import init_local_db, query_database_local

# --- Optional tiny delay per step in snapshot replay (seconds), same as Skill 2 semantics ---
SNAPSHOT_REPLAY_DELAY = float(
    os.getenv(
        "SKILL1_SNAPSHOT_REPLAY_DELAY",
        os.getenv("P1_SNAPSHOT_REPLAY_DELAY", os.getenv("P2_SNAPSHOT_REPLAY_DELAY", "0.5")),
    )
)

def _maybe_sleep():
    """Sleep a tiny bit during snapshot replay to smooth UI updates."""
    if SNAPSHOT_REPLAY_DELAY > 0:
        time.sleep(SNAPSHOT_REPLAY_DELAY)
# ============================================================================
# 3) DERIVED PATHS & RUNTIME OBJECTS (AFTER IMPORTS)
# ============================================================================

# Paths derived from the current file location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, DATA_SUBDIR_NAME)
INDEX_PATH = os.path.join(data_dir, INDEX_FILENAME)
META_PATH = os.path.join(data_dir, META_FILENAME)

# Embedding model for FAISS queries
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Provider-aware clients. The adapter preserves the original ``client1.complete``
# call surface so the downstream reasoning and report code stays unchanged.
class _OpenAICompleteAdapter:
    def __init__(self, client: OpenAI, default_model: str):
        self._client = client
        self._default_model = default_model

    def complete(self, *, model: str = "", messages, **kwargs):
        return self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,
            **kwargs,
        )


if USE_OPENAI:
    client1 = _OpenAICompleteAdapter(OpenAI(api_key=OPENAI_API_KEY), OPENAI_GPT_MODEL)
    client2 = OpenAI(api_key=OPENAI_API_KEY)
    GPT_MODEL = OPENAI_GPT_MODEL
    REASONING_MODEL = OPENAI_REASONING_MODEL
    LLM_PROVIDER = "openai"
else:
    missing = [
        name for name, value in (
            ("AZURE_GPT4O_ENDPOINT", AZURE_GPT4O_ENDPOINT),
            ("AZURE_GPT4O_API_KEY", AZURE_GPT4O_API_KEY),
            ("AZURE_O4MINI_ENDPOINT", AZURE_O4MINI_ENDPOINT),
            ("AZURE_O4MINI_API_KEY", AZURE_O4MINI_API_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "[Config] Set OPENAI_API_KEY, or provide the Azure fallback values: "
            + ", ".join(missing)
        )
    client1 = ChatCompletionsClient(
        endpoint=AZURE_GPT4O_ENDPOINT,
        credential=AzureKeyCredential(AZURE_GPT4O_API_KEY),
    )
    client2 = AzureOpenAI(
        azure_endpoint=AZURE_O4MINI_ENDPOINT,
        api_key=AZURE_O4MINI_API_KEY,
        api_version=AZURE_O4MINI_API_VERSION,
    )
    GPT_MODEL = "gpt-4o"
    REASONING_MODEL = "o4-mini"
    LLM_PROVIDER = "azure"

# ============================================================================
# 4) UTILITIES — LOGGING, DATE PARSING, PROGRESS EMITTER
# ============================================================================

def log_debug(msg: str):
    """Conditional debug logger based on VERBOSE flag."""
    if VERBOSE:
        print(f"[DEBUG] {msg}")

def parse_date(date_str: str):
    """Parse simple date strings to a date object; fallback to 1900-01-01."""
    if not date_str:
        return datetime.date(1900, 1, 1)
    for fmt in ("%Y/%m/%d", "%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return datetime.date(1900, 1, 1)

# --- Duplicate definitions preserved intentionally (original file had them twice) ---

def log_debug(msg: str):
    """(Duplicate) Conditional debug logger based on VERBOSE flag (preserved)."""
    if VERBOSE:
        print(f"[DEBUG] {msg}")

def parse_date(date_str: str):
    """(Duplicate) Parse simple date strings to a date object; fallback to 1900-01-01 (preserved)."""
    if not date_str:
        return datetime.date(1900, 1, 1)
    for fmt in ("%Y/%m/%d", "%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return datetime.date(1900, 1, 1)

# --- Unified progress emitter (unchanged logic) ---

def log(msg: str):
    """Simple stderr logger (compatible with prior usage)."""
    print(f"[LOG] {msg}", file=sys.stderr, flush=True)

class AgentProgress:
    """
    Emit compact, GUI-parsable progress events to stdout.

    The GUI listens for lines that start with '@@PROGRESS ' followed by a JSON object:
      {"run":"Skill1","kind":"start|update|done|error","stage":"search|filter|gpt|html|...",
       "step":1,"total":6,"perc":17,"t":1.23,"title":"Searching PMC",
       "detail":"term 1/3: TP53 experiment binding", "extra":{}}
    """
    def __init__(self, run: str, total_steps: int):
        self.run = run
        self.total = max(1, int(total_steps))
        self.step = 0
        self.t0 = _time.time()

    def _emit(self, payload: dict):
        payload.setdefault("run", self.run)
        payload.setdefault("t", round(_time.time() - self.t0, 2))
        s = "@@PROGRESS " + _json.dumps(payload, ensure_ascii=False)
        print(s, flush=True)  # stdout → parsed by GUI

    def start_step(self, title: str, stage: str, detail: str = ""):
        self.step += 1
        self._emit({
            "kind": "start",
            "stage": stage,
            "step": self.step,
            "total": self.total,
            "perc": int((self.step-1)/self.total*100),
            "title": title,
            "detail": detail
        })

    def update(self, msg: str, stage: str = "", perc: int | None = None, **extra):
        self._emit({
            "kind": "update", "stage": stage or "",
            "step": self.step, "total": self.total,
            "perc": int(self.step/self.total*100) if perc is None else int(perc),
            "title": msg, "detail": extra.get("detail",""), "extra": extra
        })

    def done(self, msg: str = "", **extra):
        self._emit({
            "kind": "done", "stage": "done",
            "step": self.step, "total": self.total, "perc": int(self.step/self.total*100),
            "title": msg, "detail": extra.get("detail",""), "extra": extra
        })

    def error(self, err: Exception | str, **extra):
        self._emit({
            "kind": "error", "stage": "error",
            "step": self.step, "total": self.total,
            "perc": int(self.step/self.total*100),
            "title": "Error",
            "detail": str(err), "extra": extra
        })

# ============================================================================
# 5) FAISS CONFIGURATION & HELPERS
# ============================================================================

# Singleton cache to avoid re-loading large FAISS index repeatedly
_faiss_index = None
_faiss_meta  = None
_faiss_lock  = threading.Lock()

def _get_faiss():
    """Load and cache FAISS index + metadata once."""
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        with _faiss_lock:
            if _faiss_index is None:  # second guard
                _faiss_index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "rb") as f:
                    _faiss_meta = pickle.load(f)
    return _faiss_index, _faiss_meta

def retrieve_top_docs(query: str, entity: str, top_k: int = 2000) -> List[Dict[str, Any]]:
    """Vector search → filter by `entity` substring → return per-PMID hits."""
    log_debug(f"FAISS search • query='{query}' • entity='{entity}' • top_k={top_k}")

    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        log_debug("[ERROR] FAISS index or meta file missing")
        return []

    index, meta = _get_faiss()
    emb = embedding_model.encode([query]).astype(np.float32)
    D, I = index.search(emb, top_k)

    raw_hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[idx]
        raw_hits.append({
            "pmid"    : m["pmid"],
            "pmc_url" : m.get("pmc_url") or m.get("link", ""),
            "link"    : m.get("link", ""),
            "title"   : m.get("title", ""),
            "journal" : m.get("journal", ""),
            "date"    : m.get("date", ""),
            "text"    : m.get("text", ""),
            "distance": float(dist)
        })

    if entity:
        entity_lc = entity.lower()
        raw_hits = [h for h in raw_hits if entity_lc in h["text"].lower()]

    hits_by_pm = defaultdict(list)
    for h in raw_hits:
        hits_by_pm[h["pmid"]].append(h)

    docs = []
    for pmid, hits in hits_by_pm.items():
        rep = hits[0].copy()
        rep["hit_count"] = len(hits)
        docs.append(rep)

    docs.sort(key=lambda d: -d["hit_count"])
    return docs[:3]

def aggregate_and_sort(docs):
    """Aggregate duplicate PMIDs and sort by count then by latest date."""
    stats = {}
    for d in docs:
        pmid = d["pmid"]
        dt = parse_date(d.get("date", ""))
        s = stats.setdefault(pmid, {"count": 0, "latest": dt, **d})
        s["count"] += 1
        if dt > s["latest"]:
            s["latest"] = dt
    out = list(stats.values())
    out.sort(key=lambda x: (-x["count"], -x["latest"].toordinal()))
    log_debug(f"Aggregation complete — unique PMIDs: {len(out)}")
    return out

def load_pmid_pmc_dict(path):
    """Load a simple PMID→PMC metadata map from a TSV-like text file."""
    if not os.path.exists(path):
        log_debug(f"[WARN] PMC meta file missing: {path}")
        return {}
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pmid, pmc, date, journal, title, full = line.rstrip("\n").split("\t", 5)
            d[pmid] = {"pmc_url": pmc, "date": date, "journal": journal, "title": title, "full_text": full}
    log_debug(f"Loaded {len(d)} PMID⇢PMC mappings")
    return d

# ============================================================================
# 6) DATABASE ACCESS & TRANSFORM HELPERS
# ============================================================================

def filter_ensg_names(name_list):
    """Filter out Ensembl IDs (starting with ENSG...)."""
    filtered = []
    for name in name_list:
        if re.match(r'^ENSG\d+', name):
            continue
        filtered.append(name)
    return filtered

def query_database(gene, table_name, search_conditions):
    """
    Use local file-based system (via local_db_handler) as a MySQL replacement.
    """
    columns, results = query_database_local(gene, table_name, search_conditions)
    return columns, results

def process_NPInter(gene, data):
    """Process NPInter table to infer counterpart type and format entries."""
    gene_lower = gene.lower()
    found_in_rbp = any(gene_lower in row.get("RBP Name", "").lower() for row in data)
    found_in_lnc = any(gene_lower in row.get("LncRNA Name", "").lower() for row in data)
    result = set()
    tag = ""
    if found_in_rbp:
        tag = "lncRNA"
        for row in data:
            if "LncRNA Name" in row:
                name = row["LncRNA Name"]
                cell_line = row.get("Cell line", None)
                formatted = f"{name}_{cell_line}(cell line)" if cell_line else name
                result.add(formatted)
        result = filter_ensg_names(list(result))
    elif found_in_lnc:
        tag = "RBP"
        for row in data:
            if "RBP Name" in row:
                name = row["RBP Name"]
                cell_line = row.get("Cell line", None)
                formatted = f"{name}_{cell_line}(cell line)" if cell_line else name
                result.add(formatted)
    return {"tag": tag, "result": list(result)}

def process_RNAInter(gene, data):
    """Process RNAInter table to infer counterpart type and format entries."""
    gene_lower = gene.lower()
    found_in_lnc = any(gene_lower in row.get("LncRNA Name", "").lower() for row in data)
    found_in_rbp = any(gene_lower in row.get("RBP Name", "").lower() for row in data)
    result = set()
    tag = ""
    if found_in_lnc:
        tag = "RBP"
        for row in data:
            if "RBP Name" in row:
                result.add(row["RBP Name"])
    elif found_in_rbp:
        tag = "lncRNA"
        for row in data:
            if "LncRNA Name" in row:
                result.add(row["LncRNA Name"])
        result = filter_ensg_names(list(result))
    return {"tag": tag, "result": list(result)}

def process_RNAInter_mRNA(gene, data):
    gene_lower = gene.lower()
    found_in_mrna = any(gene_lower in row.get("mRNA Name", "").lower() for row in data)
    found_in_rbp  = any(gene_lower in row.get("RBP Name",  "").lower() for row in data)

    result = set()
    tag = ""

    if found_in_mrna:
        tag = "RBP"
        for row in data:
            if "RBP Name" in row:
                result.add(row["RBP Name"])

    elif found_in_rbp:
        tag = "mRNA"
        for row in data:
            if "mRNA Name" in row:
                result.add(row["mRNA Name"])

    return {"tag": tag, "result": list(result)}

def process_starBase(gene, data):
    """Process starBase table to infer counterpart type and format entries."""
    gene_lower = gene.lower()
    found_in_rbp = any(gene_lower in row.get("RBP Name", "").lower() for row in data)
    found_in_lnc = any(gene_lower in row.get("LncRNA Name", "").lower() for row in data)
    result = set()
    tag = ""
    if found_in_rbp:
        tag = "lncRNA"
        for row in data:
            if "LncRNA Name" in row:
                name = row["LncRNA Name"]
                cell_line = row.get("cellline/tissue", None)
                formatted = f"{name}_{cell_line}(cell line)" if cell_line else name
                result.add(formatted)
        result = filter_ensg_names(list(result))
    elif found_in_lnc:
        tag = "RBP"
        for row in data:
            if "RBP Name" in row:
                name = row["RBP Name"]
                cell_line = row.get("cellline/tissue", None)
                formatted = f"{name}_{cell_line}(cell line)" if cell_line else name
                result.add(formatted)
    return {"tag": tag, "result": list(result)}

def process_GTEx_Tissue(db_data):
    """
    Reorder GTEx tissue expression columns:
    1) Collect numeric columns and compute their mean.
    2) Sort numeric columns by descending mean.
    3) Place 'Name' and 'Description' first.
    """
    if not db_data:
        return {"tag": "Tissue", "data": [], "columns": []}

    numeric_cols_map = {}
    for row in db_data:
        for key, val in row.items():
            if key not in ["Name", "Description"]:
                try:
                    val_float = float(val)
                except:
                    val_float = 0.0
                numeric_cols_map.setdefault(key, []).append(val_float)

    col_avg = {}
    for col, values in numeric_cols_map.items():
        col_avg[col] = (sum(values) / len(values)) if values else 0.0

    sorted_numeric_cols = sorted(col_avg.keys(), key=lambda c: col_avg[c], reverse=True)
    final_columns = ["Name", "Description"] + sorted_numeric_cols

    new_data = []
    for row in db_data:
        new_row = {}
        for col in final_columns:
            new_row[col] = row.get(col, "")
        new_data.append(new_row)

    return {"tag": "Tissue", "data": new_data, "columns": final_columns}

def deduplicate_category(result_dict, order):
    """
    Deduplicate by preserving first appearance according to `order` priority.
    Works with either strings or dict items (uses 'name' if dict).
    """
    seen = set()
    for db in order:
        if db in result_dict:
            deduped = []
            for item in result_dict[db]["result"]:
                if isinstance(item, dict):
                    identifier = str(item.get("name"))
                else:
                    identifier = str(item)
                if identifier not in seen:
                    deduped.append(item)
                    seen.add(identifier)
            result_dict[db]["result"] = deduped
    return result_dict

# ============================================================================
# 7) EXTERNAL FILE PROCESSING
# ============================================================================

def process_external_information(ext_param: str) -> str:
    """
    Accept items like:
      /abs/path/to/file1.txt (description: RNA-pulldown result...) | /abs/other.tsv (description: ...)
    or relative paths; fall back to EXTERNAL_BASE_DIR if needed.
    Returns concatenated text blocks suitable for prompt usage.
    """
    if not ext_param:
        return "No File Input"

    blocks = []
    items = [x.strip() for x in ext_param.split("|") if x.strip()]
    for item in items:
        # Robustly split "[path] (description: ...)" from the right,
        # so parentheses inside the path won't break parsing.
        desc = "No description"
        m = re.search(r'\s*\((?:description:)\s*(.+)\)\s*$', item, flags=re.I)
        if m:
            desc = m.group(1).strip()
            fname = item[:m.start()].strip()
        else:
            fname = item.strip()

        # Expand ~ and env vars; strip quotes if any
        fname = os.path.expanduser(os.path.expandvars(fname))
        if (fname.startswith('"') and fname.endswith('"')) or (fname.startswith("'") and fname.endswith("'")):
            fname = fname[1:-1]

        # Candidate locations: absolute as-is; CWD; ./temp/<basename>
        candidates = [
            fname,
            os.path.join(os.getcwd(), fname),
            os.path.join(EXTERNAL_BASE_DIR, os.path.basename(fname)),
        ]
        fpath = next((c for c in candidates if os.path.isfile(c)), None)

        if not fpath:
            blocks.append(f"File: {fname}\nDescription: {desc}\n[ERROR] File not found.\n")
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            content = f"[ERROR] Can not read file. {e}"

        blocks.append(f"File: {fpath}\nDescription: {desc}\n{content}")

    return "\n\n".join(blocks) if blocks else "No File Input"

# ============================================================================
# 8) LLM REASONING
# ============================================================================

# 7.5) SNAPSHOT SYSTEM (content-aware and multi-file)
# ============================================================================
_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
_SNAPSHOT_VERSION = "P1SNAPv2"  # kept for snapshot-format compatibility
os.makedirs(_SNAPSHOT_DIR, exist_ok=True)

def _normalize_text(s: str) -> str:
    """Collapse whitespace/newlines; trims ends (used for stable keys)."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def _sha256_hex_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data or b"")
    return h.hexdigest()

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _resolve_ext_file_path(spec_path: str) -> str | None:
    """
    Resolve external file path robustly (absolute, CWD, EXTERNAL_BASE_DIR,
    alongside this script). Returns the first existing path or None.
    """
    if not spec_path:
        return None
    # Expand user/env
    p0 = os.path.expanduser(os.path.expandvars(spec_path.strip().strip('"').strip("'")))
    # Candidate locations
    cand = [
        p0,
        os.path.join(os.getcwd(), p0) if not os.path.isabs(p0) else p0,
        os.path.join(EXTERNAL_BASE_DIR, os.path.basename(p0)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(p0)),
    ]
    for p in cand:
        try:
            if p and os.path.isfile(p):
                return p
        except Exception:
            continue
    return None

def _parse_external_specs(ext_param: str) -> list[dict]:
    """
    Parse 'external_information' into a list of {'name','desc','path_raw'}.
    Supports multiple items joined by '|' and optional '(description: ...)' suffix.
    """
    out = []
    if not ext_param:
        return out
    items = [x.strip() for x in ext_param.split("|") if x.strip()]
    for item in items:
        desc = "No description"
        m = re.search(r'\s*\((?:description:)\s*(.+)\)\s*$', item, flags=re.I)
        if m:
            desc = m.group(1).strip()
            fname = item[:m.start()].strip()
        else:
            fname = item.strip()
        out.append({
            "name": os.path.basename(fname.strip().strip('"').strip("'")),
            "desc": desc,
            "path_raw": fname
        })
    return out

def _external_file_records(ext_param: str) -> tuple[list[dict], list[str]]:
    """
    Return (records, sha_list_sorted). Each record:
      {'name','resolved','exists','size','sha256'}
    sha_list_sorted contains only SHA256 strings for existing files, sorted.
    """
    specs = _parse_external_specs(ext_param)
    records = []
    sha_list = []
    for spec in specs:
        resolved = _resolve_ext_file_path(spec["path_raw"])
        rec = {
            "name": spec["name"],
            "resolved": resolved or "",
            "exists": bool(resolved and os.path.isfile(resolved)),
            "size": 0,
            "sha256": ""
        }
        if rec["exists"]:
            try:
                rec["size"] = os.path.getsize(resolved)
                rec["sha256"] = _sha256_of_file(resolved)
                sha_list.append(rec["sha256"])
            except Exception:
                pass
        records.append(rec)
    sha_list_sorted = sorted(sha_list)
    return records, sha_list_sorted

def _canonical_inputs_p1(args) -> dict:
    """
    Build a canonical, content-aware inputs dict.
    KEY fields (used for snapshot matching / key derivation):
      • gene_of_interest (uppercased)
      • user_query_sha256 (normalized text)
      • reasoning_function_sha256 (normalized text)
      • binding_database_set (sorted, lowercase)
      • literature_searching_set (sorted, lowercase)
      • external_files_sha256_list (sorted) — content-based
    AUX fields (for transparency/debug; not used for key derivation):
      • user_query_norm, reasoning_function_norm
      • external_files_records (name/resolved/exists/size/sha256)
    """
    gene = (args.gene_of_interest or "").strip().upper()
    uq_norm = _normalize_text(args.user_query or "")
    rf_norm = _normalize_text(args.reasoning_function or "")

    bd_set = sorted({x.strip().lower() for x in (args.binding_database or "").split(",") if x.strip()})
    lit_set = sorted({x.strip().lower() for x in (args.literature_searching or "").split(",") if x.strip()})

    ext_records, ext_sha_list = _external_file_records(getattr(args, "external_information", "") or "")

    canon = {
        # KEY fields
        "gene": gene,
        "user_query_sha256": _sha256_hex_bytes(uq_norm.encode("utf-8")) if uq_norm else "EMPTY",
        "reasoning_function_sha256": _sha256_hex_bytes(rf_norm.encode("utf-8")) if rf_norm else "EMPTY",
        "binding_database_set": bd_set,
        "literature_searching_set": lit_set,
        "external_files_sha256_list": ext_sha_list,
        # AUX fields
        "user_query_norm": uq_norm,
        "reasoning_function_norm": rf_norm,
        "external_files_records": ext_records,
        "external_files_count": len(ext_records),
    }
    return canon

def _snapshot_key_p1(canon_inputs: dict) -> str:
    """
    Derive a short stable key from the KEY fields only.
    """
    payload = {
        "v": _SNAPSHOT_VERSION,
        "gene": canon_inputs.get("gene",""),
        "uq_sha": canon_inputs.get("user_query_sha256",""),
        "rf_sha": canon_inputs.get("reasoning_function_sha256",""),
        "bd": canon_inputs.get("binding_database_set", []),
        "lit": canon_inputs.get("literature_searching_set", []),
        "ext": canon_inputs.get("external_files_sha256_list", []),
    }
    j = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:16]

def _slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:maxlen] or "na"

def get_snapshot_path_p1_from_canon(canon_inputs: dict) -> str:
    key = _snapshot_key_p1(canon_inputs)
    fname = (
        f"{_SNAPSHOT_VERSION}_{key}"
        f"__gene={_slugify(canon_inputs.get('gene',''), 24)}"
        f"__reason={_slugify(canon_inputs.get('reasoning_function_norm',''), 24)}"
        f"__lit={_slugify(','.join(canon_inputs.get('literature_searching_set', [])), 16)}.json.gz"
    )
    return os.path.join(_SNAPSHOT_DIR, fname)

def get_snapshot_path_p1(args) -> str:
    ci = _canonical_inputs_p1(args)
    return get_snapshot_path_p1_from_canon(ci)

def save_snapshot_p1(path: str, args, html_content: str, metadata: dict | None = None) -> None:
    ci = _canonical_inputs_p1(args)
    snap = {
        "snapshot_version": _SNAPSHOT_VERSION,
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {
            # Only include fields necessary for strict match + transparency
            "gene": ci.get("gene",""),
            "user_query_sha256": ci.get("user_query_sha256",""),
            "user_query_norm": ci.get("user_query_norm",""),
            "reasoning_function_sha256": ci.get("reasoning_function_sha256",""),
            "reasoning_function_norm": ci.get("reasoning_function_norm",""),
            "binding_database_set": ci.get("binding_database_set", []),
            "literature_searching_set": ci.get("literature_searching_set", []),
            "external_files_sha256_list": ci.get("external_files_sha256_list", []),
            "external_files_records": ci.get("external_files_records", []),
        },
        "artifacts": {"html": html_content or ""},
        "metadata": metadata or {}
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)

def load_snapshot_p1(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("artifacts", {"html": ""})
    data.setdefault("params", {})
    data.setdefault("metadata", {})
    return data

def _params_match_strict_p1(params: dict, canon: dict) -> bool:
    """Strict content-aware match using KEY fields only (ignore filenames/order)."""
    if not params:
        return False
    try:
        return (
            (params.get("gene","").strip().upper() == canon.get("gene","")) and
            (params.get("user_query_sha256","") == canon.get("user_query_sha256","")) and
            (params.get("reasoning_function_sha256","") == canon.get("reasoning_function_sha256","")) and
            (sorted(params.get("binding_database_set", [])) == sorted(canon.get("binding_database_set", []))) and
            (sorted(params.get("literature_searching_set", [])) == sorted(canon.get("literature_searching_set", []))) and
            (sorted(params.get("external_files_sha256_list", [])) == sorted(canon.get("external_files_sha256_list", [])))
        )
    except Exception:
        return False

def _locate_matching_snapshot_p1(canon: dict) -> tuple[str | None, str]:
    """
    Try to locate an existing snapshot file whose stored params strictly match
    the current canonical inputs. Returns (path_or_None, status), where status
    is one of {'strict','none'}.
    """
    # Direct computed path
    direct = get_snapshot_path_p1_from_canon(canon)
    if os.path.exists(direct):
        return direct, "strict"

    # Otherwise scan the snapshot directory
    try:
        for fn in os.listdir(_SNAPSHOT_DIR):
            if not fn.endswith(".json.gz"):
                continue
            path = os.path.join(_SNAPSHOT_DIR, fn)
            try:
                snap = load_snapshot_p1(path)
                params = snap.get("params", {})
            except Exception:
                continue
            if _params_match_strict_p1(params, canon):
                return path, "strict"
    except Exception:
        pass
    return None, "none"
# 8) LLM REASONING (ENSEMBLE PROMPTS)
# ============================================================================

def call_gpt_for_multistep_regulatory_mechanisms(
    output: Dict[str, Any],
    gene: str,
    reasoning_function: str,
    append_text: str,
    user_query: str
):
    """
    Ensemble-prompt version: uses DB information and external file(s) only.
    Returns final HTML (string) from the model response.
    """
    prompt_variants = [
        "Please provide a Top5 list of candidate regulatory genes based on relevance, sorted from most to least likely.",
        "List the 5 genes most likely to regulate the target gene under the experimental conditions, ordered by relevance.",
        "Based on the experimental conditions below, recommend the top 5 genes most strongly associated with regulating the target gene, in order of importance.",
        "Prioritize the 5 genes that best match the experimental conditions and output the Top5 list, most relevant first.",
        "Given the following experimental description, here are the Top5 gene recommendations, sorted by relevance."
    ]

    def run_simulation(iteration: int):
        """
        One 'multi-round' simulation. Calls GPT as needed.
        Skips steps automatically when inputs are empty.
        """
        db_empty = not output or not any(output.values())
        append_empty = (append_text == 'No File Input') or (not append_text.strip())

        # If both are empty, skip calling GPT
        if db_empty and append_empty:
            return "", ""

        db_info = json.dumps(output, ensure_ascii=False, indent=2) if not db_empty else ""
        append_info = "" if append_empty else append_text

        # Second round: database information
        if not db_empty:
            second_prompt = "\n".join([
                f"Gene of interest: {gene}",
                f"User query: {user_query}",
                "Database Information:", db_info,
                f"Based on the above, select up to 5 candidate genes that likely be downstream factor of {gene} in the context of ({reasoning_function}).",
                "Format each as: ['gene name': name, 'data resource': Database Names, 'mechanism explanation': hypothesis]. Do not any other word excapt result in format."
            ])
            try:
                second_response = client1.complete(
                    model=GPT_MODEL,
                    temperature=0,
                    top_p=1,
                    seed=42,
                    messages=[{"role": "user", "content": second_prompt}]
                ).choices[0].message.content.strip()
            except Exception as e:
                second_response = f"[ERROR obtaining second_response] {e}"
        else:
            second_response = ""

        # Third round: external text
        if not append_empty:
            append_prompt = "\n".join([
                f"Gene of interest: {gene}",
                f"User query: {user_query}",
                "External Information:", append_info,
                f"Select up to 5 candidate genes from the external info that likely be downstream factor of {gene} in the context of ({reasoning_function}).",
                "Format each as: ['gene name': name, 'data resource': details from append_info, 'mechanism explanation': hypothesis]. Do not any other word excapt result in format."
            ])
            try:
                append_response = client1.complete(
                    model=GPT_MODEL,
                    temperature=0,
                    top_p=1,
                    seed=42,
                    messages=[{"role": "user", "content": append_prompt}]
                ).choices[0].message.content.strip()
            except Exception as e:
                append_response = f"[ERROR obtaining append_response] {e}"
        else:
            append_response = ""

        return second_response, append_response

    # Run 5 simulation iterations (threads)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_simulation, i) for i in range(5)]
        wait(futures)
        simulation_results = [f.result() for f in futures]

    # Build final aggregation prompt
    third_sections = [
        f"Gene of interest: {gene}",
        f"Reasoning / Experiment description: {reasoning_function}",
        f"User query: {user_query}",
        "Results from 5 simulation iterations:",
    ]
    for idx, (sr, ar) in enumerate(simulation_results, start=1):
        third_sections.append(f"Iteration {idx} - Second round:\n{sr}\nAppend round:\n{ar}\n")
    third_sections.extend([
        "Instruction: Step 1: (Please extract every gene mentioned above and compute how many iterations it appeared.)\n",
        "Step 2: (List all genes with a hit count equals to 5, sort these genes alphabetically by gene name in the final HTML.)\n",
        "Step 3:(Format each candidate in HTML blocks as:\n<div class='gene-entry'>\n<h2>Candidate i</h2>\n<p><strong>Gene Name:</strong> ...<br>\n<strong>Hit Count:</strong> ...<br>\n<strong>Potential Mechanism:</strong> ...<br>\n<strong>Information Source:</strong> ...<br></p>\n</div><br>)",
        "Return only the HTML for direct insertion without extra wrappers. Do not add ``` or html in the front or in the end."
    ])
    third_prompt = "\n".join(third_sections)

    final_response = client2.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": third_prompt}]
    ).choices[0].message.content.strip()
    '''
    # Original alternative path (kept as a comment in source)
    final_response = client1.complete(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": third_prompt}]
    ).choices[0].message.content.strip()
    '''
    return final_response

# ============================================================================
# 9) HTML EXTRACTION / PARSING HELPERS
# ============================================================================

def extract_subtable_from_html(html: str, keyword: str) -> str:
    """
    Extract rows containing `keyword` from each <table> in `html` and return
    fully-styled tables that reuse the same markup & CSS hooks as the KG tables:
      • Wrapped in <div class='table-container'>…</div>
      • Proper <thead>/<tbody> (no inline borders/styles)
      • Each cell uses <div class="cell-content">…</div> for wrapping
      • Preserves inner HTML inside cells (links, italics, etc.)
    """
    if not html or not keyword:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    kw = keyword.strip().lower()
    out = []

    for tbl in soup.find_all("table"):
        # 1) Find a header row (prefer <thead>, else first <tr>)
        header_tr = None
        if tbl.thead:
            header_tr = tbl.thead.find("tr")
        if not header_tr:
            if tbl.tbody:
                body_trs = tbl.tbody.find_all("tr", recursive=False)
                header_tr = body_trs[0] if body_trs else None
            else:
                all_trs = tbl.find_all("tr", recursive=False)
                header_tr = all_trs[0] if all_trs else None

        # 2) Scan body rows (skip header & any thead/tfoot) and keep matches
        body_trs = tbl.tbody.find_all("tr") if tbl.tbody else tbl.find_all("tr")
        rows_to_scan = body_trs
        if header_tr and rows_to_scan and rows_to_scan[0] is header_tr:
            rows_to_scan = rows_to_scan[1:]

        matching_rows = []
        for tr in rows_to_scan:
            if tr.find_parent("thead") or tr.find_parent("tfoot"):
                continue
            row_text = tr.get_text(" ", strip=True).lower()
            if kw not in row_text:
                continue
            # Standardize each cell to <td><div class="cell-content">…</div></td>
            cells = []
            for cell in tr.find_all(["td", "th"], recursive=False):
                cells.append(
                    f'<td><div class="cell-content">{cell.decode_contents()}</div></td>'
                )
            if cells:
                matching_rows.append(f"<tr>{''.join(cells)}</tr>")

        if not matching_rows:
            continue

        # 3) Build a normalized header (semantic <th scope="col">…</th>)
        if header_tr:
            header_cells_html = "".join(
                f'<th scope="col">{c.decode_contents()}</th>'
                for c in header_tr.find_all(["th", "td"], recursive=False)
            )
        else:
            # Fallback generic header if we couldn't find one
            from bs4 import BeautifulSoup as _BS
            ncols = len(_BS(matching_rows[0], "html.parser").find_all("td"))
            header_cells_html = "".join(
                f'<th scope="col">Col_{i+1}</th>' for i in range(ncols)
            )

        # 4) Output using the same container/markup your CSS targets
        table_html = (
            "<div class='table-container'>\n"
            "  <table>\n"
            f"    <thead><tr>{header_cells_html}</tr></thead>\n"
            "    <tbody>\n"
            f"{''.join(matching_rows)}\n"
            "    </tbody>\n"
            "  </table>\n"
            "</div>\n"
        )
        out.append(table_html)

    return "\n".join(out)

def parse_subtasks(raw_response):
    """
    Ask GPT to parse text into a standard JSON array of subtasks:
      - index (string), title (gene name), details (text), keywords ([gene name only])
    Returns Python list (or [] on failure).
    """
    system_prompt = "You are a helpful assistant that parses text into a standard JSON array."
    user_prompt = f"""Please parse the following text into a standard JSON array. Each element in the array should be an object with the following keys:
- "index": the subtask number as a string,
- "title": the subtask title (gene name),
- "details": the detailed description of the subtask,
- "keywords": an array that only contains the gene name extracted from the subtask title (ignore any other words or symbols).
Please output only pure JSON without any extra markdown formatting (such as triple backticks) or additional text.
Text:
{raw_response}"""
    try:
        response = client1.complete(
            model=GPT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        final_answer = response.choices[0].message.content.strip()
        log_debug(f"parse_subtasks response from client1:\n{final_answer}")
        if not final_answer:
            log_debug("client1 returned an empty string.")
            return []
        if final_answer.startswith("```") and final_answer.endswith("```"):
            final_answer = final_answer.strip("```").strip()
        subtasks = json.loads(final_answer)
        return subtasks
    except Exception as e:
        error_msg = f"[ERROR] Error parsing subtasks: {e}"
        log_debug(error_msg)
        print(error_msg)
        return []

def retrieve_related_info(keywords, tables_html, data_analysis_html):
    """
    Retrieve keyword-matched subtables from provided HTML blocks (DB tables and analysis).
    """
    combined_info = ""
    for kw in keywords:
        if not kw:
            continue
        subtable_tables = extract_subtable_from_html(tables_html, kw)
        subtable_data = extract_subtable_from_html(data_analysis_html, kw)
        if subtable_tables:
            combined_info += subtable_tables + "\n\n"
        if subtable_data:
            combined_info += subtable_data + "\n\n"
    if not combined_info.strip():
        combined_info = ""
    return combined_info

# ============================================================================
# 10) LITERATURE SEARCH (GOOGLE + FAISS) & DETAILED FETCH
# ============================================================================

def get_google_pmc_links(query, api_key, cx, quota_user=None, target_count=3, max_results=5):
    """
    Use Vertex AI Discovery Engine searchLite (replaces Google Custom Search) to find PMC links.
    Deduplicate and limit results.
    Retracted/withdrawn items are detected (via PubMed E-utilities and PMC page banner) and removed.

    Tip: set environment variable NCBI_API_KEY to raise E-utilities limits (optional).

    Return format unchanged:
      [{"pmc_url": "...", "google_title": "..."} , ...]
    """
    import os
    import re
    import requests

    log_debug(f"Starting Vertex Search Lite for query: {query}, quotaUser={quota_user}")

    # --- config (project is required; engine_id/api_key can come from args or globals) ---
    project   = (VERTEX_PROJECT or "").strip()
    engine_id = ((cx or "") or (VERTEX_ENGINE_ID or "")).strip()
    key       = ((api_key or "") or (VERTEX_API_KEY or "")).strip()
    location  = (VERTEX_LOCATION or "global").strip()
    collection = (VERTEX_COLLECTION or "default_collection").strip()
    serving_config = (VERTEX_SERVING_CONFIG or "default_search").strip()

    if not project:
        raise SystemExit("[Config] Missing VERTEX_PROJECT in profile.json (or env var).")
    if not engine_id:
        raise SystemExit("[Config] Missing VERTEX_ENGINE_ID (or legacy GOOGLE_CX) in profile.json (or env var).")
    if not key:
        raise SystemExit("[Config] Missing VERTEX_API_KEY (or legacy GOOGLE_API_KEY) in profile.json (or env var).")

    BASE = (
        f"https://discoveryengine.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/collections/{collection}/"
        f"engines/{engine_id}/servingConfigs/{serving_config}:searchLite"
    )

    serving_cfg_path = (
        f"projects/{project}/locations/{location}/collections/{collection}/"
        f"engines/{engine_id}/servingConfigs/{serving_config}"
    )

    # --- keep your existing exclusions ---
    exclude_links = {
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC5181118/"
    }

    def _vertex_search_lite(q: str, page_size: int = 10, offset: int = 0) -> dict:
        payload = {
            # keep this field (official samples do this)
            "servingConfig": serving_cfg_path,
            "query": q,
            "pageSize": int(page_size),
            "offset": int(offset),  # 0-based
            "queryExpansionSpec": {"condition": "AUTO"},
            "spellCorrectionSpec": {"mode": "AUTO"},
            "languageCode": "en-US",
            "userInfo": {"timeZone": "America/New_York"},
            "userPseudoId": quota_user or "local-test-1",
        }
        r = requests.post(BASE, params={"key": key}, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _iter_vertex_items(resp: dict):
        """
        Yield simplified items with fields similar to Google CSE: title/link/snippet.
        """
        for item in (resp.get("results", []) or []):
            d = (item.get("document", {}).get("derivedStructData") or {})
            title = d.get("title") or d.get("htmlTitle") or ""
            link = (d.get("link") or "").split("#")[0]
            snippet = ""
            snippets = d.get("snippets") or []
            if snippets:
                snippet = snippets[0].get("snippet", "") or ""
            if link:
                yield {"title": title, "link": link, "snippet": snippet}

    # ---- helpers (unchanged logic) ----
    def _extract_pmcid(url: str):
        m = re.search(r"/articles/(PMC\d+)", url)
        return m.group(1) if m else None

    def _pmcid_to_pmid(pmcid: str, ncbi_key: str | None):
        try:
            r = requests.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={"format": "json", "ids": pmcid, **({"api_key": ncbi_key} if ncbi_key else {})},
                timeout=10,
            )
            if r.status_code != 200:
                log_debug(f"idconv failed {r.status_code}: {r.text[:200]}")
                return None
            j = r.json()
            recs = j.get("records", [])
            if not recs:
                return None
            return recs[0].get("pmid")
        except Exception as e:
            log_debug(f"idconv exception: {e}")
            return None

    def _is_retracted_pubmed_esummary(pmid: str, ncbi_key: str | None) -> bool:
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json", **({"api_key": ncbi_key} if ncbi_key else {})},
                timeout=10,
            )
            if r.status_code != 200:
                return False
            data = r.json()
            result = data.get("result", {})
            uid = (result.get("uids") or [pmid])[0]
            rec = result.get(uid, {})
            pubtypes = {str(pt).strip().lower() for pt in rec.get("pubtype", [])}
            retracted_types = {"retracted publication", "retraction of publication", "withdrawn publication"}
            return bool(pubtypes & retracted_types)
        except Exception as e:
            log_debug(f"esummary exception: {e}")
            return False

    def _is_retracted_pubmed_efetch(pmid: str, ncbi_key: str | None) -> bool:
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "xml", **({"api_key": ncbi_key} if ncbi_key else {})},
                timeout=10,
            )
            if r.status_code != 200:
                return False
            xml = r.text
            if re.search(r"<PublicationType[^>]*>\s*Retracted Publication\s*</PublicationType>", xml, re.I):
                return True
            if re.search(r"<PublicationType[^>]*>\s*Retraction of Publication\s*</PublicationType>", xml, re.I):
                return True
            if re.search(r"<PublicationType[^>]*>\s*Withdrawn Publication\s*</PublicationType>", xml, re.I):
                return True
            if re.search(r"<PublicationStatus>\s*retracted\s*</PublicationStatus>", xml, re.I):
                return True
            if re.search(r'(?:RefType|CommentsCorrectionsType)="RetractionIn"', xml, re.I):
                return True
            return False
        except Exception as e:
            log_debug(f"efetch exception: {e}")
            return False

    def _is_retracted_pmc_page(pmc_url: str) -> bool:
        try:
            r = requests.get(pmc_url, timeout=10)
            if r.status_code != 200:
                return False
            t = r.text
            if re.search(r"This article has been retracted", t, re.I):
                return True
            if re.search(r"\bRetraction in:\b", t, re.I) and re.search(r"\bRetracted\b", t, re.I):
                return True
            if re.search(r"\bhas been retracted\b", t, re.I):
                return True
        except Exception as e:
            log_debug(f"PMC page check exception: {e}")
        return False

    def _is_retracted_link(pmc_url: str, ncbi_key: str | None) -> bool:
        pmcid = _extract_pmcid(pmc_url)
        if not pmcid:
            return _is_retracted_pmc_page(pmc_url)
        pmid = _pmcid_to_pmid(pmcid, ncbi_key)
        if pmid and (_is_retracted_pubmed_esummary(pmid, ncbi_key) or _is_retracted_pubmed_efetch(pmid, ncbi_key)):
            return True
        return _is_retracted_pmc_page(pmc_url)

    ncbi_key = os.getenv("NCBI_API_KEY")  # optional
    distinct_results = []

    # Vertex paging uses offset (0-based). Keep your previous "10 per page" behavior.
    page_size = 10
    max_results = int(max_results or 0)

    for offset in range(0, max_results, page_size):
        try:
            resp = _vertex_search_lite(query, page_size=page_size, offset=offset)
        except Exception as e:
            msg = f"Error during search request: {e}"
            print(msg)
            log_debug(msg)
            break

        items = list(_iter_vertex_items(resp))

        for item in items:
            link = (item.get("link") or "").split("#")[0]
            if "pmc.ncbi.nlm.nih.gov" not in link:
                continue
            if link in exclude_links:
                continue
            if any(link == existing_item["pmc_url"] for existing_item in distinct_results):
                continue

            # --- keep your retraction filtering ---
            try:
                if _is_retracted_link(link, ncbi_key):
                    log_debug(f"Filtered retracted/withdrawn link: {link}")
                    continue
            except Exception as e:
                log_debug(f"Retraction check failed for {link}: {e}")

            distinct_results.append({
                "pmc_url": link,
                "google_title": item.get("title", "")
            })

            if len(distinct_results) >= target_count:
                break

        # stop if enough or no more pages
        if len(distinct_results) >= target_count or len(items) < page_size:
            break

    log_debug(f"Vertex search returned {len(distinct_results)} PMC links after retraction filtering.")
    return distinct_results


def get_combined_pmc_links(query: str,
                           api_key: str,
                           cx: str,
                           quota_user: str = None,
                           search_term: str = None,
                           google_target: int = 3,
                           faiss_target: int = 3,
                           enable_google: bool = True,
                           enable_faiss: bool = True) -> List[Dict[str, Any]]:
    """
    Return a deduplicated list of PMC links from Google and/or FAISS meta index.
    """
    google_links: List[Dict[str, Any]] = []
    if enable_google:
        google_links = get_google_pmc_links(query, api_key, cx, quota_user, target_count=google_target)

    faiss_results: List[Dict[str, Any]] = []
    if enable_faiss:
        faiss_docs = retrieve_top_docs(query, search_term or "", top_k=2000)
        if faiss_docs:
            for item in aggregate_and_sort(faiss_docs)[:faiss_target]:
                faiss_results.append({
                    "pmc_url"     : item["pmc_url"] or item["link"],
                    "google_title": item["title"],
                    "pmid"        : item["pmid"],
                    "source"      : "faiss",
                })

    combined: List[Dict[str, Any]] = []
    seen_urls = set()
    for entry in google_links + faiss_results:
        url = entry.get("pmc_url")
        if url and url not in seen_urls:
            combined.append(entry)
            seen_urls.add(url)
    return combined

def detailed_google_search(
    search_term: str,
    search_type: str,
    reasoning_function: str,
    gene_of_interest: str,
    enable_google: bool,
    enable_faiss: bool,
    quotaUser: str,
    *,
    max_workers: int = 1
):
    """
    Run multi-aspect search (General Function / Under Condition / Subcellular Localization).
    For each aspect, collect PMC pages, fetch content, filter by entity, and have GPT extract key evidence.

    Fallback rule added for General Information only:
      1) First search: "{search_term} gene function"
      2) If no PMC results, fallback search: "{search_term}"
    The downstream GPT evidence prompt remains unchanged.
    """
    log_debug(f"Starting detailed_google_search for term: {search_term}, type: {search_type}")

    aspects = [
        ("General Information",           f"{search_term} gene function"),
        ("Gene Function under condition", f"{search_term} gene function {reasoning_function}"),
        ("Subcellular Localization",      f"{search_term} subcellular localization"),
    ]

    html_result = "<div>"

    def _search_with_fallback(aspect_name: str, primary_query: str):
        """
        For 'General Information', if '{search_term} gene function' returns no PMC results,
        fallback to searching '{search_term}' directly.
        """
        effective_query = primary_query
        pmc_results = get_combined_pmc_links(
            primary_query,
            GOOGLE_API_KEY,
            GOOGLE_CX,
            quotaUser,
            search_term,
            enable_google=enable_google,
            enable_faiss=enable_faiss
        )

        if not pmc_results and aspect_name == "General Information":
            fallback_query = search_term
            log_debug(
                f"No PMC results for primary query '{primary_query}'. "
                f"Trying fallback query '{fallback_query}'."
            )
            fallback_results = get_combined_pmc_links(
                fallback_query,
                GOOGLE_API_KEY,
                GOOGLE_CX,
                quotaUser,
                search_term,
                enable_google=enable_google,
                enable_faiss=enable_faiss
            )
            if fallback_results:
                pmc_results = fallback_results
                effective_query = fallback_query
                log_debug(
                    f"Fallback query succeeded for '{search_term}'. "
                    f"Using '{fallback_query}' as General Information evidence source."
                )

        return pmc_results, effective_query

    for aspect, query in aspects:
        pmc_results, effective_query = _search_with_fallback(aspect, query)
        if not pmc_results:
            continue

        # Fetch in parallel (bounded)
        def _process_pmc(result):
            url = result.get("pmc_url", "")
            if not url:
                return None
            full_text = fetch_full_text(url)
            if not full_text:
                return None
            sentences = filter_text_by_entity(full_text, search_term)
            return (sentences, url) if sentences else None

        evidence_items = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(_process_pmc, r): r for r in pmc_results}
            for fut in as_completed(future_to_url):
                out = fut.result()
                if out:
                    evidence_items.append(out)

        if not evidence_items:
            continue

        # Combine evidence paragraphs
        combined_evidence = ""
        for idx, (sentences, url) in enumerate(evidence_items, 1):
            combined_evidence += (
                f"<p>Article {idx} Source: {url}\n" +
                "\n".join(sentences) +
                "</p>\n"
            )

        # Keep GPT prompt template unchanged; only the actual successful search query may differ
        second_prompt = (
            "You are a scientific evidence evaluator. Extract up to 5 sentences from the provided evidence paragraphs "
            "most relevant to the user query. Prioritize sentences from the Abstract section first, then Introduction, "
            "then Results, and finally Discussion. Provide each as 'Section: sentence' and include the source link. "
            "If none are relevant, respond with 'No relevant evidence found.'\n\n"
            f"Search Query: {effective_query}\n\nEvidence Items:\n<pre>{combined_evidence}</pre>"
            "The extracted sentence should be responsed as this format:\n"
            "1. Section: -Abstract- the sentence. [Source: the link]"
        )

        try:
            gpt_evaluation = client1.complete(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": second_prompt}]
            ).choices[0].message.content.strip()
        except Exception as e:
            gpt_evaluation = f"Error evaluating evidence: {e}"

        if "no relevant evidence" in gpt_evaluation.lower():
            continue

        html_result += (
            f"<h3>{aspect}</h3>"
            f"<p><strong>Search Query:</strong> {effective_query}</p>"
            f"<h4>GPT Evaluated Evidence for {aspect}</h4>"
            f"<pre>{gpt_evaluation}</pre><hr>"
        )

    html_result += "</div>"
    return html_result

def fetch_full_text(pmc_url):
    """
    Download a PMC page; extract meta, sections (Abstract/Introduction/Results/Discussion),
    and return a dict: {'pmid','pmc_url','date','journal','title','sections':{...}}
    """
    log_debug(f"Fetching full text from: {pmc_url}")
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/98.0.4758.102 Safari/537.36'
        )
    }
    try:
        response = requests.get(pmc_url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        if VERBOSE:
            print(f"Error fetching PMC page {pmc_url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    def get_meta_content(name):
        tag = soup.find("meta", {"name": name})
        return tag["content"].strip() if tag and tag.get("content") else None

    pmid = get_meta_content("citation_pmid")
    date = get_meta_content("citation_publication_date")
    if date and re.match(r'^\d{4} [A-Za-z]{3} \d{1,2}$', date):
        try:
            date_obj = datetime.datetime.strptime(date, "%Y %b %d")
            date = date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            if VERBOSE:
                print("Date format conversion failed:", e)

    journal = get_meta_content("citation_journal_title")
    title = get_meta_content("citation_title")

    for style in soup.find_all('style'):
        style.decompose()

    article_section = soup.find("section", attrs={"aria-label": "Article content"})
    if article_section:
        for ref_section in article_section.find_all("section", id="Bib1"):
            ref_section.decompose()
        for ref_section in article_section.find_all("section", class_="ref-list"):
            ref_section.decompose()
        for h2 in article_section.find_all("h2"):
            if h2.get_text(strip=True).lower() == "references":
                parent_section = h2.find_parent("section")
                if parent_section:
                    parent_section.decompose()
        full_text = article_section.get_text(separator=" ", strip=True)
        content_root = article_section
    else:
        full_text = soup.get_text(separator=" ", strip=True)
        content_root = soup

    soup.decompose()
    del soup, response

    full_text = re.sub(r'\s+', ' ', full_text)

    # Split into sections
    sections = {}
    section_order = [
        ("Abstract", ["abstract"]),
        ("Introduction", ["introduction"]),
        ("Results", ["result"]),
        ("Discussion", ["discussion"])
    ]
    headings = content_root.find_all(['h2', 'h3'])
    for sect_name, keywords in section_order:
        for heading in headings:
            text = heading.get_text(strip=True).lower()
            if any(keyword in text for keyword in keywords):
                content = []
                for sib in heading.next_siblings:
                    if getattr(sib, 'name', None) in ['h2', 'h3']:
                        break
                    if hasattr(sib, 'get_text'):
                        content.append(sib.get_text(separator=" ", strip=True))
                sections[sect_name] = re.sub(r'\s+', ' ', ' '.join(content))
                log_debug(f"Parsed section {sect_name} (length: {len(sections[sect_name])}).")
                break
    if not sections:
        sections['Results'] = full_text
        log_debug("No sections parsed; defaulting entire text to 'Results' section.")

    time.sleep(0.5)
    return {
        "pmid": pmid,
        "pmc_url": pmc_url,
        "date": date,
        "journal": journal,
        "title": title,
        "sections": sections
    }

def filter_text_by_entity(full_text_data, entity):
    """Extract sentences from sections that contain the given entity."""
    sections = full_text_data['sections']
    if not sections:
        sections = {'Results': ''}
    entity_lower = entity.lower()
    extracted = []
    for sect_name, text in sections.items():
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent_strip = sent.strip()
            sent_lower = sent_strip.lower()
            if entity_lower in sent_lower or any(word in sent_lower for word in entity_lower.split()):
                extracted.append(f"{sect_name}: {sent_strip}")
        log_debug(f"Filtering text for entity '{entity}' in section '{sect_name}'. Found {len([s for s in extracted if s.startswith(sect_name)])} sentences.")
    return extracted

# ============================================================================
# 11) INFERENCE BUILDING & FACT CHECK PIPELINE
# ============================================================================

def get_regulatory_inference(
    subtask_prompt: str,
    gene_of_interest: str,
    related_gene: str,
    experiment: str,
    tissue: str,
):
    """
    Construct knowledge-graph tables (Nodes/Edges) in HTML with two steps:
    - Step 1: Edges 1–6 based on facts and available evidence.
    - Step 2: Edges 7–10 with relation classification (Strong/Weak/No).
    """
    EDGE_ROW_TEMPLATE = (
        "<tr>"
        "<td>Edge_ID</td>"
        "<td>Connection</td>"
        "<td>Node_in_Edge</td>"
        "<td>Edge_Type</td>"
        "<td>Explanation</td>"
        "</tr>"
    )

    html_template = (
        "<div class='table-container'>"
        "<h3>Nodes</h3>"
        "<table id='kg-nodes'>"
        "<thead><tr><th>Node_ID</th><th>Node_Name</th><th>Definition</th></tr></thead>"
        "<tbody>{nodes_rows}</tbody>"
        "</table>"
        "</div>"
        "<div class='table-container'>"
        "<h3>Edges</h3>"
        "<table id='kg-edges'>"
        "<thead><tr><th>Edge_ID</th><th>Connection</th><th>Node_in_Edge</th>"
        "<th>Edge_Type</th><th>Explanation</th></tr></thead>"
        "<tbody>{edges_rows}</tbody>"
        "</table>"
        "</div>"
    )

    # Assemble Nodes rows
    nodes_rows = (
        f"<tr><td>Node_1</td><td>{gene_of_interest}</td><td>Node of gene of interest</td></tr>"
        f"<tr><td>Node_2</td><td>{related_gene}</td><td>Node of related gene</td></tr>"
        f"<tr><td>Node_3</td><td>{gene_of_interest}_GF</td><td>Gene-function node</td></tr>"
        f"<tr><td>Node_4</td><td>{gene_of_interest}_SL</td><td>Sub-cellular localisation node</td></tr>"
        f"<tr><td>Node_5</td><td>{gene_of_interest}_BD</td><td>Binding-data node</td></tr>"
        f"<tr><td>Node_6</td><td>{related_gene}_GF</td><td>Gene-function node</td></tr>"
        f"<tr><td>Node_7</td><td>{related_gene}_SL</td><td>Sub-cellular localisation node</td></tr>"
        f"<tr><td>Node_8</td><td>{related_gene}_BD</td><td>Binding-data node</td></tr>"
    )

    # Step 1 prompt
    step1_prompt = (
        f"Based on the Subtask Information, please generate Edge_1 to Edge_6 in HTML format with information from Subtask Information. "
        f"When you list evidence, include as many relevant pieces of evidence as possible. Focus on summarizing the positive findings and insights, "
        f"and avoid stating only negative conclusions (e.g., 'no evidence found'). Instead, highlight what is known and suggest areas for further exploration if data is limited. "
        f"Do not add any other words except for the 6 Edge HTML lines:\n\n"
        "# ---------- Edges ----------\n"
        f"   - [Edge_ID: Edge_1, Connection: Node_1-Node_3, Node_in_Edge: {gene_of_interest}-{gene_of_interest}_GF, Edge_Type: Fact, Explanation: Summarize evidence from (Detailed Google Search Results) and (Gene of Interest Fact Check) to describe the gene function of {gene_of_interest}. Highlight key findings and, if evidence is limited, suggest potential areas for further research.]\n"
        f"   - [Edge_ID: Edge_2, Connection: Node_1-Node_4, Node_in_Edge: {gene_of_interest}-{gene_of_interest}_SL, Edge_Type: Fact, Explanation: Summarize evidence from (Detailed Google Search Results) and (Gene of Interest Fact Check) to describe the subcellular localization of {gene_of_interest}. Include specific compartments or locations, and mention if more studies are needed for confirmation.]\n"
        f"   - [Edge_ID: Edge_3, Connection: Node_1-Node_5, Node_in_Edge: {gene_of_interest}-{gene_of_interest}_BD, Edge_Type: Fact, Explanation: Summarize evidence from (Related Information) about binding events involving {gene_of_interest} and {related_gene}. Focus on specific interactions and contexts, and suggest potential unexplored binding mechanisms if data is limited.]\n"
        f"   - [Edge_ID: Edge_4, Connection: Node_2-Node_6, Node_in_Edge: {related_gene}-{related_gene}_GF, Edge_Type: Fact, Explanation: Provide a detailed summary of the gene function of {related_gene} based on Subtask Information. Highlight key roles and pathways, and note if further functional studies could reveal additional insights.]\n"
        f"   - [Edge_ID: Edge_5, Connection: Node_2-Node_7, Node_in_Edge: {related_gene}-{related_gene}_SL, Edge_Type: Fact, Explanation: Provide a detailed summary of the subcellular localization of {related_gene} based on Subtask Information. Mention specific locations and indicate if more precise localization studies are needed.]\n"
        f"   - [Edge_ID: Edge_6, Connection: Node_2-Node_8, Node_in_Edge: {related_gene}-{related_gene}_BD, Edge_Type: Fact, Explanation: Summarize binding events involving {gene_of_interest} and {related_gene} from Subtask Information. Focus on specific interactions and experimental contexts, and suggest potential areas for further binding studies if data is incomplete.]\n"
        "Construct your Edge_1-Edge_6 answer using the Edge part in HTML template below:\n\n"
        f"{EDGE_ROW_TEMPLATE}\n\n"
        f"Subtask Information:\n{subtask_prompt}"
    )

    step1_response = (
        client2.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": step1_prompt}],
        )
        .choices[0]
        .message.content.strip()
    )

    # Step 2 prompt
    step2_prompt = (
        f"Based on the Subtask Information, please generate Edge_7 to Edge_10 in HTML format. Focus on providing insightful explanations by summarizing positive evidence and suggesting potential connections or areas for further research. Avoid cold or purely negative statements (e.g., 'no evidence found' or 'no direct link'). Instead, classify the relation as 'Strong Relation', 'Weak Relation', or 'No Relation' based on the strength of evidence, and highlight both existing insights and potential future directions in the explanation. Do not add any other words except for the 4 Edge HTML lines:\n\n"
        "# ---------- Edges ----------\n"
        f"   - [Edge_ID: Edge_7, Connection: Node_3-Node_6, Node_in_Edge: {gene_of_interest}_GF-{related_gene}_GF, Edge_Type: Strong Relation/Weak Relation/No Relation (choose only one based on evidence strength), Explanation: Summarize the evidence linking the gene functions of {gene_of_interest} and {related_gene}. Highlight shared pathways, regulatory mechanisms, or functional overlaps. If evidence supports a direct regulatory axis (≤2 intermediaries), classify as 'Strong Relation'. If evidence shows overlap in processes but lacks direct links, classify as 'Weak Relation'. If evidence is insufficient or unrelated, classify as 'No Relation'. In all cases, suggest potential connections or further studies to explore their functional relationship.]\n"
        f"   - [Edge_ID: Edge_8, Connection: Node_4-Node_7, Node_in_Edge: {gene_of_interest}_SL-{related_gene}_SL, Edge_Type: Strong Relation/Weak Relation/No Relation (choose only one based on evidence strength), Explanation: Summarize evidence regarding subcellular localization overlap between {gene_of_interest} and {related_gene}. If there is evidence of co-localization with functional relevance, classify as 'Strong Relation'. If separate localization data suggests potential overlap without direct proof, classify as 'Weak Relation'. If localizations do not overlap or data is insufficient, classify as 'No Relation'. Suggest areas for further localization studies to confirm potential interactions in all cases.]\n"
        f"   - [Edge_ID: Edge_9, Connection: Node_5-Node_8, Node_in_Edge: {gene_of_interest}_BD-{related_gene}_BD, Edge_Type: Strong Relation/Weak Relation/No Relation (choose only one based on evidence strength), Explanation: Summarize evidence of binding events between {gene_of_interest} and {related_gene}. If at least one binding event is reported in the same tissue or organ as the experiment ({experiment}), classify as 'Strong Relation'. If binding events exist but not in the matching context, classify as 'Weak Relation'. If no binding data is available, classify as 'No Relation'. Highlight specific interactions and suggest further binding studies to explore potential direct interactions in all cases.]\n"
        f"   - [Edge_ID: Edge_10, Connection: Node_1-Node_2, Node_in_Edge: {gene_of_interest}-{related_gene}, Edge_Type: Strong Relation/Weak Relation/No Relation (choose only one based on combined evidence), Explanation: Provide a conclusive summary based on the combined evidence from gene function (GF-GF), subcellular localization (SL-SL), and binding data (BD-BD). Classify as 'Strong Relation' if at least one edge is 'Strong Relation' and another is at least 'Weak Relation'. Classify as 'Weak Relation' if there is suggestive evidence but not strong across multiple edges. Classify as 'No Relation' if evidence is largely insufficient, especially for GF-GF and BD-BD. Highlight key insights from existing evidence and propose potential directions for future research to confirm or refute a connection in all cases.]\n\n"
        "[Guidelines for Evidence and Classification]:\n"
        "- Evidence Summarization: summarize only the most relevant evidence; focus on positive findings; suggest next experiments when data is limited.\n"
        "- GF-GF, SL-SL, BD-BD classification rules as described above.\n"
        "- Main-Main classification: follows the combination rules described above.\n"
        f"Construct your Edge_7-Edge_10 answer using the Edge part in HTML template below:\n"
        f"{EDGE_ROW_TEMPLATE}\n\n"
        f"Subtask Information (Edge_1-Edge_6):\n{step1_response}"
    )

    step2_response = (
        client2.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": step2_prompt}],
        )
        .choices[0]
        .message.content.strip()
    )

    # Extract <tr> rows from both blocks and merge
    def _extract_edge_rows(html_block: str) -> str:
        rows = re.findall(r"<tr>.*?</tr>", html_block, flags=re.S)
        # Normalize any th…/th to td…/td
        rows = [re.sub(r"</?th", lambda m: m.group().replace("th", "td"), r) for r in rows]
        return "\n".join(rows)

    edges_rows = _extract_edge_rows(step1_response) + _extract_edge_rows(step2_response)

    final_html_text = html_template.format(
        nodes_rows=nodes_rows,
        edges_rows=edges_rows,
    )
    return final_html_text

def Fact_Check(
    gpt_response: str,
    tables_html: str,
    data_analysis_html: str,
    gene_of_interest: str,
    experiment_conditions: str,
    species: str,
    tissue: str,
    quotaUser: str,
    ENABLE_FAISS: bool,
    ENABLE_GOOGLE: bool,
) -> Tuple[str, List[str]]:
    """
    Fact-check pipeline executed sequentially (minimize peak memory):
      1) Parse subtasks (related genes)
      2) Build Gene-of-Interest (GoI) evidence block
      3) For each subtask gene: detailed search + inference + merge
      4) Return final HTML block and list of gene names
    """
    # 1) Parse subtasks
    subtasks = parse_subtasks(gpt_response)
    if not subtasks:
        return "<p>No subtasks could be parsed.</p>", []

    # 2) Gene-of-Interest block
    gene_names: List[str] = [gene_of_interest]
    html_chunks: List[str] = ["<div>"]

    safe_id = "gene_1"
    gene_info = detailed_google_search(
        gene_of_interest,
        "gene_of_interest",
        experiment_conditions,
        gene_of_interest,
        ENABLE_GOOGLE,
        ENABLE_FAISS,
        quotaUser,
    )
    html_chunks.append(
        f"""
        <div id='{safe_id}' class='fact-check-gene-of-interest'
             style='border:1px solid #ccc; padding:10px; margin-bottom:10px;
                    word-wrap:break-word; overflow-wrap:break-word;'>
            <br><h2>Gene of Interest Fact Check</h2>
            <div class='fact-check-table-container'>{gene_info}</div>
        </div><hr>"""
    )

    # Helper: fetch one keyword block
    def _fetch_one_keyword(kw: str) -> str:
        """Sequentially fetch single keyword literature (Google/FAISS)."""
        res = detailed_google_search(
            kw,
            "subtask_gene",
            experiment_conditions,
            gene_of_interest,
            ENABLE_GOOGLE,
            ENABLE_FAISS,
            quotaUser,
        )
        return res.replace("<tr><tr>", "<tr>").replace("</tr></tr>", "</tr>")

    # Helper: process one subtask gene
    def _process_one_subtask(idx: int, subtask: dict) -> Tuple[str, str]:
        keywords = subtask.get("keywords") or re.split(r"[\s_]+", subtask.get("title", ""))
        keywords = [kw for kw in keywords if kw]
        related_gene = keywords[0] if keywords else "unknown"

        g_parts = [_fetch_one_keyword(kw) for kw in keywords]
        g_results = "".join(g_parts)

        rel_info = retrieve_related_info(keywords, tables_html, data_analysis_html) \
                     .replace("<tr><tr>", "<tr>").replace("</tr></tr>", "</tr>")

        subtask_prompt = (
            f"Gene of Interest: {gene_of_interest}\n"
            f"Gene of Interest Information: {gene_info}\n"
            f"Related Gene: {related_gene}\n"
            f"Subtask Title: {subtask.get('title', 'No Title')}\n"
            f"Related Information: {rel_info}\n"
            f"Detailed Google Search Results: {g_results}"
        )
        inference = get_regulatory_inference(
            subtask_prompt,
            gene_of_interest,
            related_gene,
            experiment_conditions,
            tissue,
        ).replace("<tr><tr>", "<tr>").replace("</tr></tr>", "</tr>")

        chunk = (
            f"<div id='gene_{idx + 2}' class='fact-check-subtask'"
            " style='border:1px solid #ccc; padding:10px; margin-bottom:10px;"
            "        word-wrap:break-word; overflow-wrap:break-word;'>"
            f"<br><h3>{subtask.get('title', 'No Title')}</h3>"
            "<br><strong>Related Information from Databases and Data Analysis:</strong><br>"
            f"<div class='fact-check-table-container'>{rel_info}</div>"
            "<br><h2>Detailed Google Search Results:</h2><br>"
            f"<div class='fact-check-table-container'>{g_results}</div>"
            "<br><h2>Knowledge Graph Construction:</h2><br>"
            f"<div class='fact-check-table-container'>{inference}</div>"
            "</div><hr>"
        )

        # Release large locals promptly
        del rel_info, g_results, inference, subtask_prompt
        gc.collect()

        return chunk, related_gene

    # 3) Iterate subtasks sequentially
    for idx, st in enumerate(subtasks):
        chunk, r_gene = _process_one_subtask(idx, st)
        html_chunks.append(chunk)
        if r_gene not in gene_names:
            gene_names.append(r_gene)
        gc.collect()

    # 4) Finalize
    html_chunks.append("</div>")
    final_html = "".join(html_chunks)

    del subtasks, html_chunks
    gc.collect()

    return final_html, gene_names

# ============================================================================
# 12) GRAPH GENERATION
# ============================================================================

def generate_knowledge_graph(fact_check_html, gene_of_interest):
    """
    Parse 'Knowledge Graph Construction' sections to collect nodes/edges,
    then render a Cytoscape graph with a legend. GoI node is highlighted.
    """
    def normalize_label(label):
        # Replace various unicode dashes with ASCII '-'
        return re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015]', '-', label).strip()

    soup = BeautifulSoup(fact_check_html, 'html.parser')

    global_nodes = {}  # key: normalized label, value: node dict
    global_edges = []  # list of edge dicts
    node_id_counter = 1
    edge_id_counter = 1

    # Find all Knowledge Graph Construction sections
    kg_sections = soup.find_all('h2', string=lambda text: text and "Knowledge Graph Construction" in text)
    soup.decompose()
    del soup

    for kg in kg_sections:
        parent_div = kg.find_parent('div')
        if not parent_div:
            continue

        # Parse Nodes
        nodes_header = parent_div.find('h3', string=lambda text: text and "Nodes" in text)
        if nodes_header:
            nodes_table = nodes_header.find_next('table')
            if nodes_table:
                rows = nodes_table.find_all('tr')[1:]  # skip header
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 3:
                        continue
                    original_label = cols[1].get_text(strip=True)
                    label = normalize_label(original_label)
                    definition = cols[2].get_text(strip=True)
                    node_type = 'sub' if any(suffix in label for suffix in ['_GF', '_SL', '_EXP', '_BD']) else 'main'
                    color = 'orange' if node_type == 'main' else 'blue'
                    if label not in global_nodes:
                        global_nodes[label] = {
                            'id': f'Node_{node_id_counter}',
                            'label': label,
                            'definition': definition,
                            'node_type': node_type,
                            'bgColor': color
                        }
                        node_id_counter += 1
                    else:
                        if not global_nodes[label]['definition'] and definition:
                            global_nodes[label]['definition'] = definition

        # Parse Edges
        edges_header = parent_div.find('h3', string=lambda text: text and "Edges" in text)
        if edges_header:
            edges_table = edges_header.find_next('table')
            if edges_table:
                rows = edges_table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        continue
                    connection = cols[1].get_text(strip=True)
                    node_in_edge = cols[2].get_text(strip=True)
                    edge_type = cols[3].get_text(strip=True)
                    explanation = cols[4].get_text(strip=True)
                    if edge_type.strip().lower() == 'no relation':
                        continue

                    # Try to find source/target by matching existing labels
                    source_label = None
                    target_label = None
                    norm_in_edge = normalize_label(node_in_edge)
                    for label in global_nodes:
                        if norm_in_edge.startswith(label):
                            rest = norm_in_edge[len(label):].lstrip('-').strip()
                            if rest in global_nodes:
                                source_label, target_label = label, rest
                                break

                    # Fallback split
                    if source_label is None or target_label is None:
                        parts = [p.strip() for p in re.split(r'\s-\s|,', norm_in_edge) if p.strip()]
                        if len(parts) == 2:
                            source_label, target_label = parts

                    if not source_label or not target_label:
                        continue

                    # Ensure nodes exist
                    for lbl in (source_label, target_label):
                        if lbl not in global_nodes:
                            node_type_local = 'sub' if any(s in lbl for s in ['_GF', '_SL', '_EXP', '_BD']) else 'main'
                            color_local = 'orange' if node_type_local == 'main' else 'blue'
                            global_nodes[lbl] = {
                                'id': f'Node_{node_id_counter}',
                                'label': lbl,
                                'definition': '',
                                'node_type': node_type_local,
                                'bgColor': color_local
                            }
                            node_id_counter += 1

                    source_id = global_nodes[source_label]['id']
                    target_id = global_nodes[target_label]['id']

                    et = edge_type.strip().lower()
                    if 'strong' in et:
                        lineColor = 'red'
                    elif 'weak' in et:
                        lineColor = 'blue'
                    else:
                        lineColor = 'orange'

                    edge = {
                        'id': f'Edge_{edge_id_counter}',
                        'source': source_id,
                        'target': target_id,
                        'lineColor': lineColor
                    }
                    edge_id_counter += 1
                    if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in global_edges):
                        global_edges.append(edge)

    # Highlight gene_of_interest node in red
    normalized_gene = normalize_label(gene_of_interest)
    for node in global_nodes.values():
        if node['label'] == normalized_gene:
            node['bgColor'] = 'red'

    # Build Cytoscape elements
    elements = []
    for node in global_nodes.values():
        elements.append({'data': node})
    for edge in global_edges:
        elements.append({'data': edge})

    # Legend
    legend_html = '''
<div id="legend" style="margin-top:20px;">
  <h3>Legend</h3>
  <ul style="list-style: none; padding-left: 0;">
    <li><span style="display:inline-block;width:20px;height:20px;background:red;margin-right:5px;"></span> Gene of Interest</li>
    <li><span style="display:inline-block;width:20px;height:20px;background:orange;margin-right:5px;"></span> RBP</li>
    <li><span style="display:inline-block;width:20px;height:20px;background:blue;margin-right:5px;"></span> Attribute</li>
    <li><span style="display:inline-block;width:20px;height:20px;background:red;margin-right:5px;"></span> Strong Edge</li>
    <li><span style="display:inline-block;width:20px;height:20px;background:blue;margin-right:5px;"></span> Weak Edge</li>
    <li><span style="display:inline-block;width:20px;height:20px;background:orange;margin-right:5px;"></span> Fact Edge</li>
  </ul>
</div>
'''

    graph_html = f"""
<style>
#cytoscape-graph{{
  width:100%;
  height:600px;
  display:block;
}}
</style>
<div id="cytoscape-graph"></div>
<script>
document.addEventListener("DOMContentLoaded", function(){{
    var cy = cytoscape({{
        container: document.getElementById('cytoscape-graph'),
        elements: {json.dumps(elements)},
        style: [
            {{
                selector: 'node',
                style: {{
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': 'data(bgColor)',
                    'background-color': 'data(bgColor)',
                    'width': 40,
                    'height': 40,
                    'font-size': 10,
                    'text-wrap': 'wrap',
                    'text-max-width': 80
                }}
            }},
            {{
                selector: 'edge',
                style: {{
                    'width': 2,
                    'curve-style': 'bezier',
                    'line-color': 'data(lineColor)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': 'data(lineColor)',
                    'arrow-scale': 0.8
                }}
            }}
        ],
        layout   : {{
          name   : 'concentric',
          fit    : true,
          padding: 50,
          animate: false,
          concentric : n => n.degree(),
          levelWidth : () => 2
        }}
    }});

      cy.on('layoutstop', () => cy.fit());

      const resizeObserver = new ResizeObserver(() => {{
        cy.resize();
        cy.fit();
      }});
      resizeObserver.observe(document.getElementById('cytoscape-graph'));

      document.addEventListener('myTabShown', () => {{
        cy.resize();
        cy.fit();
      }});
}});
</script>
{legend_html}
"""
    return graph_html

# ============================================================================
# 13) HTML PAGE WRAPPER
# ============================================================================


# ============================================================================
# 13) SHARED HEAD (CSS/JS) + HTML WRAPPERS (deduped)
# ============================================================================

# Shared, deduplicated CSS used by BOTH the simple placeholder page and the full report.
COMMON_CSS = """<style>
  /* === Shared base tokens and layout === */
  :root {
    --base-font: 16.5px;
    --h1: 22px; --h2: 20px; --h3: 18px; --h4: 16px;
    --sidebar-width: 260px; --sidebar-left: 24px; --content-gap: 28px;

    --surface: #ffffff; --surface-2: #f7f9fb;
    --primary: #0b6cff; --primary-600: #084fc4;
    --border: #e5e7eb; --text: #0f172a; --muted: #64748b;
    --table-head: #f1f5f9;

    /* DB table scroll vars */
    --db-max-height: 60vh;
    --db-min-table-w: 960px;
    --db-cell-w: 160px;
    --gtex-name-w: 260px;

    /* NEW: per-table/per-tab column widths */
    --npinter-interaction-w: 700px;  /* NPInter → Interaction summary */
    --edges-explanation-w: 460px;    /* Fact Check → KG Edges 'Explanation' */
  }

  * { margin:0; padding:0; box-sizing:border-box; }
  html { scroll-behavior: smooth; }
  body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: var(--base-font);
    line-height: 1.6;
    color: var(--text);
    background: var(--surface-2);
  }

  h1 { font-size: var(--h1); }
  h2 { font-size: var(--h2); }
  h3 { font-size: var(--h3); }
  h4 { font-size: var(--h4); }

  header {
    background: linear-gradient(to right, #00274d, #5a9bd6);
    color: #fff;
    padding: 18px 20px;
    text-align: center;
  }
  footer {
    background: #00274d;
    color: #fff;
    text-align: center;
    padding: 14px;
    margin-top: 20px;
  }

  /* Container defaults used by both pages */
  .container {
    padding: 20px;
    margin-left: calc(var(--sidebar-left) + var(--sidebar-width) + var(--content-gap));
    margin-right: 20px;
  }
  .textbox {
    width: 90%;
    margin: 0 auto;
    padding: 18px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(15,23,42,.06);
    overflow-y: auto;
    overflow-x: hidden;
    max-height: calc(100vh - 220px);
  }

  /* Sidebar (used on report page; harmless elsewhere) */
  #factCheckSidebar {
    position: fixed !important;
    top: 150px !important;
    left: var(--sidebar-left) !important;
    width: var(--sidebar-width) !important;
    max-height: calc(100vh - 140px) !important;
    overflow: auto !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 28px rgba(15,23,42,.10) !important;
    padding: 12px !important;
  }
  #factCheckSidebar h3 { margin: 2px 4px 10px; font-weight: 700; }
  .sidebar-button {
    display: flex; align-items: center; gap: 8px;
    width: 100%; padding: 10px 12px; margin-bottom: 8px;
    background: #f1f5f9; color: var(--text);
    border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer;
    transition: background .18s ease, transform .12s ease, border-color .18s ease;
    text-align: left; white-space: normal; word-break: break-word;
  }
  .sidebar-button:hover { background: #e2e8f0; transform: translateY(-1px); }
  .sidebar-button.active {
    background: var(--primary); color: #fff; border-color: var(--primary);
    box-shadow: 0 0 0 1px var(--primary) inset, 0 4px 10px rgba(8,79,196,.25);
  }

  /* Tables (shared) */
  .table-container {
    max-height: var(--db-max-height);
    overflow: auto;              /* both axes when needed */
    margin: 12px 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: #fff;
    cursor: grab;
  }
  .table-container.dragging { cursor: grabbing; user-select: none; }

  .table-container table {
    width: max-content;          /* do not shrink */
    min-width: var(--db-min-table-w);
    border-collapse: collapse;
    table-layout: fixed;         /* honor fixed cell widths */
    font-size: 14px;
  }

  .table-container thead th {
    position: sticky; top: 0; z-index: 1;
    background: var(--table-head); font-weight: 700;
  }

  .table-container th,
  .table-container td {
    width: var(--db-cell-w);
    min-width: var(--db-cell-w);
    max-width: var(--db-cell-w);
    border: 1px solid #e5e7eb;
    padding: 10px;
    text-align: left;
    vertical-align: top;
    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
  }

  .table-container tr:nth-child(even) { background: #fafcff; }
  .table-container tr:hover { background: #f2f8ff; }

  .cell-content {
    max-width: 100%;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
  }

  /* GTEx first column wider */
  .gtex-container table th:first-child,
  .gtex-container table td:first-child {
    width: var(--gtex-name-w) !important;
    min-width: var(--gtex-name-w) !important;
    max-width: var(--gtex-name-w) !important;
  }
  .gtex-container table { min-width: 1400px !important; }
  #dbResults .col-npinter-interaction {
    width: var(--npinter-interaction-w) !important;
    min-width: var(--npinter-interaction-w) !important;
    max-width: var(--npinter-interaction-w) !important;
  }
  #factCheck table#kg-edges thead th:nth-child(5),
  #factCheck table#kg-edges tbody td:nth-child(5) {
    width: var(--edges-explanation-w) !important;
    min-width: var(--edges-explanation-w) !important;
    max-width: var(--edges-explanation-w) !important;
  }
  /* Fact Check card containers */
  .fact-check-table-container {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    background: #fbfdff;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
  }
  .fact-check-gene-of-interest, .fact-check-subtask {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    background: var(--surface);
    box-shadow: 0 6px 14px rgba(15,23,42,.06);
    overflow: hidden;
  }
  .fact-check-gene-of-interest *, .fact-check-subtask * { max-width: 100%; }

  .fact-check-gene-of-interest pre,
  .fact-check-subtask pre,
  .fact-check-table-container pre,
  .fact-check-gene-of-interest code,
  .fact-check-subtask code,
  .fact-check-table-container code {
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    font-size: 14px;
  }

  /* Cytoscape graph area tuned for 1080p */
  #cytoscape-graph { width: 100%; height: 680px; min-height: 520px; }

  /* Responsive fallback */
  @media (max-width: 980px) {
    #factCheckSidebar {
      position: static !important;
      width: auto !important;
      max-height: none !important;
      left: auto !important;
    }
    .container {
      margin-left: 20px !important;
      margin-right: 20px !important;
    }
    .tabs, .textbox { width: 100%; }
  }
</style>"""

# Shared, deduplicated JS used by BOTH pages (drag-to-scroll + table wrapping).
COMMON_JS = """<script>
document.addEventListener('DOMContentLoaded', function(){
  // Drag-to-scroll for all .table-container boxes
  var boxes = document.querySelectorAll('.table-container');
  boxes.forEach(function(el){
    var down = false, sx = 0, sy = 0, sl = 0, st = 0;
    el.addEventListener('mousedown', function(e){
      if (e.button !== 0) return;
      down = true; sx = e.pageX; sy = e.pageY; sl = el.scrollLeft; st = el.scrollTop;
      el.classList.add('dragging');
    });
    window.addEventListener('mousemove', function(e){
      if (!down) return;
      el.scrollLeft = sl - (e.pageX - sx);
      el.scrollTop  = st - (e.pageY - sy);
    });
    window.addEventListener('mouseup', function(){
      down = false; el.classList.remove('dragging');
    });
    el.addEventListener('touchstart', function(e){
      if (!e.touches || !e.touches.length) return;
      down = true; sx = e.touches[0].clientX; sy = e.touches[0].clientY; sl = el.scrollLeft; st = el.scrollTop;
    }, { passive: true });
    el.addEventListener('touchmove', function(e){
      if (!down || !e.touches || !e.touches.length) return;
      el.scrollLeft = sl - (e.touches[0].clientX - sx);
      el.scrollTop  = st - (e.touches[0].clientY - sy);
    }, { passive: true });
    el.addEventListener('touchend', function(){ down = false; });
  });

  // === FACT CHECK FIX ===
  // Ensure every table inside the Fact Check area is wrapped in .table-container and has .cell-content
  var scopes = document.querySelectorAll('#factCheck .fact-check-table-container, #factCheck');
  scopes.forEach(function(scope){
    var tables = scope.querySelectorAll('table');
    tables.forEach(function(tbl){
      var inContainer = tbl.closest('.table-container');
      if (!inContainer) {
        var wrapper = document.createElement('div');
        wrapper.className = 'table-container';
        tbl.parentNode.insertBefore(wrapper, tbl);
        wrapper.appendChild(tbl);
      }
      // add missing .cell-content wrappers for robustness
      tbl.querySelectorAll('td').forEach(function(td){
        if (!td.querySelector('.cell-content')) {
          var div = document.createElement('div');
          div.className = 'cell-content';
          while (td.firstChild) div.appendChild(td.firstChild);
          td.appendChild(div);
        }
      });
    });
  });

  // === NPInter width fallback ===
  (function(){
    var db = document.getElementById('dbResults');
    if (!db) return;
    db.querySelectorAll('h3').forEach(function(h3){
      if ((h3.textContent || '').trim().toLowerCase() !== 'npinter') return;
      var container = h3.nextElementSibling;
      var tbl = container && container.querySelector ? container.querySelector('table') : null;
      if (!tbl) return;
      var headers = Array.from(tbl.querySelectorAll('thead th'));
      var idx = headers.findIndex(function(th){
        var t = (th.textContent || '').trim().toLowerCase().replace(/[\\s_-]+/g,' ');
        return (t === 'summary' || (t.includes('interaction') && t.includes('summary')));
      });
      if (idx >= 0 && !headers[idx].classList.contains('col-npinter-interaction')) {
        headers[idx].classList.add('col-npinter-interaction');
        Array.from(tbl.tBodies[0].rows).forEach(function(tr){
          if (tr.cells[idx]) tr.cells[idx].classList.add('col-npinter-interaction');
        });
      }
    });
  })();
});
</script>"""

def _html_shell(body_inner: str,
                *, title: str = "Skill 1 — lncRNA–RBP Inference",
                head_extra_css: str = "",
                body_extra_js: str = "",
                include_cytoscape: bool = False) -> str:
    """Small helper to assemble a full HTML page from shared head + provided body."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {COMMON_CSS}
  {head_extra_css}
</head>
<body>
  <header><h1>{title}</h1></header>
  {body_inner}
  <footer><p>&copy; 2025 AgentLnc. All Rights Reserved.</p></footer>
  {COMMON_JS}
  {body_extra_js}
  {('<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.21.1/cytoscape.min.js"></script>') if include_cytoscape else ''}
</body>
</html>"""

def build_html_page(inner_html: str) -> str:
    """Wrap provided HTML content into a minimal report shell using shared CSS/JS."""
    return _html_shell(inner_html, title="Skill 1 — lncRNA–RBP Inference")

def compose_report_html(knowledge_graph_html: str,
                        tables_html: str,
                        gpt_response: str,
                        fact_check_html: str,
                        gene_names: list[str]) -> str:
    """
    Unified final HTML for the full report (tabs + sidebar), reusing shared CSS/JS.
    Only page‑specific CSS/JS for tabs & navigation are added here.
    """
    # Sidebar (built from gene_names)
    fact_check_sidebar_html = ["<div id='factCheckSidebar'><h3>Fact Check</h3>"]
    for idx, gname in enumerate(gene_names or []):
        fact_check_sidebar_html.append(
            f"<button class='sidebar-button' data-target='gene_{idx+1}' onclick=\"goToFactCheckAndScroll('gene_{idx+1}', this)\">{gname}</button>"
        )
    fact_check_sidebar_html.append("</div>")
    fact_check_sidebar_html = "".join(fact_check_sidebar_html)

    # Tabs + tab-specific CSS
    head_extra_css = """<style>
      .tabs {
        width: 90%;
        margin: 0 auto 16px;
        overflow: hidden;
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 10px;
      }
      .tabs button {
        background: transparent;
        float: left;
        border: none;
        outline: none;
        cursor: pointer;
        padding: 12px 16px;
        transition: .2s ease;
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
      }
      .tabs button:hover { background: #eef4ff; }
      .tabs button.active { background: #e7f0ff; }

      .tabcontent { display: none; }
      .tabcontent.active { display: block; }
    </style>"""

    # Page body
    body_inner = f"""
  {fact_check_sidebar_html}
  <div class="container">
    <div class="tabs">
      <button class="tablinks" id="defaultTab" onclick="openTab(event, 'knowledgeGraph')">Knowledge Graph</button>
      <button class="tablinks" onclick="openTab(event, 'dbResults')">Database Search Results</button>
      <button class="tablinks" onclick="openTab(event, 'gptResponse')">GPT Answer: Hypothesis</button>
      <button id="factCheckTab" class="tablinks" onclick="openTab(event, 'factCheck')">Fact Check</button>
    </div>
    <div class="textbox">
      <div id="knowledgeGraph" class="tabcontent">{knowledge_graph_html}</div>
      <div id="dbResults" class="tabcontent">{tables_html}</div>
      <div id="gptResponse" class="tabcontent">{gpt_response}</div>
      <div id="factCheck" class="tabcontent">{fact_check_html}</div>
    </div>
  </div>"""

    # Small amount of page-specific JS for tab switching + sidebar navigation
    body_extra_js = """<script>
      function openTab(evt, tabName) {
        var tabcontents = document.getElementsByClassName("tabcontent");
        for (var i = 0; i < tabcontents.length; i++) {
          tabcontents[i].style.display = "none";
          tabcontents[i].classList.remove("active");
        }
        var tablinks = document.getElementsByClassName("tablinks");
        for (var i = 0; i < tablinks.length; i++) {
          tablinks[i].classList.remove("active");
        }
        document.getElementById(tabName).style.display = "block";
        document.getElementById(tabName).classList.add("active");
        if (evt) { evt.currentTarget.classList.add("active"); }
        if (tabName === 'factCheck') {
          document.dispatchEvent(new Event('myTabShown'));
        }
      }

      function goToFactCheckAndScroll(geneId, button) {
        var buttons = document.querySelectorAll("#factCheckSidebar .sidebar-button");
        buttons.forEach(function(btn) { btn.classList.remove("active"); });
        if (button) { button.classList.add("active"); }
        var factCheckTab = document.getElementById("factCheckTab");
        if (factCheckTab) { factCheckTab.click(); } else { openTab(null, 'factCheck'); }
        setTimeout(function() {
          var element = document.getElementById(geneId);
          if (element) element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 280);
      }

      window.onload = function() {
        var defaultTab = document.getElementById("defaultTab");
        if (defaultTab) defaultTab.click();
      };
    </script>"""

    return _html_shell(body_inner,
                       title="Skill 1 — lncRNA–RBP Inference",
                       head_extra_css=head_extra_css,
                       body_extra_js=body_extra_js,
                       include_cytoscape=True)
# ============================================================================
# 14) MAIN ORCHESTRATION
# ============================================================================

def main():
    # --- Parse CLI arguments first ---
    parser = argparse.ArgumentParser(description="Skill 1: infer candidate lncRNA-binding RBPs and functions.")
    parser.add_argument('--gene_of_interest', required=True, help="Gene of interest")
    parser.add_argument('--user_query', required=True, help="User query")
    parser.add_argument('--reasoning_function', required=True, help="Reasoning / experiment description")
    parser.add_argument('--binding_database', required=True, help="Comma-separated DB list, e.g. NPInter,starBase,RNAInter")
    parser.add_argument('--literature_searching', required=True, help="Items: Similarity,PubMed or any subset separated by ','")
    parser.add_argument('--external_information', required=False, default="", help="External files specification joined by '|'")
    parser.add_argument('--output_file_name', required=False, default="", help="Output file name for HTML report (without extension)")
    parser.add_argument('--quotauser', required=False, default="", help="Id for Google Search")
    parser.add_argument('--fresh', choices=['Y','N'], default='N', help='Y = rerun; N = load snapshot if inputs (including external file content) unchanged')
    parser.add_argument('--entity_type',choices=['lncrna', 'rbp', 'mrna'],default='lncrna',help="gene type：lncrna (default)、rbp or mrna")
    args = parser.parse_args()

    # --- Initialize progress & local DB ---
    AG = AgentProgress(run="Skill1", total_steps=11)
    AG.start_step("Initialize run", stage="prepare", detail=f"gene={args.gene_of_interest}")
    init_local_db()

    # --- Output paths ---
    if args.output_file_name:
        output_filename = f"{args.output_file_name}.html"
    else:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d%H%M%S")
        rand_num = random.randint(1000, 9999)
        output_filename = f"Analysis_{date_str}_{rand_num}.html"

    output_dir = "./temp"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # --- Snapshot resolution ---
    canon_inputs      = _canonical_inputs_p1(args)
    snap_match_path, _ = _locate_matching_snapshot_p1(canon_inputs)
    snap_target_path  = get_snapshot_path_p1_from_canon(canon_inputs)
    use_snapshot      = (args.fresh or "Y").upper() == "N" and bool(snap_match_path)

    # ---------- Helper: extract sections from a previously saved HTML ----------
    def _extract_sections_from_html(html_text: str) -> dict:
        soup = BeautifulSoup(html_text, "html.parser")

        def _inner_html_by_id(div_id: str) -> str:
            tag = soup.find("div", {"id": div_id})
            return tag.decode_contents() if tag else ""

        # Sections
        knowledge_graph_html = _inner_html_by_id("knowledgeGraph")
        tables_html          = _inner_html_by_id("dbResults")
        gpt_response         = _inner_html_by_id("gptResponse")
        fact_check_html      = _inner_html_by_id("factCheck")

        # Gene names (sidebar buttons)
        gene_names = []
        sidebar = soup.find("div", {"id": "factCheckSidebar"})
        if sidebar:
            for btn in sidebar.select("button.sidebar-button"):
                t = (btn.get_text() or "").strip()
                if t:
                    gene_names.append(t)

        return {
            "knowledge_graph_html": knowledge_graph_html or "",
            "tables_html": tables_html or "",
            "gpt_response": gpt_response or "",
            "fact_check_html": fact_check_html or "",
            "gene_names": gene_names
        }

    # ---------- Snapshot mode (replaying with real artifacts) ----------
    if use_snapshot:
        try:
            snap = load_snapshot_p1(snap_match_path)
        except Exception as e:
            AG.error(f"Failed to load snapshot: {e}")
            # Fallback to fresh run below
            use_snapshot = False

    if use_snapshot:
        AG.update("Snapshot matched. Extracting artifacts per step…", stage="snapshot")
        html_from_snap = (snap.get("artifacts") or {}).get("html", "") or ""
        snap_params    = snap.get("params")    or {}
        snap_meta      = snap.get("metadata")  or {}

        if not html_from_snap.strip():
            AG.error("Snapshot contains no HTML artifact; running fresh.")
            use_snapshot = False
        else:
            # ---------------- Step 1: Initialize (write lightweight placeholder) ----------------
            placeholder_body = """
<div class="container">
  <div class="textbox">
    <p style="font-size:1.1em;text-align:center;margin-top:36px;">
      Replaying from snapshot: reconstructing step outputs…
    </p>
  </div>
</div>
"""
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(build_html_page(placeholder_body))
                AG.done("Placeholder page written (snapshot)", path=output_path)
                _maybe_sleep()
            except Exception as e:
                AG.error(f"Failed to write placeholder (snapshot): {e}")

            # Parse sections
            sections = _extract_sections_from_html(html_from_snap)

            # ---------------- Step 2: Query local databases (reconstructed) ----------------
            AG.start_step("Query local databases", stage="db", detail="Restored from snapshot")
            tables_html = sections["tables_html"]
            db_tbl_count = 0
            if tables_html:
                _soup_db = BeautifulSoup(tables_html, "html.parser")
                db_tbl_count = len(_soup_db.find_all("table"))
            AG.done(f"DB tables restored: {db_tbl_count} table(s)")
            _maybe_sleep()

            # ---------------- Step 3: Load external files (reconstructed) ----------------
            AG.start_step("Load external files", stage="files", detail="Restored from snapshot")
            ext_recs = snap_params.get("external_files_records", [])
            ext_found = sum(1 for r in ext_recs if r.get("exists"))
            AG.done(f"External files restored: {ext_found}/{len(ext_recs)}")
            _maybe_sleep()

            # ---------------- Step 4: Run GPT ensemble reasoning (reconstructed) ----------------
            AG.start_step("Run GPT ensemble reasoning", stage="gpt", detail="Restored from snapshot")
            gpt_response = sections["gpt_response"]
            # quick signal: how many candidate blocks?
            cand_blocks = 0
            if gpt_response:
                _soup_gpt = BeautifulSoup(gpt_response, "html.parser")
                cand_blocks = len(_soup_gpt.select("div.gene-entry"))
            AG.done(f"GPT answer restored (Top candidates: ~{cand_blocks})")
            _maybe_sleep()

            # ---------------- Step 5: Build database tables (reconstructed) ----------------
            AG.start_step("Build database tables", stage="html")
            # Already reconstructed as tables_html
            AG.done("Database tables reconstructed")
            _maybe_sleep()

            # ---------------- Step 6: Fact-check gene of interest (reconstructed) ----------------
            AG.start_step("Fact-check gene of interest", stage="search")
            fact_check_html = sections["fact_check_html"]
            subtask_count = 0
            if fact_check_html:
                _soup_fc = BeautifulSoup(fact_check_html, "html.parser")
                subtask_count = len(_soup_fc.select("div.fact-check-subtask"))
            AG.done(f"Fact-check restored (subtasks: {subtask_count})")
            _maybe_sleep()

            # ---------------- Step 7: Evaluate related genes (reconstructed) ----------------
            AG.start_step("Evaluate related genes", stage="search")
            gene_names = sections["gene_names"][:]
            # Fallback to metadata if missing
            goi_from_meta = (snap_meta.get("Gene of Interest") or "").strip()
            if not gene_names and goi_from_meta:
                gene_names = [goi_from_meta]
            if not gene_names:
                # last resort—try to guess from args
                gene_names = [args.gene_of_interest]
            others = [g for g in gene_names if g and g != gene_names[0]]
            N = max(1, len(others))
            for i, rg in enumerate(others, start=1):
                AG.update(f"Related gene {i}/{len(others)}: {rg}", stage="search",
                          perc=40 + int(i / N * 20))
            AG.done(f"Related genes enumerated: {len(others)}")
            _maybe_sleep()

            # ---------------- Step 8: Construct knowledge graph (reconstructed / regenerate) ----
            AG.start_step("Construct knowledge graph", stage="graph")
            knowledge_graph_html = sections["knowledge_graph_html"]
            if not knowledge_graph_html:
                # If snapshot didn't contain the graph tab (older snapshot), regenerate from fact-check.
                goi_label = (goi_from_meta or args.gene_of_interest or gene_names[0]).strip()
                try:
                    knowledge_graph_html = generate_knowledge_graph(fact_check_html, goi_label)
                    AG.update("Knowledge graph regenerated from fact-check", stage="graph")
                except Exception as e:
                    knowledge_graph_html = "<p>Knowledge graph unavailable.</p>"
                    AG.update(f"Graph regeneration failed: {e}", stage="graph")
            AG.done("Graph ready")
            _maybe_sleep()

            # ---------------- Step 9: Assemble final report (rebuild from pieces) -------------
            AG.start_step("Assemble final report", stage="html")
            html_content = compose_report_html(knowledge_graph_html, tables_html, gpt_response, fact_check_html, gene_names)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                AG.done("Report rebuilt from snapshot artifacts", path=output_path)
                _maybe_sleep()
            except Exception as e:
                AG.error(f"Failed to write rebuilt HTML: {e}")
                sys.exit(1)

            # ---------------- Step 10: Export Excel supplement ------------------
            AG.start_step("Export Excel supplement", stage="excel")
            try:
                excel_path = os.path.splitext(output_path)[0] + ".xlsx"
                metadata = dict(snap_meta)
                metadata.update({
                    "Gene of Interest": snap_meta.get("Gene of Interest", args.gene_of_interest),
                    "User Query": args.user_query,
                    "Reasoning Function": args.reasoning_function,
                    "Binding Databases": args.binding_database,
                    "Literature Searching": args.literature_searching,
                    "External Information": args.external_information or "None",
                    "Output HTML": output_path,
                    "Restored From Snapshot": os.path.basename(snap_match_path),
                    "Rebuilt At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                create_excel_supplement_from_html(html_content, excel_path, metadata)
                AG.done("Excel supplement saved", path=excel_path)
            except Exception as e:
                AG.error(f"Excel export failed (snapshot): {e}")

            # ---------------- Step 11: Finish ----------------------------------
            AG.start_step("Finish", stage="done")
            AG.done("All steps completed (snapshot replay)")
            return

    # --------------------------------------------------------------------------
    # Fresh run (unchanged logic, apart from minor robustness tweaks)
    # --------------------------------------------------------------------------

    lit_set = set([x.strip().lower() for x in args.literature_searching.split(",") if x.strip()])
    ENABLE_FAISS  = "similarity" in lit_set
    ENABLE_GOOGLE = "pubmed"     in lit_set

    gene           = args.gene_of_interest
    db_requested   = [x.strip() for x in args.binding_database.split(",") if x.strip()]
    db_requested_upper = [x.upper() for x in db_requested]

    entity_type = getattr(args, 'entity_type', 'lncrna').lower()
    if entity_type == 'rbp' and 'RNAINTER_MRNA' not in db_requested_upper:
        db_requested_upper.append('RNAINTER_MRNA')
    if entity_type == 'mrna':
        db_requested_upper = ['RNAINTER_MRNA']
    
    quotaUser      = args.quotauser

    placeholder_body = """
<div class="container">
  <div class="textbox">
    <p style="font-size:1.2em;text-align:center;margin-top:40px;">
      Your task is accepted. The process will take approximately 10–20 minutes.<br>
      The report will automatically refresh once the process is complete.
    </p>
  </div>
</div>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(build_html_page(placeholder_body))
    AG.done("Placeholder page written", path=output_path)

    # --- Step 2: Query local databases ---
    AG.start_step("Query local databases", stage="db", detail=args.binding_database)
   
    if entity_type == 'lncrna':
        ALL_DB = {
            "NPINTER":       ("NPInter",        [("LncRNA Name",   lambda g: f"%{g}%")]),
            "STARBASE":      ("starBase",       [("LncRNA Name",   lambda g: f"%{g}%")]),
            "RNAINTER":      ("RNAInter",       [("LncRNA Name",   lambda g: f"%{g}%")]),
            "GTEX_TISSUE":   ("GTEx_Tissue",    [("Description",   lambda g: f"%{g}%")]),
        }
    
    elif entity_type == 'rbp':
        ALL_DB = {
            "NPINTER":         ("NPInter",         [("RBP Name",     lambda g: f"%{g}%")]),
            "STARBASE":        ("starBase",        [("RBP Name",     lambda g: f"%{g}%")]),
            "RNAINTER":        ("RNAInter",        [("RBP Name",     lambda g: f"%{g}%")]),
            "RNAINTER_MRNA":   ("RNAInter_mRNA",   [("RBP Name",     lambda g: f"%{g}%")]),
            "GTEX_TISSUE":     ("GTEx_Tissue",     [("Description",  lambda g: f"%{g}%")]),
        }
    
    elif entity_type == 'mrna':
        ALL_DB = {
            "RNAINTER_MRNA": ("RNAInter_mRNA",  [("mRNA Name",     lambda g: f"%{g}%")]),
            "GTEX_TISSUE":   ("GTEx_Tissue",    [("Description",   lambda g: f"%{g}%")]),
        }
    
    else:
        ALL_DB = {
            "NPINTER":       ("NPInter",        [("LncRNA Name",   lambda g: f"%{g}%")]),
            "STARBASE":      ("starBase",       [("LncRNA Name",   lambda g: f"%{g}%")]),
            "RNAINTER":      ("RNAInter",       [("LncRNA Name",   lambda g: f"%{g}%")]),
            "GTEX_TISSUE":   ("GTEx_Tissue",    [("Description",   lambda g: f"%{g}%")]),
        }
    
    db_results = {}
    for key, (table_name, cond) in ALL_DB.items():
        if key == "GTEX_TISSUE" or key in db_requested_upper:
            cols, rows = query_database(gene, table_name, cond)
            db_results[table_name] = {"columns": cols, "data": [dict(zip(cols, r)) for r in rows]}

    rbp_info = {}
    if "NPInter" in db_results:
        rbp_info["NPInter"] = process_NPInter(gene, db_results["NPInter"]["data"])
    if "RNAInter" in db_results:
        rbp_info["RNAInter"] = process_RNAInter(gene, db_results["RNAInter"]["data"])
    if "starBase" in db_results:
        rbp_info["starBase"] = process_starBase(gene, db_results["starBase"]["data"])
    if "RNAInter_mRNA" in db_results:
        rbp_info["RNAInter_mRNA"] = process_RNAInter_mRNA(gene, db_results["RNAInter_mRNA"]["data"])
    rbp_info = deduplicate_category(rbp_info, ["NPInter", "RNAInter", "starBase", "RNAInter_mRNA"])
    
    if "GTEx_Tissue" in db_results and db_results["GTEx_Tissue"].get("data"):
        goi_norm = (gene or "").strip()
        exact_rows = [
            row for row in db_results["GTEx_Tissue"]["data"]
            if str(row.get("Description", "")).strip() == goi_norm
        ]
        db_results["GTEx_Tissue"]["data"] = exact_rows
        
    if "GTEx_Tissue" in db_results:
        gtex_proc = process_GTEx_Tissue(db_results["GTEx_Tissue"]["data"])
        db_results["GTEx_Tissue"]["data"] = gtex_proc["data"]
        db_results["GTEx_Tissue"]["columns"] = gtex_proc["columns"]

    output = {"RBP information": rbp_info}
    AG.done("DB results aggregated")

    # --- Step 3: Load external files ---
    AG.start_step("Load external files", stage="files",
                  detail=(args.external_information or "No File Input"))
    append_text = process_external_information(args.external_information)
    AG.done("External info loaded")

    # --- Step 4: Run GPT ensemble reasoning ---
    AG.start_step("Run GPT ensemble reasoning", stage="gpt")
    gpt_response = call_gpt_for_multistep_regulatory_mechanisms(
        output, gene, args.reasoning_function, append_text, args.user_query
    )
    AG.done("Reasoning complete")

    # --- Step 5: Build database tables (HTML fragments) ---
    AG.start_step("Build database tables", stage="html")
    tables_html = ""
    for db_name, result in db_results.items():
        columns = result.get("columns", [])
        data = result.get("data", [])
        tables_html += f"<h3>{db_name}</h3>\n"
        tables_html += "<div class='table-container gtex-container'>\n" if db_name == "GTEx_Tissue" else "<div class='table-container'>\n"
        if data:
            # Helper (robust NPInter header match)
            def _is_npinter_interaction_col(name: str) -> bool:
                n = (name or "").strip().lower()
                n = re.sub(r'[\s_-]+', ' ', n)
                # match "summary" alone or "interaction summary" and common variants
                return (('interaction' in n and 'summary' in n) or
                        n in {'summary', 'interaction summary',
                              'interaction detail', 'interaction details'})
    
            tables_html += "<table>\n<thead>\n<tr>"
            for col in columns:
                th_class = ''
                if db_name == "NPInter" and _is_npinter_interaction_col(col):
                    th_class = ' class="col-npinter-interaction"'
                tables_html += f"<th{th_class}>{col}</th>"
            tables_html += "</tr>\n</thead>\n<tbody>\n"
            for row in data:
                tables_html += "<tr>"
                for col in columns:
                    value = row.get(col, "")
                    td_class = ''
                    if db_name == "NPInter" and _is_npinter_interaction_col(col):
                        td_class = ' class="col-npinter-interaction"'
                    tables_html += f"<td{td_class}><div class=\"cell-content\">{value}</div></td>"
                tables_html += "</tr>\n"
            tables_html += "</tbody>\n</table>\n"
        else:
            tables_html += "<p>No data found.</p>\n"
        tables_html += "</div>\n<hr>\n"

    # External file table(s)
    if args.external_information:
        file_specs = [p.strip() for p in args.external_information.split("|") if p.strip()]
        for spec in file_specs:
            file_name = spec.split("(", 1)[0].strip()
            abs_path = os.path.join(EXTERNAL_BASE_DIR, file_name)
            tables_html += f"<h3>{file_name}</h3>\n<div class='table-container'>\n"
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    append_content = f.read()
            except Exception as e:
                tables_html += f"<p>Error reading append file '{file_name}': {e}</p>\n</div>\n<hr>\n"
                append_content = None

            if append_content:
                lines = append_content.rstrip().splitlines()
                if len(lines) < 2:
                    tables_html += "<p>No data found in append file.</p>\n</div>\n<hr>\n"
                else:
                    headers = lines[0].split("\t")
                    tables_html += "<table>\n<thead>\n<tr>"
                    tables_html += "".join(f"<th>{h}</th>" for h in headers)
                    tables_html += "</tr>\n</thead>\n<tbody>\n"
                    for line in lines[1:]:
                        cols_line = line.split("\t")
                        tables_html += "<tr>" + "".join(f"<td><div class=\"cell-content\">{c}</div></td>" for c in cols_line) + "</tr>\n"
                    tables_html += "</tbody>\n</table>\n</div>\n<hr>\n"
            else:
                tables_html += "</div>\n<hr>\n"

    data_analysis_html = ""  # kept empty per original logic
    AG.done("DB tables built")

    # --- Step 6: Fact-check gene of interest ---
    AG.start_step("Fact-check gene of interest", stage="search")
    fact_check_html, gene_names = Fact_Check(
        gpt_response, tables_html, data_analysis_html, gene,
        args.reasoning_function, "human", "", quotaUser, ENABLE_FAISS, ENABLE_GOOGLE
    )
    AG.done("GOI fact-check built")

    # --- Step 7: Evaluate related genes ---
    AG.start_step("Evaluate related genes", stage="search")
    others = [g for g in gene_names if g != gene]
    N = max(1, len(others))
    for i, related_gene in enumerate(others, start=1):
        AG.update(f"Related gene {i}/{len(others)}: {related_gene}",
                  stage="search", perc=40 + int(i / N * 20))
    AG.done("Related genes evaluated")

    # --- Step 8: Construct knowledge graph ---
    AG.start_step("Construct knowledge graph", stage="graph")
    knowledge_graph_html = generate_knowledge_graph(fact_check_html, gene)
    AG.done("Graph ready")

    # --- Step 9: Assemble final report ---
    AG.start_step("Assemble final report", stage="html")
    html_content = compose_report_html(knowledge_graph_html, tables_html, gpt_response, fact_check_html, gene_names)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        AG.done("Report saved", path=output_path)
    except Exception:
        AG.done("Report failed to save")
        sys.exit(1)

    # --- Step 10: Export Excel Supplement ---
    AG.start_step("Export Excel supplement", stage="excel")
    try:
        excel_path = os.path.splitext(output_path)[0] + ".xlsx"
        metadata = {
            "Gene of Interest": args.gene_of_interest,
            "User Query": args.user_query,
            "Reasoning Function": args.reasoning_function,
            "Binding Databases": args.binding_database,
            "Literature Searching": args.literature_searching,
            "External Information": args.external_information or "None",
            "Output HTML": output_path,
            "Generated At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        create_excel_supplement_from_html(html_content, excel_path, metadata)
        AG.done("Excel supplement saved", path=excel_path)
    except Exception as e:
        AG.error(f"Excel export failed: {e}")

    # --- Step 11: Finish & save snapshot for next time ---
    try:
        meta = {
            "Gene of Interest": args.gene_of_interest,
            "User Query": args.user_query,
            "Reasoning Function": args.reasoning_function,
            "Binding DB": args.binding_database,
            "Literature Searching": args.literature_searching,
            "Append Files (by content)": f"{len(canon_inputs.get('external_files_sha256_list', []))} files",
            "Output HTML": output_path,
            "Generated At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_snapshot_p1(snap_target_path, args, html_content, meta)
    except Exception as e:
        log(f"[WARN] Snapshot save failed: {e}")

    AG.start_step("Finish", stage="done")
    AG.done("All steps completed")

# ============================================================================
# 15) EXCEL SUPPLEMENT EXPORT (NEW)
# ============================================================================


def create_excel_supplement_from_html(html_text: str, excel_path: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    Create a multi-sheet Excel workbook that captures all major information
    presented in the final HTML report, formatted as a scientific supplementary file.

    NEW (ordering): Tabs are written in this exact order:
      1) Readme
      2) Contents
      3) KG_Nodes
      4) KG_Edges
      5) Top candidates
      6) Factcheck_evidence
      7) All database results (S1_DB_*), preserving their relative order

    Other improvements preserved from the prior version:
      • Robust evidence parsing and canonicalization of Section/Aspect labels
      • Deduplication of repeated evidence lines
      • Defensive HTML→table parsing, column wrapping, auto width, freeze panes, filters
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # ---------------------------- Helpers ------------------------------------
    def _norm(s: Any) -> str:
        return re.sub(r'\s+', ' ', str(s).strip()) if s is not None else ""

    def _sheet_name_safe(name: str, existing: set) -> str:
        base = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        base = re.sub(r'[\[\]\*\?:]', '_', base)
        base = base[:31] if len(base) > 31 else base
        if base not in existing:
            existing.add(base)
            return base
        for i in range(2, 1000):
            candidate = (base[:31 - len(f"_{i}")]) + f"_{i}"
            if candidate not in existing:
                existing.add(candidate)
                return candidate
        raise RuntimeError("Too many duplicate sheet names")

    def _bs_table_to_df(tbl) -> pd.DataFrame:
        headers = [ _norm(th.get_text()) for th in tbl.select("thead th") ]
        if not headers:
            first_tr = tbl.find("tr")
            if first_tr:
                headers = [ _norm(x.get_text()) for x in first_tr.find_all(["th","td"]) ]
        rows = []
        body_trs = tbl.select("tbody tr")
        if not body_trs:
            trs = tbl.find_all("tr")
            trs = trs[1:] if headers and trs else trs
        else:
            trs = body_trs
        for tr in trs:
            cols = tr.find_all(["td","th"])
            vals = []
            for td in cols:
                vals.append(_norm(td.get_text(" ")))
            if headers and len(vals) != len(headers):
                if len(vals) < len(headers):
                    vals += [""] * (len(headers) - len(vals))
                else:
                    vals = vals[:len(headers)]
            rows.append(vals)
        if not headers:
            if rows:
                headers = [f"Col_{i+1}" for i in range(max(len(r) for r in rows))]
            else:
                headers = ["Column"]
        df = pd.DataFrame(rows, columns=headers)
        return df

    # Excel writer -------------------------------------------------------------
    def _get_writer(path: str):
        try:
            writer = pd.ExcelWriter(path, engine="openpyxl")
            return writer, "openpyxl"
        except Exception:
            writer = pd.ExcelWriter(path, engine="xlsxwriter")
            return writer, "xlsxwriter"

    def _apply_format_openpyxl(writer, sheet_name: str, title: str, wrap_cols: list[str] | None = None):
        if not _HAS_OPENPYXL:
            return
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A3"
        ws.cell(row=1, column=1, value=title)
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=max(1, ws.max_column))
        title_cell = ws.cell(row=1, column=1)
        title_cell.font = Font(bold=True, size=12)
        title_cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        header_font = Font(bold=True)
        thin = Side(style="thin", color="DDDDDD")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        for col_idx in range(1, ws.max_column + 1):
            c = ws.cell(row=2, column=col_idx)
            c.fill = header_fill
            c.font = header_font
            c.border = border
            c.alignment = Alignment(wrap_text=True, vertical="center")
        max_width = 60
        min_width = 8
        for col_idx in range(1, ws.max_column + 1):
            col_letter = get_column_letter(col_idx)
            texts = []
            texts.append(str(ws.cell(row=2, column=col_idx).value or ""))
            for r in range(3, min(ws.max_row, 3 + 500)):
                texts.append(str(ws.cell(row=r, column=col_idx).value or ""))
            width = min(max(int(max(len(t) for t in texts) * 0.9) + 2, min_width), max_width)
            ws.column_dimensions[col_letter].width = width
        wrap_cols = set(wrap_cols or [])
        if wrap_cols:
            name_to_idx = {str(ws.cell(row=2, column=c).value): c for c in range(1, ws.max_column+1)}
            for name in wrap_cols:
                idx = name_to_idx.get(name)
                if idx:
                    for r in range(3, ws.max_row + 1):
                        ws.cell(row=r, column=idx).alignment = Alignment(wrap_text=True, vertical="top")
        ws.auto_filter.ref = ws.dimensions

    def _apply_format_xlsxwriter(writer, sheet_name: str, title: str, df: pd.DataFrame,
                                 wrap_cols: list[str] | None = None):
        wb  = writer.book
        ws  = writer.sheets[sheet_name]
        fmt_title = wb.add_format({"bold": True, "font_size": 12, "align": "left", "valign": "vcenter", "text_wrap": True})
        fmt_header = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "bottom": 1})
        fmt_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        ws.merge_range(0, 0, 0, max(0, df.shape[1] - 1), title, fmt_title)
        for c in range(df.shape[1]):
            ws.write(1, c, df.columns[c], fmt_header)
        ws.freeze_panes(2, 0)
        max_width = 60
        min_width = 8
        for c in range(df.shape[1]):
            texts = [str(df.columns[c])]
            sample = df.iloc[:min(len(df), 500), c].astype(str).tolist()
            texts += sample
            width = min(max(int(max(len(t) for t in texts) * 0.9) + 2, min_width), max_width)
            ws.set_column(c, c, width)
        wrap_cols = set(wrap_cols or [])
        for c, name in enumerate(df.columns.tolist()):
            if name in wrap_cols:
                ws.set_column(c, c, None, fmt_wrap)
        ws.autofilter(1, 0, max(1, len(df.index)) + 1, max(0, df.shape[1]-1))

    def _write_df(writer, df: pd.DataFrame, sheet_name: str, title: str,
                  wrap_cols: list[str] | None = None):
        df = df.copy().fillna("")
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        if writer.engine == "openpyxl":
            _apply_format_openpyxl(writer, sheet_name, title, wrap_cols)
        else:
            _apply_format_xlsxwriter(writer, sheet_name, title, df, wrap_cols)

    # Canonicalization helpers for Section & Aspect ---------------------------
    SECTION_MAP = {
        "abstract": "Abstract",
        "background": "Abstract",
        "introduction": "Introduction",
        "methods": "Methods",
        "materials and methods": "Methods",
        "materials & methods": "Methods",
        "results": "Results",
        "discussion": "Discussion",
        "conclusions": "Discussion",
        "conclusion": "Discussion",
        "case report": "Results",
        "other": "Other",
        "unspecified": "Other",
        "text body": "Other",
    }
    KNOWN_ASPECTS = {
        "general information": "General Information",
        "gene function under condition": "Gene Function under condition",
        "subcellular localization": "Subcellular Localization",
        "subcellular localisation": "Subcellular Localization",
    }

    def canonicalize_section(raw: str, fallback_from_text: str = "") -> str:
        r = (raw or "").strip().lower()
        r = re.sub(r'[-–—]+', ' ', r)
        r = re.sub(r'\s+', ' ', r)
        for key, canon in SECTION_MAP.items():
            if key in r:
                return canon
        t = (fallback_from_text or "").lower()
        for key, canon in SECTION_MAP.items():
            if key != "other" and key in t:
                return canon
        return "Results"

    def canonicalize_aspect(raw: str) -> str:
        r = (raw or "").strip().lower()
        r = re.sub(r'\s+', ' ', r)
        for key, canon in KNOWN_ASPECTS.items():
            if key in r:
                return canon
        return (raw or "").strip() or "General Information"

    def parse_evidence_line(line: str) -> tuple[str, str, str]:
        s = _norm(line)
        src = ""
        m = re.search(r'\[Source:\s*([^\]]+?)\s*\]\s*$', s, flags=re.I)
        if m:
            src = _norm(m.group(1))
            s = s[:m.start()].strip()
        s = re.sub(r'^\s*\d+[\.\)]\s*', '', s)
        section_raw = ""
        m = re.search(r'Section\s*:\s*-?\s*([A-Za-z &/]+?)(?:\s*-|:|\s)\s*', s, flags=re.I)
        if m:
            section_raw = _norm(m.group(1))
            s = s[m.end():].strip()
        else:
            m2 = re.match(r'^(Abstract|Introduction|Results|Discussion|Methods|Conclusion|Conclusions|Background)\s*[-–—:]?\s*(.*)$', s, flags=re.I)
            if m2:
                section_raw = _norm(m2.group(1))
                s = _norm(m2.group(2))
        section = canonicalize_section(section_raw, fallback_from_text=s)
        evidence = s
        return section, evidence, src

    # ----------------------- Collect tables into DataFrames -------------------
    sheet_defs: list[tuple[str, str, pd.DataFrame, list[str]]] = []
    used_names = set()

    # S1_DB_*: Database Search Results
    db_container = soup.find("div", {"id": "dbResults"})
    if db_container:
        for h3 in db_container.find_all("h3"):
            section_title = _norm(h3.get_text())
            tbl = h3.find_next("table")
            if not tbl:
                continue
            df = _bs_table_to_df(tbl)
            sname = _sheet_name_safe(f"S1_DB_{section_title[:20]}", used_names)
            title = f"Table — Database: {section_title}"
            wrap_cols = [c for c in df.columns if any(x in c.lower() for x in ["definition","explanation","notes","description","mechanism"])]
            sheet_defs.append((sname, title, df, wrap_cols))

    # GPT Top candidates
    gpt_container = soup.find("div", {"id": "gptResponse"})
    if gpt_container:
        entries = []
        for block in gpt_container.find_all("div", {"class": "gene-entry"}):
            rank_tag = block.find(["h2","h3"])
            rank = _norm(rank_tag.get_text()) if rank_tag else ""
            p = block.find("p") or block
            def _fetch_value(label: str) -> str:
                strong = p.find("strong", string=lambda s: s and label in s)
                if not strong:
                    return ""
                vals = []
                for sib in strong.next_siblings:
                    if getattr(sib, "name", None) == "br":
                        break
                    if getattr(sib, "name", None) == "strong":
                        break
                    vals.append(_norm(getattr(sib, "get_text", lambda *a, **k: str(sib))()))
                return _norm(" ".join(vals))
            gene_name = _fetch_value("Gene Name")
            hit_count = _fetch_value("Hit Count")
            mech      = _fetch_value("Potential Mechanism")
            source    = _fetch_value("Information Source")
            if any([gene_name, hit_count, mech, source, rank]):
                entries.append({
                    "Rank / Candidate": rank,
                    "Gene Name": gene_name,
                    "Hit Count": hit_count,
                    "Potential Mechanism": mech,
                    "Information Source": source
                })
        if entries:
            df = pd.DataFrame(entries)
            sname = _sheet_name_safe("S_GPT_TopCandidates", used_names)
            title = "GPT‑Inferred Top Candidates"
            sheet_defs.append((sname, title, df, ["Potential Mechanism", "Information Source"]))

    # Knowledge Graph (Nodes/Edges) from Fact Check subtasks
    fact_container = soup.find("div", {"id": "factCheck"})
    nodes_rows, edges_rows = [], []
    if fact_container:
        for sub in fact_container.find_all("div", {"class": "fact-check-subtask"}):
            sub_title_tag = sub.find("h3")
            sub_title = _norm(sub_title_tag.get_text()) if sub_title_tag else "Subtask"
            nodes_tbl = sub.find("table", {"id": "kg-nodes"})
            if nodes_tbl:
                df_nodes = _bs_table_to_df(nodes_tbl)
                if not df_nodes.empty:
                    df_nodes.insert(0, "Subtask", sub_title)
                    nodes_rows.append(df_nodes)
            edges_tbl = sub.find("table", {"id": "kg-edges"})
            if edges_tbl:
                df_edges = _bs_table_to_df(edges_tbl)
                if not df_edges.empty:
                    df_edges.insert(0, "Subtask", sub_title)
                    edges_rows.append(df_edges)

    if nodes_rows:
        df_nodes_all = pd.concat(nodes_rows, ignore_index=True)
        sname_nodes = _sheet_name_safe("S_KG_Nodes", used_names)
        title_nodes = "Knowledge Graph — Nodes (aggregated)"
        sheet_defs.append((sname_nodes, title_nodes, df_nodes_all, ["Definition"]))

    if edges_rows:
        df_edges_all = pd.concat(edges_rows, ignore_index=True)
        sname_edges = _sheet_name_safe("S_KG_Edges", used_names)
        title_edges = "Knowledge Graph — Edges (aggregated)"
        sheet_defs.append((sname_edges, title_edges, df_edges_all, ["Explanation", "Node_in_Edge"]))

    # Fact Check evidence (flattened) — robust & dedup
    evidence_entries = []

    def _collect_evidence(context_div, context_label: str):
        if not context_div:
            return
        for h4 in context_div.find_all("h4", string=lambda s: s and "GPT Evaluated Evidence for" in s):
            h4_text = _norm(h4.get_text(" "))
            m_aspect = re.search(r'GPT\s+Evaluated\s+Evidence\s+for\s+(.+)$', h4_text, flags=re.I)
            aspect_raw = _norm(m_aspect.group(1)) if m_aspect else ""
            aspect = canonicalize_aspect(aspect_raw)
            p = h4.find_previous(lambda tag: tag.name == "p" and tag.find("strong", string=lambda s: s and "Search Query" in s))
            query = ""
            if p:
                strong = p.find("strong", string=lambda s: s and "Search Query" in s)
                q_parts = []
                for sib in (strong.next_siblings if strong else []):
                    if getattr(sib, "name", None) == "br":
                        break
                    q_parts.append(_norm(getattr(sib, "get_text", lambda *a, **k: str(sib))()))
                query = _norm(" ".join(q_parts))
            pre = h4.find_next("pre")
            if not pre:
                continue
            lines = [l for l in pre.get_text("\n").splitlines() if _norm(l)]
            for l in lines:
                section, evidence, src = parse_evidence_line(l)
                if not evidence:
                    continue
                evidence_entries.append({
                    "Context": context_label,
                    "Aspect": aspect,
                    "Search Query": query,
                    "Section": section,
                    "Evidence": evidence,
                    "Source": src
                })

    if fact_container:
        goi_div = fact_container.find("div", {"class": "fact-check-gene-of-interest"})
        _collect_evidence(goi_div, "Gene of Interest")
        for sub in fact_container.find_all("div", {"class": "fact-check-subtask"}):
            sub_title = sub.find("h3")
            sub_label = _norm(sub_title.get_text()) if sub_title else "Subtask"
            _collect_evidence(sub, sub_label)

    if evidence_entries:
        df_evi = pd.DataFrame(evidence_entries)
        df_evi["Section"] = df_evi.apply(
            lambda r: canonicalize_section(r.get("Section", ""), r.get("Evidence", "")),
            axis=1
        )
        df_evi = df_evi.drop_duplicates(subset=["Context", "Aspect", "Evidence", "Source"], keep="first").reset_index(drop=True)
        sname = _sheet_name_safe("S_FactCheck_Evidence", used_names)
        title = "GPT‑Evaluated Evidence (flattened, deduplicated)"
        sheet_defs.append((sname, title, df_evi, ["Evidence", "Source", "Search Query"]))

    # README & (to-be-built) Contents -----------------------------------------
    readme_rows = []
    if metadata:
        for k, v in metadata.items():
            readme_rows.append({"Key": k, "Value": str(v)})
    readme_df = pd.DataFrame(readme_rows) if readme_rows else pd.DataFrame({"Key": [], "Value": []})

    # ----------------------------- Order & Write ------------------------------
    # Desired relative priority for data sheets (lower rank = earlier)
    # We will insert "Readme" and "Contents" in front of these.
    def _rank_for_data_sheet(sname: str) -> int:
        n = sname.lower()
        if "kg_nodes" in n:              return 2
        if "kg_edges" in n:              return 3
        if "gpt_topcandidates" in n:     return 4
        if "factcheck_evidence" in n:    return 5
        if n.startswith("s1_db_"):       return 6
        return 99  # anything else goes to the end

    # Keep stable order among same-rank items
    enumerated = list(enumerate(sheet_defs))
    ordered_data = [item for _, item in sorted(enumerated, key=lambda t: (_rank_for_data_sheet(t[1][0]), t[0]))]

    # Build the Contents sheet from the final order (including Readme, excluding Contents itself)
    contents_rows = [{"Sheet": "Readme", "Title": "Supplementary File — Provenance"}]
    contents_rows += [{"Sheet": s, "Title": t} for (s, t, _df, _wrap) in ordered_data]
    contents_df = pd.DataFrame(contents_rows, columns=["Sheet", "Title"])

    # Final sequence to write
    final_defs: list[tuple[str, str, pd.DataFrame, list[str]]] = []
    final_defs.append(("Readme",   "Supplementary File — Provenance", readme_df, ["Value"]))
    final_defs.append(("Contents", "Supplementary File — Contents",   contents_df, []))
    final_defs.extend(ordered_data)

    writer, engine = _get_writer(excel_path)
    try:
        for (sname, title, df, wrap_cols) in final_defs:
            _write_df(writer, df, sname, title, wrap_cols)
        writer.close()
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        raise

    return excel_path


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
