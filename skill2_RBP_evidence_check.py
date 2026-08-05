#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 2 — RBP evidence check with experimental extraction and a pre-HTML placeholder.
NEW:
  • Snapshot system (./snapshot) keyed by (query, gene), independent of output name.
  • --fresh flag: Y = rerun everything; N = load snapshot, skip all search/API/LLM.
  • Excel supplementary export (same base path as HTML).

Examples:
  # Fresh run from scratch (creates/updates the snapshot)
  python ./skill2_RBP_evidence_check.py \
      --query  "HDLBP function in liver" \
      --gene   HDLBP \
      --output HDLBP_skill2_01 \
      --fresh  Y

  # Reuse snapshot for same (query, gene), skip all external calls
  python ./skill2_RBP_evidence_check.py \
      --query  "HDLBP function in liver" \
      --gene   HDLBP \
      --output HDLBP_skill2_02 \
      --fresh  N
"""

# ============================================================================
# 1) PARAMETERS (edit these as needed)
#    NOTE: We intentionally place tunables first per your instruction.
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

# A non-empty OpenAI key takes priority; otherwise keep using the original
# Azure deployments. Environment variables override profile.json.
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or _CFG.get("OPENAI_API_KEY") or "").strip()
OPENAI_GPT_MODEL = (os.getenv("OPENAI_GPT_MODEL") or _CFG.get("OPENAI_GPT_MODEL") or "gpt-4o").strip()
OPENAI_REASONING_MODEL = (
    os.getenv("OPENAI_REASONING_MODEL")
    or _CFG.get("OPENAI_REASONING_MODEL")
    or "o4-mini"
).strip()
USE_OPENAI = bool(OPENAI_API_KEY)

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

# Evidence filter
CITATION_MIN = 5  # minimum citation count to keep an article

# Verbose debug logging
VERBOSE = False

# Original Azure OpenAI clients (fallback)
AZURE_GPT4O_ENDPOINT = (_CFG.get("AZURE_GPT4O_ENDPOINT") or "").strip()
AZURE_GPT4O_KEY      = (_CFG.get("AZURE_GPT4O_API_KEY") or "").strip()
AZURE_GPT4O_API_VER  = "2024-08-01-preview"

AZURE_O4MINI_ENDPOINT = (_CFG.get("AZURE_O4MINI_ENDPOINT") or "").strip()
AZURE_O4MINI_KEY      = (_CFG.get("AZURE_O4MINI_API_KEY") or "").strip()
AZURE_O4MINI_API_VER  = "2025-01-01-preview"

# Entrez email (required by NCBI)
ENTREZ_EMAIL = (_CFG.get("ENTREZ_EMAIL") or "you@example.com").strip()

# ============================================================================
# 2) IMPORTS
# ============================================================================

import argparse
import concurrent.futures as cf
import html as _html
import json
import os
import pickle
import re
import time
import gzip
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import faiss
import numpy as np
import requests
from Bio import Entrez
from bs4 import BeautifulSoup
from openai import OpenAI, AzureOpenAI
from sentence_transformers import SentenceTransformer

# pandas + Excel styling helpers (aligned with Skill 1 style)
import pandas as pd
try:
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    _HAS_OPENPYXL = True
except Exception:
    _HAS_OPENPYXL = False
# --- Optional tiny delay per step in snapshot replay (seconds) ---
SNAPSHOT_REPLAY_DELAY = float(
    os.getenv(
        "SKILL2_SNAPSHOT_REPLAY_DELAY",
        os.getenv("P2_SNAPSHOT_REPLAY_DELAY", os.getenv("P1_SNAPSHOT_REPLAY_DELAY", "0.5")),
    )
)

def _maybe_sleep():
    """Sleep a tiny bit during snapshot replay to smooth UI updates."""
    if SNAPSHOT_REPLAY_DELAY > 0:
        time.sleep(SNAPSHOT_REPLAY_DELAY)
# ============================================================================
# 3) PATHS, GLOBAL OBJECTS, CLIENTS
# ============================================================================

# File system layout
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "database")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH  = os.path.join(DATA_DIR, "meta.pkl")

# NEW: Snapshot directory and version tag
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "snapshot")
SNAPSHOT_VERSION = "P2SNAPv1"  # kept for snapshot-format compatibility
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Lazy-loaded embedding model (so the pre-HTML can be written quickly first)
_embedding_model = None
def get_embedding_model():
    """Load the sentence-transformers model at first use."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

# Provider-aware clients; all later Chat Completions calls retain their original
# request and response handling.
if USE_OPENAI:
    client_gpt = OpenAI(api_key=OPENAI_API_KEY)
    client2 = OpenAI(api_key=OPENAI_API_KEY)
    GPT_MODEL = OPENAI_GPT_MODEL
    REASONING_MODEL = OPENAI_REASONING_MODEL
    LLM_PROVIDER = "openai"
else:
    missing = [
        name for name, value in (
            ("AZURE_GPT4O_ENDPOINT", AZURE_GPT4O_ENDPOINT),
            ("AZURE_GPT4O_API_KEY", AZURE_GPT4O_KEY),
            ("AZURE_O4MINI_ENDPOINT", AZURE_O4MINI_ENDPOINT),
            ("AZURE_O4MINI_API_KEY", AZURE_O4MINI_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "[Config] Set OPENAI_API_KEY, or provide the Azure fallback values: "
            + ", ".join(missing)
        )
    client_gpt = AzureOpenAI(
        azure_endpoint=AZURE_GPT4O_ENDPOINT,
        api_key=AZURE_GPT4O_KEY,
        api_version=AZURE_GPT4O_API_VER,
    )
    client2 = AzureOpenAI(
        azure_endpoint=AZURE_O4MINI_ENDPOINT,
        api_key=AZURE_O4MINI_KEY,
        api_version=AZURE_O4MINI_API_VER,
    )
    GPT_MODEL = "gpt-4o"
    REASONING_MODEL = "o4-mini"
    LLM_PROVIDER = "azure"

# Entrez settings
Entrez.email = ENTREZ_EMAIL

# ============================================================================
# 4) UTILITIES (logging, progress, text cleaning, helpers)
# ============================================================================

# ───────────────── AgentProgress: unified progress emitter ─────────────────
import json as _json, sys as _sys, time as _time

class AgentProgress:
    """
    Emit compact, GUI-parsable progress events to stdout.
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
        print(s, flush=True)

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
# ───────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)

def log_debug(msg: str) -> None:
    if VERBOSE:
        print(f"[DEBUG] {msg}", flush=True)

def clean_json_response(raw: str) -> str:
    """Extract a valid JSON array/object from an LLM response."""
    if not raw:
        log_debug("[CLEAN_JSON] Empty response received.")
        return "[]"
    s = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r'(\[.*\]|\{.*\})', s, re.DOTALL)
    return m.group(0).strip() if m else "[]"

def extract_relevant_sentences(text: str, gene: str) -> str:
    """Return sentences that mention the gene symbol."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant = [s for s in sentences if gene.lower() in s.lower()]
    return " ".join(relevant) if relevant else "No Relevant sentence."

# ============================================================================
# 4.1) SNAPSHOT HELPERS
# ============================================================================

def _slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r'\s+', ' ', (s or '').strip())
    s = re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()
    return s[:maxlen] or "na"

def _canonical_inputs(query: str, gene: str) -> Dict[str, str]:
    """Canonicalize user inputs for stable snapshot keys."""
    return {
        "query": re.sub(r'\s+', ' ', (query or '').strip()),
        "gene": (gene or '').strip(),
    }

def _snapshot_key(canon_inputs: Dict[str, str]) -> str:
    payload = {"v": SNAPSHOT_VERSION, "inputs": canon_inputs}
    j = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:16]

def get_snapshot_path(query: str, gene: str) -> str:
    ci = _canonical_inputs(query, gene)
    key = _snapshot_key(ci)
    fname = f"{SNAPSHOT_VERSION}_{key}__gene={_slugify(ci['gene'])}__query={_slugify(ci['query'])}.json.gz"
    return os.path.join(SNAPSHOT_DIR, fname)

def save_snapshot(path: str,
                  query: str,
                  gene: str,
                  results: List[Dict[str, Any]],
                  statistics: Dict[str, Any] | None,
                  summaries: Dict[str, str] | None) -> None:
    """Write a compressed JSON snapshot to disk."""
    canon = _canonical_inputs(query, gene)
    snap = {
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": canon,                       # only user inputs, per requirement
        "runtime_meta": {                      # for provenance / debugging
            "python": sys.version.split()[0],
            "citation_min": CITATION_MIN,
        },
        "results": results or [],
        "statistics": statistics or {},
        "summaries": summaries or {},
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)
    log(f"[SNAPSHOT] Saved → {path}")

def load_snapshot(path: str) -> Dict[str, Any]:
    """Load snapshot (returns dict with keys: results, statistics, summaries, params, created_at, runtime_meta)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    # minimal sanity
    data.setdefault("results", [])
    data.setdefault("statistics", {})
    data.setdefault("summaries", {})
    return data

def split_snapshot_for_replay(snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompose a Skill 2 snapshot into step-wise pieces so we can replay the pipeline
    without any network/API calls. Mirrors the stages in run_pipeline.

    Returns a dict with keys:
      params, results, statistics, summaries,
      candidate_links, citation_map, passed_filter_links,
      articles_meta, evidence_counts
    """
    results: List[Dict[str, Any]] = list(snap.get("results", []))
    params  = dict(snap.get("params", {}))
    stats   = dict(snap.get("statistics", {}) or {})
    sums    = dict(snap.get("summaries", {}) or {})

    cand_links: List[str] = []
    cits: Dict[str, int] = {}
    arts_meta: List[Dict[str, Any]] = {}
    arts_meta = []
    ev_counts: Dict[str, Dict[str, int]] = {}

    for art in results:
        meta = art.get("meta", {}) or {}
        url  = meta.get("pmc_url") or ""
        if not url:
            continue

        cand_links.append(url)
        cits[url] = int(meta.get("citation_count", 0) or 0)

        arts_meta.append({
            "pmc_url": url,
            "pmid": meta.get("pmid", ""),
            "title": meta.get("title", ""),
            "journal": meta.get("journal", ""),
            "date": meta.get("date", ""),
            "citation_count": cits[url],
            "figures_n": len(art.get("figures", []) or []),
        })
        ev_counts[url] = {
            "conclusions_n": len(art.get("conclusions", []) or []),
            "experiments_n": len(art.get("experiments", []) or []),
        }

    passed = [u for u in cand_links if cits.get(u, 0) >= CITATION_MIN]

    return {
        "params": params,
        "results": results,
        "statistics": stats,
        "summaries": sums,
        "candidate_links": cand_links,
        "citation_map": cits,
        "passed_filter_links": passed,
        "articles_meta": arts_meta,
        "evidence_counts": ev_counts,
    }

# ============================================================================
# 5) SEARCH & RETRIEVAL (search terms, FAISS, Google, metadata/citations)
# ============================================================================

def generate_search_terms(query: str, gene: str) -> List[str]:
    """Produce 3 simple templates."""
    return [
        f"{gene} experiment {query}",
        f"{gene} {query} experiment in vivo",
        f"{gene} {query} experiment in vitro",
    ]

def retrieve_top_docs(term: str, gene: str, top_k: int = 2000) -> List[str]:
    """FAISS vector search over local index → return at most 10 PMC links whose text contains the gene."""
    if not Path(INDEX_PATH).exists() or not Path(META_PATH).exists():
        return []
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as fh:
        meta = pickle.load(fh)

    model = get_embedding_model()
    q_vec = model.encode([term]).astype(np.float32)
    D, I = index.search(q_vec, top_k)

    pmc_links: List[str] = []
    seen: Set[str] = set()
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        if gene.lower() not in m.get("full_text", "").lower():
            continue
        url = m.get("pmc_url", "")
        if url and url not in seen:
            seen.add(url)
            pmc_links.append(url)
        if len(pmc_links) >= 10:
            break
    log_debug(f"[FAISS SEARCH] Term: {term}, Retrieved {len(pmc_links)} PMC links.")
    return pmc_links

def google_pmc_search(term: str, limit: int = 10, quota_user: str | None = None) -> List[str]:
    """
    Vertex AI Discovery Engine searchLite (replaces Google CSE) to find PMC links.
    Return: List[str] of PMC URLs (deduplicated), up to `limit`.
    """
    import requests

    log_debug(f"[VERTEX SEARCH] Term: {term}, quotaUser={quota_user}")

    # --- config (same as Skill 1: allow VERTEX_* and legacy GOOGLE_* aliases) ---
    project   = (VERTEX_PROJECT or "").strip()
    engine_id = ((GOOGLE_CX or "") or (VERTEX_ENGINE_ID or "")).strip()
    key       = ((GOOGLE_API_KEY or "") or (VERTEX_API_KEY or "")).strip()
    location  = (VERTEX_LOCATION or "global").strip()
    collection = (VERTEX_COLLECTION or "default_collection").strip()
    serving_config = (VERTEX_SERVING_CONFIG or "default_search").strip()

    if not project:
        raise SystemExit("[Config] Missing VERTEX_PROJECT in profile.json (or env var).")
    if not engine_id:
        raise SystemExit("[Config] Missing VERTEX_ENGINE_ID (or legacy GOOGLE_CX) in profile.json (or env var).")
    if not key:
        raise SystemExit("[Config] Missing VERTEX_API_KEY (or legacy GOOGLE_API_KEY) in profile.json (or env var).")

    base = (
        f"https://discoveryengine.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/collections/{collection}/"
        f"engines/{engine_id}/servingConfigs/{serving_config}:searchLite"
    )

    serving_cfg_path = (
        f"projects/{project}/locations/{location}/collections/{collection}/"
        f"engines/{engine_id}/servingConfigs/{serving_config}"
    )

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

    target = max(1, int(limit or 0))
    page_size = 10  # keep the same paging style as Skill 1
    max_results = max(target, page_size)

    links: List[str] = []
    seen = set()

    for offset in range(0, max_results, page_size):
        payload = {
            "servingConfig": serving_cfg_path,   # keep this field (same as Skill 1)
            "query": term,
            "pageSize": int(page_size),
            "offset": int(offset),               # 0-based
            "queryExpansionSpec": {"condition": "AUTO"},
            "spellCorrectionSpec": {"mode": "AUTO"},
            "languageCode": "en-US",
            "userInfo": {"timeZone": "America/New_York"},
            "userPseudoId": quota_user or "local-test-1",
        }

        try:
            r = requests.post(base, params={"key": key}, json=payload, timeout=60)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            log(f"[VertexSearch] error: {e}")
            break

        items = list(_iter_vertex_items(resp))

        for it in items:
            link = (it.get("link") or "").split("#")[0]
            if "pmc.ncbi.nlm.nih.gov" not in link:
                continue
            if link in seen:
                continue
            seen.add(link)
            links.append(link)
            if len(links) >= target:
                break

        # stop if enough or no more pages
        if len(links) >= target or len(items) < page_size:
            break

    log_debug(f"[VERTEX SEARCH] Term: {term}, Retrieved {len(links)} PMC links.")
    return links[:target]

def get_citation_count(pmid: str) -> int:
    """Use Entrez to fetch citation/backlink count for a PubMed article."""
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pubmed", id=pmid, linkname="pubmed_pubmed_citedin")
        result = Entrez.read(handle)
        handle.close()
        if result and result[0].get("LinkSetDb"):
            c = len(result[0]["LinkSetDb"][0]["Link"])
            log(f"[CITATION] PMID {pmid} has {c} citations.")
            return c
    except Exception as e:
        log(f"[CITATION ERROR] Failed to fetch citations for PMID {pmid}: {e}")
        return 0
    return 0

# ============================================================================
# 6) FULL-TEXT FETCH (HTML parsing, figures, metadata)
# ============================================================================

def fetch_full_text(pmc_url: str) -> Dict[str, Any] | None:
    """
    Download a PMC article page and extract:
      - pmid, date, journal, title
      - cleaned text content (references removed when possible)
      - figures: list of {'src','alt'}
    """
    log_debug(f"Fetching full text from: {pmc_url}")
    headers = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/98.0.4758.102 Safari/537.36')}
    try:
        response = requests.get(pmc_url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        log_debug(f"Error fetching PMC page {pmc_url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    def get_meta_content(name):
        tag = soup.find("meta", {"name": name})
        return tag["content"].strip() if tag and tag.get("content") else None

    pmid    = get_meta_content("citation_pmid")
    date    = get_meta_content("citation_publication_date")
    journal = get_meta_content("citation_journal_title")
    title   = get_meta_content("citation_title")

    # Normalize "YYYY Mon DD" → "YYYY-MM-DD"
    if date and re.match(r'^\d{4} [A-Za-z]{3} \d{1,2}$', date):
        try:
            date_obj = datetime.strptime(date, "%Y %b %d")
            date     = date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print("Date format conversion failed:", e)

    # Remove styles
    for style in soup.find_all('style'):
        style.decompose()

    # Remove references and collect main text
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
    else:
        full_text = soup.get_text(separator=" ", strip=True)
    full_text = re.sub(r'\s+', ' ', full_text)
    log_debug(f"Fetched full text from {pmc_url} (length: {len(full_text)}).")

    # Extract figures
    figures: List[Dict[str, str]] = []
    img_tags = soup.find_all("img", class_="graphic")
    for img_tag in img_tags:
        alt_text = img_tag.get("alt", "No description available")
        if not alt_text or not re.search(r'fig|figure', alt_text, re.I):
            continue
        href = None
        parent_a = img_tag.find_parent("a", href=True)
        if parent_a:
            href = parent_a.get("href")
        else:
            href = img_tag.get("src", "")

        if not href:
            continue

        if not href.startswith("http"):
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = "https://pmc.ncbi.nlm.nih.gov" + href

        figures.append({"src": href, "alt": alt_text})
        log_debug(f"[FIGURE EXTRACTED] href={href}, alt={alt_text}")

    if not figures:
        log_debug(f"[FIGURE EXTRACTED] No figures found in {pmc_url}.")

    return {
        "pmid": pmid,
        "pmc_url": pmc_url,
        "date": date,
        "journal": journal,
        "title": title,
        "text": full_text,
        "figures": figures,
    }

# ============================================================================
# 7) GPT HELPERS (sentence selection, conclusions, experiments, summaries/stats)
# ============================================================================

def extract_figure_table_sentences(text: str) -> List[str]:
    """Extract sentences that contain 'Figure', 'Fig.', or 'Table'."""
    sentence_delim_re = re.compile(r'(?<=[\.\?!。！？])\s+')
    sentences = sentence_delim_re.split(text)
    kw_re = re.compile(r'\b(?:Figure|Fig\.?|Table)\b', re.I)
    fig_sentences = [s.strip() for s in sentences if kw_re.search(s)]

    seen: Set[str] = set()
    uniq: List[str] = []
    for s in fig_sentences:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def gpt_extract_conclusions(text: str, gene: str, query: str) -> List[str]:
    """LLM: identify conclusion sentences tied to gene+query."""
    candidate_sentences = extract_figure_table_sentences(text)
    if not candidate_sentences:
        log_debug("[GPT CONCLUSIONS] No Figure/Table sentences found; fallback to full text.")
        candidate_block = text
    else:
        candidate_block = "\n".join(candidate_sentences)

    system = (
        "You are a biomedical literature assistant.\n"
        "From the SENTENCES provided, identify EVERY conclusion that reports an "
        "experimental finding related to BOTH the specified gene and the user's query.\n"
        "Rewrite each as one clear sentence. Return ONLY a JSON array of strings."
    )
    user_content = (
        f"GENE  : {gene}\n"
        f"QUERY : {query}\n"
        "SENTENCES:\n"
        f"{candidate_block}"
    )

    try:
        resp = client2.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user_content}],
        )
        raw_response = resp.choices[0].message.content.strip()
        log_debug(f"[GPT CONCLUSIONS RAW OUTPUT] {raw_response}")

        cleaned = clean_json_response(raw_response)
        conclusions = json.loads(cleaned) if cleaned else []
        if not isinstance(conclusions, list):
            conclusions = []
        return conclusions
    except Exception as e:
        log(f"[gpt_extract_conclusions] error: {e}")
        return []

def gpt_extract_experiments(
    article_text: str,
    conclusions: List[str],
    gene: str,
    query: str,
    pmc_url: str,
) -> List[Dict[str, Any]]:
    """LLM: map conclusions → detailed supporting experiments."""
    if not conclusions:
        return []

    system = (
        "You are a biomedical text-mining assistant.\n"
        "For EACH conclusion supplied by the user, locate ALL experiments in the article "
        "that directly support that conclusion.\n"
        "For every experiment you find, output one JSON object containing exactly these keys:\n"
        "  pmc_url, conclusion, model, intervention, readout, result_stats, extra_data, "
        "  evidence_type, experiment_excerpt\n\n"
        "Categorize the evidence_type as EXACTLY one of:\n"
        "• Molecular Mechanism Experiments – in-vitro or molecular assays (qPCR, Western blot, RNA pulldown, ChIP, ELISA, etc.).\n"
        "• Cellular Function Experiments – cultured-cell functional assays (proliferation, apoptosis, migration, reporter, etc.).\n"
        "• Animal Function Experiments – in-vivo whole-animal studies (mouse/rat/zebrafish injections, transgenics, xenografts, physiology, pathology, etc.).\n"
        "• Clinical Information Analysis Experiments – human imaging or pathology procedures (MRI, CT, PET, biopsy histology, smears, etc.).\n"
        "• Bioinformatic Analysis – purely computational analyses (TCGA mining, sequence alignment, public dataset statistics, cohort analysis etc.).\n\n"
        "Make sure that ALL input conclusions are covered – if an input conclusion has no "
        "supporting experiment, still include one JSON object with empty fields except "
        "`conclusion` and `pmc_url`, and set evidence_type to \"No Supporting Experiment\".\n"
        "Return a pure JSON array. DO NOT wrap in code fences."
    )

    user = (
        f"GENE      : {gene}\n"
        f"QUERY     : {query}\n"
        f"PMC_URL   : {pmc_url}\n"
        f"CONCLUSIONS (to be covered):\n{json.dumps(conclusions, ensure_ascii=False)}\n\n"
        "FULL ARTICLE TEXT:\n"
        f"{article_text}"
    )

    try:
        resp = client2.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        cleaned = clean_json_response(resp.choices[0].message.content.strip())
        experiments = json.loads(cleaned) if cleaned else []
        if not isinstance(experiments, list):
            experiments = []
        return experiments
    except Exception as e:
        log(f"[gpt_extract_experiments] error: {e}")
        return []

def generate_experimental_report(meta: Dict[str, Any],
                                 conclusions: List[str],
                                 experiments: List[Dict[str, Any]]) -> str:
    """Produce a plain-text block summarizing conclusions and experiments."""
    lines = []
    lines.append(f"[PMC_URL]: {meta['pmc_url']}")
    lines.append("[CONCLUSIONS]:")
    for i, c in enumerate(conclusions, 1):
        lines.append(f"  - {i}. {c}")

    lines.append("[EXPERIMENTAL_DETAILS]:")
    if not experiments:
        lines.append("  - No experimental details extracted.")
    else:
        for i, e in enumerate(experiments, 1):
            lines.append(f"  - {i}. Conclusion: {e.get('conclusion','')}")
            lines.append(f"      Model: {e.get('model','')}")
            lines.append(f"      Intervention: {e.get('intervention','')}")
            lines.append(f"      Readout: {e.get('readout','')}")
            lines.append(f"      Result/Stats: {e.get('result_stats','')}")
            lines.append(f"      Extra Data: {e.get('extra_data','')}")
            lines.append(f"      Evidence Type: {e.get('evidence_type', 'N/A')}")
            lines.append(f"      Excerpt: {e.get('experiment_excerpt','')}")
    lines.append("[SUMMARY]:")
    lines.append("  - The report compiles key experimental findings.")
    return "\n".join(lines)

def process_article(article: Dict[str, Any], gene: str, query: str) -> Dict[str, Any]:
    """Run extraction on one article object returned by fetch_full_text."""
    relevant = extract_relevant_sentences(article["text"], gene)
    conclusions = gpt_extract_conclusions(relevant, gene, query)
    experiments = gpt_extract_experiments(article["text"], conclusions, gene, query, article["pmc_url"])

    report = generate_experimental_report(
        dict(pmc_url=article["pmc_url"], pmid=article["pmid"],
             date=article["date"], journal=article["journal"], title=article["title"]),
        conclusions, experiments
    )

    return dict(
        meta=dict(pmid=article["pmid"], pmc_url=article["pmc_url"],
                  date=article["date"], journal=article["journal"],
                  title=article["title"]),
        report=report, conclusions=conclusions,
        experiments=experiments, figures=article.get("figures", [])
    )

def gpt_generate_summary(results: List[Dict[str, Any]], gene: str, query: str) -> Dict[str, str]:
    """
    For each article (with conclusions+experiments), ask GPT to generate a concise
    paragraph-level summary focused on experimental evidence. Returns {pmc_url: summary_text}.
    """
    summaries: Dict[str, str] = {}
    system_msg = (  
        "You are a biomedical research assistant. Based on the extracted experimental evidence from articles, "  
        f"generate a concise summary paragraph addressing the user's query regarding the gene ({gene}). "  
        "Summarize key findings across the provided conclusions and experiments. "  
        "Focus on experimental results, avoid speculation, and ensure the summary is clear and relevant to the query. "  
        "Return a single paragraph as plain text."  
    )  

    for art in results:
        if not art.get("conclusions", []) or not art.get("experiments", []):
            continue

        meta = art["meta"]
        conclusions = art["conclusions"]
        experiments = art["experiments"]
        pmc_url = meta["pmc_url"]

        user_msg = (
            f"User Query: {query}\n"
            f"Gene: {gene}\n"
            f"Article: {meta['title']} ({pmc_url})\n"
            f"Conclusions: {json.dumps(conclusions)}\n"
            f"Experiments: {json.dumps(experiments)}\n"
            "Generate a summary paragraph based on the above information."
        )

        try:
            resp = client_gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}]
            )
            summaries[pmc_url] = resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"[gpt_generate_summary] error for {pmc_url}: {e}")
            summaries[pmc_url] = "Failed to generate summary due to an error."

    return summaries

def gpt_generate_statistics(
    results: List[Dict[str, Any]],
    gene: str,
    query: str,
) -> Dict[str, Any]:
    """
    Aggregate experiments and ask GPT to rate evidence strength per category.
    Deterministic thresholds are enforced (Strong/Moderate/Weak/No Evidence).
    """
    import json as _json_local, re as _re_local
    from collections import defaultdict as _dd

    buckets = {
        "Molecular Mechanism Experiments": [],
        "Cellular Function Experiments": [],
        "Animal Function Experiments": [],
        "Clinical Information Analysis Experiments": [],
        "Bioinformatic Analysis": [],
    }
    article_experiments = _dd(list)

    for art in results:
        for exp in art.get("experiments", []):
            etype = (exp.get("evidence_type") or "").lower()

            if "molecular" in etype:
                cat = "Molecular Mechanism Experiments"
            elif "cellular" in etype:
                cat = "Cellular Function Experiments"
            elif "animal" in etype:
                cat = "Animal Function Experiments"
            elif "clinical" in etype:
                cat = "Clinical Information Analysis Experiments"
            elif "bioinformatic" in etype:
                cat = "Bioinformatic Analysis"
            else:
                if _re_local.search(r"cell", etype):
                    cat = "Cellular Function Experiments"
                elif _re_local.search(r"mouse|rat|zebra", _json_local.dumps(exp).lower()):
                    cat = "Animal Function Experiments"
                else:
                    continue

            buckets[cat].append(exp)
            article_experiments[cat].append((art["meta"], exp))

    if all(len(v) == 0 for v in buckets.values()):
        return {
            "molecular_mechanism_evidence": "No Evidence",
            "cellular_function_evidence": "No Evidence",
            "animal_function_evidence": "No Evidence",
            "clinical_information_evidence": "No Evidence",
            "bioinformatic_analysis_evidence": "No Evidence",
            "query_response": "No experimental data relevant to the query were found.",
            "molecular_mechanism_details": "",
            "cellular_function_details": "",
            "animal_function_details": "",
            "clinical_information_details": "",
            "bioinformatic_analysis_details": "",
        }

    def brief(exp: Dict[str, Any]) -> Dict[str, str]:
        return {
            "model": exp.get("model", ""),
            "intervention": exp.get("intervention", ""),
            "readout": exp.get("readout", ""),
            "result_stats": exp.get("result_stats", ""),
            "conclusion": exp.get("conclusion", ""),
        }

    bucket_payload, bucket_stats = {}, {}
    for cat, pairs in article_experiments.items():
        bucket_payload[cat] = [
            {"pmid": m.get("pmid", "N/A"), "experiment": brief(e)} for m, e in pairs
        ]
        unique_pmids = {m.get("pmid", "N/A") for m, _ in pairs}
        bucket_stats[cat] = {
            "num_unique_pmids": len(unique_pmids),
            "num_experiments": len(pairs),
        }

    system_msg = (
        "You are an expert systematic reviewer in biomedical science. Your task is to rate "
        "the overall strength of evidence for EACH experiment category with respect to the "
        "USER'S GENE and QUERY, using ONLY the experiments provided. Follow these exact rules:\n\n"
        f"GENE  : {gene}\n"
        f"QUERY : {query}\n\n"
        "Deterministic thresholds (must not be overridden):\n"
        "• Strong   → at least 2 unique PMIDs AND at least 3 total experiments.\n"
        "• Moderate → (exactly 1 unique PMID AND ≥1 experiment) OR (≥2 PMIDs BUT <3 experiments).\n"
        "• Weak     → a single experiment OR indirect/preliminary evidence.\n"
        "• No Evidence → no experiments in this category.\n"
        "If a case is borderline, choose the highest level justified by PMID and experiment counts. "
        "Do NOT apply personal judgment beyond these rules.\n\n"
        "Return a JSON object with EXACTLY these keys:\n"
        "  molecular_mechanism_evidence, cellular_function_evidence, animal_function_evidence,\n"
        "  clinical_information_evidence, bioinformatic_analysis_evidence, query_response,\n"
        "  molecular_mechanism_details, cellular_function_details, animal_function_details,\n"
        "  clinical_information_details, bioinformatic_analysis_details\n\n"
        "Formatting rules:\n"
        "• *_evidence fields MUST be one of: Strong, Moderate, Weak, No Evidence (no extra words).\n"
        "• *_details fields: ≤3 sentences, cite key PMIDs like \"PMID 123456 showed …\".\n"
        "• query_response: ≤1 short paragraph directly answering the QUERY using the strongest evidence.\n"
        "Do NOT output markdown, bullet points, or keys not listed above."
    )

    user_msg = (
        "EXPERIMENTS_GROUPED_BY_CATEGORY:\n"
        f"{_json_local.dumps(bucket_payload)}\n\n"
        "CATEGORY_COUNTS_FOR_RULES:\n"
        f"{_json_local.dumps(bucket_stats)}"
    )

    try:
        resp = client2.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        cleaned = clean_json_response(resp.choices[0].message.content.strip())
        stats = json.loads(cleaned)
    except Exception as e:
        log(f"[gpt_generate_statistics] error: {e}")
        stats = {}

    defaults = {
        "molecular_mechanism_evidence": "No Evidence",
        "cellular_function_evidence": "No Evidence",
        "animal_function_evidence": "No Evidence",
        "clinical_information_evidence": "No Evidence",
        "bioinformatic_analysis_evidence": "No Evidence",
        "query_response": "No relevant experimental data found.",
        "molecular_mechanism_details": "",
        "cellular_function_details": "",
        "animal_function_details": "",
        "clinical_information_details": "",
        "bioinformatic_analysis_details": "",
    }
    for k, v in defaults.items():
        stats.setdefault(k, v)

    return stats

# ============================================================================
# 8) HTML BUILDERS (placeholder pre-HTML + interactive final report)
# ============================================================================

def build_pre_html_page(message: str, title: str = "RBP Evidence Check") -> str:
    """Minimal placeholder HTML."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{_html.escape(title)}</title>
  <style>
    * {{margin:0;padding:0;box-sizing:border-box;}}
    body {{font-family:Arial, sans-serif;}}
    header {{background:linear-gradient(to right,#00274d,#5a9bd6);color:#fff;padding:20px;text-align:center;}}
    footer {{background:#00274d;color:#fff;text-align:center;padding:15px;}}
    .container {{padding:20px;margin-left:270px;margin-right:20px;}}
    .textbox {{
      width:90%;margin:auto;padding:20px;background:#fff;border:1px solid #ccc;
      border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);
      max-height:calc(100vh - 250px);overflow:auto;
    }}
  </style>
</head>
<body>
  <header><h1>{_html.escape(title)}</h1></header>
  <div class="container">
    <div class="textbox">
      <p style="font-size:1.2em;text-align:center;margin-top:40px;">{_html.escape(message)}</p>
    </div>
  </div>
  <footer><p>&copy; {datetime.now().year} LNC Research. All Rights Reserved.</p></footer>
</body>
</html>"""

def generate_html_report(
    results: list[dict],
    gene: str,
    query: str,
    output_path: str,
    *,
    statistics: Dict[str, Any] | None = None,
    summaries: Dict[str, str] | None = None,
    allow_llm: bool = True,
):
    """
    Build the interactive final HTML report.
    If statistics/summaries are provided (e.g., from snapshot), they are reused.
    If they are missing and allow_llm=True, they will be (re)computed via GPT calls.
    Returns a dict {'statistics': ..., 'summaries': ..., 'evidence_results': [...]}
    so downstream steps (Excel export) can reuse already-computed data.
    """
    import json as _json_local
    import re as _re_local

    esc = lambda t: _html.escape(str(t)) if t is not None else ""

    evidence_results = [r for r in results if r.get("experiments")]
    # Write a minimal HTML and return empty payload if nothing has experiments
    if not evidence_results:
        Path(output_path).write_text("<h2>No articles with experimental evidence were found.</h2>", "utf-8")
        return {"statistics": None, "summaries": {}, "evidence_results": []}

    # Compute statistics and summaries if not provided
    if (statistics is None or summaries is None) and allow_llm:
        if statistics is None:
            statistics = gpt_generate_statistics(evidence_results, gene, query)
        if summaries is None:
            summaries = gpt_generate_summary(evidence_results, gene, query)

    # If LLM calls are disabled and we have no statistics/summaries, fall back to blanks
    statistics = statistics or {
        "molecular_mechanism_evidence": "No Evidence",
        "cellular_function_evidence": "No Evidence",
        "animal_function_evidence": "No Evidence",
        "clinical_information_evidence": "No Evidence",
        "bioinformatic_analysis_evidence": "No Evidence",
        "query_response": "No relevant experimental data found.",
        "molecular_mechanism_details": "",
        "cellular_function_details": "",
        "animal_function_details": "",
        "clinical_information_details": "",
        "bioinformatic_analysis_details": "",
    }
    summaries = summaries or {}

    category_map = [
        ("Molecular Mechanism Experiments", "molecular_mechanism_evidence", "molecular_mechanism_details"),
        ("Cellular Function Experiments",   "cellular_function_evidence",   "cellular_function_details"),
        ("Animal Function Experiments",     "animal_function_evidence",     "animal_function_details"),
        ("Clinical Information Analysis Experiments", "clinical_information_evidence", "clinical_information_details"),
        ("Bioinformatic Analysis",          "bioinformatic_analysis_evidence", "bioinformatic_analysis_details"),
    ]
    strength_to_val = {"No Evidence": 0, "Weak": 1, "Moderate": 2, "Strong": 3}
    color_map = {"Weak": "rgba(13,110,253,0.7)", "Moderate": "rgba(255,193,7,0.9)", "Strong": "rgba(220,53,69,0.9)"}

    table_rows_html, chart_labels, chart_values, chart_colors = [], [], [], []
    for label, e_key, d_key in category_map:
        level = statistics.get(e_key, "No Evidence")
        if level == "No Evidence":
            continue

        detail_raw = statistics.get(d_key, "")
        segs = [x.strip() for x in _re_local.split(r"[;；。\n]", detail_raw) if x.strip()]
        pmid_segs = [s for s in segs if _re_local.search(r"PMID\s*:?\s*\d+", s, re.I)]
        use_segs  = pmid_segs if pmid_segs else segs
        detail_html = "<ul class='mb-0'>" + "".join(f"<li>{esc(s)}</li>" for s in use_segs) + "</ul>"

        badge_class = {"Weak": "bg-primary", "Moderate": "bg-warning text-dark", "Strong": "bg-danger"}[level]

        table_rows_html.append(
            f"<tr><td>{esc(label)}</td>"
            f"<td><span class='badge {badge_class}'>{esc(level)}</span></td>"
            f"<td>{detail_html}</td></tr>"
        )

        chart_labels.append(label)
        chart_values.append(strength_to_val[level])
        chart_colors.append(color_map[level])

    evidence_chart_json = _json_local.dumps({"labels": chart_labels, "values": chart_values, "colors": chart_colors})

    citations = [a["meta"].get("citation_count", 0) for a in evidence_results]
    journals  = [a["meta"].get("journal", "Unknown") for a in evidence_results]

    def _bucket(c):
        return "0" if c == 0 else "1–10" if c <= 10 else "11–50" if c <= 50 else "51–100" if c <= 100 else "100+"

    cit_counter = Counter(_bucket(c) for c in citations)
    cit_chart_json = _json_local.dumps({"labels": list(cit_counter.keys()), "values": list(cit_counter.values())})

    jour_counter = Counter(journals).most_common(10)
    journal_chart_json = _json_local.dumps({"labels": [j for j, _ in jour_counter], "values": [n for _, n in jour_counter]})

    evidence_results.sort(key=lambda x: x["meta"].get("citation_count", 0), reverse=True)

    html_parts: List[str] = []
    fig_id_pattern = re.compile(r"(?:Fig(?:ure)?\.?\s*)(\d+[A-Za-z]?)", re.I)

    # <head>
    html_parts.append(
        """<!DOCTYPE html><html lang="en"><head>
<meta charset='utf-8'>
<title>RBP Evidence Check</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{display:flex;flex-direction:column;min-height:100vh}
  header{background:linear-gradient(to right,#00274d,#5a9bd6);color:#fff;padding:20px;text-align:center;flex-shrink:0;}
  main{flex:1;padding:24px;background:#f8f9fa}
  footer{background:#00274d;color:#fff;text-align:center;padding:15px;flex-shrink:0}
</style>
"""
    )

    # charts script
    html_parts.append(
        f"""
<script>
document.addEventListener('DOMContentLoaded',function(){{
  const ev={evidence_chart_json};
  if(ev.labels.length){{
    new Chart(document.getElementById('evidenceChart'),{{
      type:'bar',
      data:{{labels:ev.labels,datasets:[{{data:ev.values,backgroundColor:ev.colors}}]}},
      options:{{plugins:{{legend:{{display:false}}}},
                scales:{{y:{{beginAtZero:true,ticks:{{callback:v=>['None','Weak','Moderate','Strong'][v]??v}}}}}}}}
    }});
  }}
  const ct={cit_chart_json};
  new Chart(document.getElementById('citChart'),{{
    type:'bar',
    data:{{labels:ct.labels,datasets:[{{data:ct.values,backgroundColor:'rgba(54,162,235,0.8)'}}]}},
    options:{{plugins:{{legend:{{display:false}}}},
              scales:{{x:{{title:{{display:true,text:'Citation Num'}}}},
                       y:{{title:{{display:true,text:'Num of papers'}},beginAtZero:true}}}}}}
  }});
  const jr={journal_chart_json};
  new Chart(document.getElementById('journalChart'),{{
    type:'pie',
    data:{{labels:jr.labels,
           datasets:[{{data:jr.values,
                       backgroundColor:jr.labels.map((_,i)=>`hsl(${{i*37}},70%,60%)`)}}]}}
  }});
}});
</script>"""
    )

    html_parts.append("</head><body>")
    html_parts.append("<header><h1>RBP Evidence Check</h1></header>")

    # main header
    html_parts.append("<main class='container-lg'>")
    html_parts.append(
        f"<h2>Experimental evidence for <code>{esc(gene)}</code></h2>"
        f"<p class='text-muted'><strong>User query:</strong> {esc(query)}</p>"
        f"<p class='fst-italic text-muted'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
    )

    # charts + table
    html_parts.append(
        """
<div class="row gy-3">
  <div class="col-lg-6">
    <div class="card h-100"><div class="card-body">
      <h4 class="card-title mb-3">Evidence charts</h4>
      <h6 class="fw-semibold">Evidence strength</h6>
      <canvas id="evidenceChart" height="200" class="mb-4"></canvas>
      <h6 class="fw-semibold">Citation histogram</h6>
      <canvas id="citChart" height="180" class="mb-4"></canvas>
      <h6 class="fw-semibold">Journal distribution</h6>
      <canvas id="journalChart" height="220"></canvas>
    </div></div>
  </div>
  <div class="col-lg-6">
    <div class="card h-100"><div class="card-body">
      <h4 class="card-title mb-3">Evidence table</h4>
      <div class='table-responsive'><table class='table table-sm align-middle'>
        <thead class='table-light'><tr><th>Category</th><th>Evidence</th><th>Details</th></tr></thead><tbody>
"""
    )
    html_parts.extend(table_rows_html)
    html_parts.append(
        "</tbody></table></div>"
        "<h6 class='mt-3'>Answer to user query</h6>"
        f"<p>{_html.escape(statistics.get('query_response','No relevant data'))}</p>"
        "</div></div></div></div>"
    )

    # Article TOC
    html_parts.append("<h4 class='mt-4 mb-3'>Articles</h4><ol>")
    for i, art in enumerate(evidence_results, 1):
        html_parts.append(f"<li><a href='#art{i}'>{esc(art['meta'].get('title','Untitled'))}</a></li>")
    html_parts.append("</ol>")

    # Accordion per-article details
    html_parts.append("<div class='accordion' id='articleAccordion'>")
    for idx, art in enumerate(evidence_results, 1):
        m     = art["meta"]
        exps  = art["experiments"]
        concl = art["conclusions"]
        figs  = art.get("figures", [])

        fig_lookup = {}
        for fg in figs:
            alt = fg.get("alt", "")
            for tkn in fig_id_pattern.findall(alt):
                num = re.match(r"(\d+)", tkn).group(1)
                if num not in fig_lookup:
                    fig_lookup[num] = {"src": fg["src"], "alt": alt}

        html_parts.append(
            f"""
<div class='accordion-item'>
  <h2 class='accordion-header' id='heading{idx}'>
    <button class='accordion-button collapsed' type='button' data-bs-toggle='collapse'
            data-bs-target='#collapse{idx}' aria-expanded='false' aria-controls='collapse{idx}' id='art{idx}'>
      {idx}. {esc(m.get('title','Untitled'))}
      <span class='badge bg-success ms-2'>Has Evidence</span>
    </button>
  </h2>
  <div id='collapse{idx}' class='accordion-collapse collapse' aria-labelledby='heading{idx}' data-bs-parent='#articleAccordion'>
    <div class='accordion-body'>
"""
        )

        meta_line = (
            f"{esc(m.get('journal',''))} ({esc(m.get('date',''))}) · "
            f"PMID {esc(m.get('pmid',''))} · Citations: {esc(m.get('citation_count','N/A'))} · "
            f"<a href='{esc(m.get('pmc_url',''))}' target='_blank'>PMC link</a>"
        )
        html_parts.append(f"<p class='small text-muted'>{meta_line}</p>")

        # Summary (from LLM / snapshot)
        html_parts.append(
            "<details open class='mb-2'><summary class='fw-semibold'>Summary</summary>"
            f"<div class='ps-3'>{esc(summaries.get(m.get('pmc_url',''),'No summary'))}</div></details>"
        )

        # Conclusions
        html_parts.append("<details class='mb-2'><summary class='fw-semibold'>Conclusions</summary><ol class='ps-3'>")
        for c in concl:
            html_parts.append(f"<li>{esc(c)}</li>")
        html_parts.append("</ol></details>")

        # Experiments table
        html_parts.append(
            "<details class='mb-2'><summary class='fw-semibold'>Experiments</summary>"
            "<div class='table-responsive'><table class='table table-bordered table-sm align-middle'>"
            "<thead class='table-light'><tr><th>#</th><th>Model</th><th>Intervention</th>"
            "<th>Readout</th><th>Result / Stats</th><th>Extra</th><th>Evidence Type</th><th>Conclusion</th></tr>"
            "</thead><tbody>"
        )
        for j, e in enumerate(exps, 1):
            extra = e.get("extra_data", "")
            formatted_extra = ""
            if isinstance(extra, str) and extra.strip():
                parts = re.split(r"[;,]", extra)
                buf = []
                for part in parts:
                    part = part.strip()
                    m2   = fig_id_pattern.search(part)
                    if m2:
                        num = re.match(r"(\d+)", m2.group(1)).group(1)
                        if num in fig_lookup:
                            src = fig_lookup[num]["src"]
                            buf.append(f"<a href='{esc(src)}' target='_blank' rel='noopener noreferrer'>{_html.escape(part)}</a>")
                            continue
                    buf.append(_html.escape(part))
                formatted_extra = ", ".join(buf)

            html_parts.append(
                "<tr>"
                f"<td>{j}</td><td>{_html.escape(e.get('model',''))}</td><td>{_html.escape(e.get('intervention',''))}</td>"
                f"<td>{_html.escape(e.get('readout',''))}</td><td>{_html.escape(e.get('result_stats',''))}</td>"
                f"<td>{formatted_extra}</td><td>{_html.escape(e.get('evidence_type',''))}</td>"
                f"<td>{_html.escape(e.get('conclusion',''))}</td></tr>"
            )
        html_parts.append("</tbody></table></div></details>")

        # Raw text block
        html_parts.append(
            f"<p><a class='btn btn-sm btn-outline-secondary' data-bs-toggle='collapse' "
            f"href='#raw{idx}' aria-expanded='false' aria-controls='raw{idx}'>Show raw text</a></p>"
            f"<div class='collapse' id='raw{idx}'><pre class='p-3 bg-light border'>{_html.escape(art.get('report',''))}</pre></div>"
            "</div></div></div>"
        )
    html_parts.append("</div>")  # accordion end

    html_parts.append(
        "</main><footer>&copy; "
        f"2025 AgentLnc. All Rights Reserved.</footer></body></html>"
    )

    Path(output_path).write_text("".join(html_parts), "utf-8")
    # Return payload for Excel export
    return {"statistics": statistics, "summaries": summaries, "evidence_results": evidence_results}

# ============================================================================
# 9) EXCEL SUPPLEMENT EXPORT — modeled after Skill 1, adapted to Skill 2 results
# ============================================================================

def create_excel_supplement_from_results(
    results: List[Dict[str, Any]],
    gene: str,
    query: str,
    output_html_path: str,
    statistics: Dict[str, Any] | None = None,
    summaries: Dict[str, str] | None = None,
    *,
    allow_llm: bool = True,
) -> str:
    """
    NEW minimal 3-sheet Excel:
      1) 'readme'   — provenance / parameters
      2) 'evidence' — Evidence table + Evidence charts (in the SAME sheet)
      3) 'articles' — One big table: article meta + conclusion + each evidence on its own row
                      (if a single conclusion has multiple pieces of evidence, each occupies a row)
    If statistics is None and allow_llm=True, statistics will be computed; otherwise left blank/default.
    """
    import re as _re
    from collections import Counter as _Counter

    # Keep only articles that actually have extracted experiments
    evidence_results = [r for r in results if r.get("experiments")]
    summaries = summaries or {}

    # If statistics weren't passed in (should be available from HTML step), compute them
    if statistics is None and evidence_results and allow_llm:
        statistics = gpt_generate_statistics(evidence_results, gene, query)
    statistics = statistics or {
        "molecular_mechanism_evidence": "No Evidence",
        "cellular_function_evidence": "No Evidence",
        "animal_function_evidence": "No Evidence",
        "clinical_information_evidence": "No Evidence",
        "bioinformatic_analysis_evidence": "No Evidence",
        "query_response": "No relevant experimental data found.",
        "molecular_mechanism_details": "",
        "cellular_function_details": "",
        "animal_function_details": "",
        "clinical_information_details": "",
        "bioinformatic_analysis_details": "",
    }

    # -----------------------------
    # 1) README dataframe
    # -----------------------------
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_readme = pd.DataFrame([
        {"Key": "Gene", "Value": gene},
        {"Key": "User Query", "Value": query},
        {"Key": "Generated At", "Value": now_str},
        {"Key": "Output HTML", "Value": output_html_path},
        {"Key": "Articles (with evidence)", "Value": str(len(evidence_results))},
    ])

    # -----------------------------
    # 2) Evidence table + charts data
    # -----------------------------
    cat_map = [
        ("Molecular Mechanism Experiments", "molecular_mechanism_evidence", "molecular_mechanism_details"),
        ("Cellular Function Experiments",   "cellular_function_evidence",   "cellular_function_details"),
        ("Animal Function Experiments",     "animal_function_evidence",     "animal_function_details"),
        ("Clinical Information Analysis Experiments", "clinical_information_evidence", "clinical_information_details"),
        ("Bioinformatic Analysis",          "bioinformatic_analysis_evidence", "bioinformatic_analysis_details"),
    ]
    strength_to_val = {"No Evidence": 0, "Weak": 1, "Moderate": 2, "Strong": 3}

    # Evidence table shown to the user
    ev_rows = []
    ev_bar_labels, ev_bar_values = [], []
    for label, key_e, key_d in cat_map:
        lvl = statistics.get(key_e, "No Evidence")
        det = statistics.get(key_d, "")
        ev_rows.append({"Category": label, "Evidence Level": lvl, "Details": det})
        if lvl != "No Evidence":
            ev_bar_labels.append(label)
            ev_bar_values.append(strength_to_val.get(lvl, 0))
    df_evidence_table = pd.DataFrame(ev_rows)
    df_query = pd.DataFrame([{"Answer to query": statistics.get("query_response", "")}])

    # Chart backing data (written into the same sheet; charts will reference these cells)
    df_ev_bar = pd.DataFrame({"Label": ev_bar_labels, "Value": ev_bar_values})

    # Citation histogram
    citations = [a["meta"].get("citation_count", 0) for a in evidence_results]
    def _bucket(c): 
        return "0" if c == 0 else "1–10" if c <= 10 else "11–50" if c <= 50 else "51–100" if c <= 100 else "100+"
    cit_counter = _Counter(_bucket(c) for c in citations)
    df_cit_hist = pd.DataFrame({"Bin": list(cit_counter.keys()), "Count": list(cit_counter.values())})

    # Journal distribution (top 10)
    journals = [a["meta"].get("journal", "Unknown") for a in evidence_results]
    jctr = _Counter(journals).most_common(10)
    df_journal_dist = pd.DataFrame({"Journal": [j for j, _ in jctr], "Count": [n for _, n in jctr]})

    # -----------------------------
    # 3) Combined 'articles' table (one row per evidence)
    # -----------------------------
    def _norm(s: str) -> str:
        return _re.sub(r"\s+", " ", (s or "").strip().lower())

    combined_rows: List[Dict[str, Any]] = []
    for art in evidence_results:
        m = art["meta"]
        pmid = m.get("pmid", "")
        pmc  = m.get("pmc_url", "")
        summ = summaries.get(pmc, "")
        title = m.get("title", "")
        jn    = m.get("journal", "")
        date  = m.get("date", "")
        cits  = m.get("citation_count", "")

        # Map conclusions to indices to help users track which conclusion each evidence supports
        concl_idx = { _norm(c): (i+1) for i, c in enumerate(art.get("conclusions", [])) }

        for e in art.get("experiments", []):
            c_text = e.get("conclusion", "")
            c_no   = concl_idx.get(_norm(c_text), "")
            combined_rows.append({
                "PMID": pmid,
                "PMCID/URL": pmc,
                "Date": date,
                "Journal": jn,
                "Title": title,
                "Citation Count": cits,
                "Conclusion #": c_no,
                "Conclusion": c_text,
                "Evidence Type": e.get("evidence_type", ""),
                "Model": e.get("model", ""),
                "Intervention": e.get("intervention", ""),
                "Readout": e.get("readout", ""),
                "Result/Stats": e.get("result_stats", ""),
                "Extra": e.get("extra_data", ""),
                "Excerpt": e.get("experiment_excerpt", ""),
                "Article Summary": summ,
            })
    df_articles = pd.DataFrame(combined_rows)

    # -----------------------------
    # 4) Write the 3-sheet workbook
    # -----------------------------
    excel_path = os.path.splitext(output_html_path)[0] + ".xlsx"

    # Prefer xlsxwriter (better charts). Fallback to openpyxl if needed.
    try:
        writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
        _engine = "xlsxwriter"
    except Exception:
        writer = pd.ExcelWriter(excel_path, engine="openpyxl")
        _engine = "openpyxl"

    try:
        # -- Sheet 1: readme
        df_readme.to_excel(writer, sheet_name="readme", index=False)

        # -- Sheet 2: evidence (table + charts)
        sh = "evidence"

        # layout (0-based rows); keep gaps so charts have room
        r_tbl = 1  # evidence table start
        df_evidence_table.to_excel(writer, sheet_name=sh, index=False, startrow=r_tbl)

        r_query = r_tbl + df_evidence_table.shape[0] + 3
        df_query.to_excel(writer, sheet_name=sh, index=False, startrow=r_query)

        r_evdata = r_query + df_query.shape[0] + 3
        df_ev_bar.to_excel(writer, sheet_name=sh, index=False, startrow=r_evdata)

        r_citdata = r_evdata + max(2, df_ev_bar.shape[0] + 3)
        df_cit_hist.to_excel(writer, sheet_name=sh, index=False, startrow=r_citdata)

        r_jrdata = r_citdata + max(2, df_cit_hist.shape[0] + 3)
        df_journal_dist.to_excel(writer, sheet_name=sh, index=False, startrow=r_jrdata)

        # -- Sheet 3: articles (wide table)
        df_articles.to_excel(writer, sheet_name="articles", index=False)

        # Freeze panes for usability
        if _engine == "xlsxwriter":
            ws_ev = writer.sheets[sh]
            ws_ev.freeze_panes(r_tbl + 1, 0)       # after header
            ws_ar = writer.sheets["articles"]
            ws_ar.freeze_panes(1, 0)
        else:  # openpyxl
            ws_ev = writer.sheets[sh]
            ws_ar = writer.sheets["articles"]
            try:
                ws_ev.freeze_panes = ws_ev["A{}".format(r_tbl + 2)]
                ws_ar.freeze_panes = ws_ar["A2"]
            except Exception:
                pass

        # -----------------------------
        # Insert charts in 'evidence'
        # -----------------------------
        if _engine == "xlsxwriter":
            book = writer.book

            # Evidence strength (column)
            if df_ev_bar.shape[0] > 0:
                ch1 = book.add_chart({"type": "column"})
                # categories/values: rows are startrow+1 .. +N  (skip header)
                ch1.add_series({
                    "name": "Evidence strength (0–3)",
                    "categories": [sh, r_evdata + 1, 0, r_evdata + df_ev_bar.shape[0], 0],
                    "values":     [sh, r_evdata + 1, 1, r_evdata + df_ev_bar.shape[0], 1],
                })
                ch1.set_title({"name": "Evidence strength"})
                ch1.set_y_axis({"name": "Level (0–3)", "major_unit": 1, "min": 0, "max": 3})
                ws_ev.insert_chart("H3", ch1, {"x_scale": 1.2, "y_scale": 1.2})

            # Citation histogram (column)
            if df_cit_hist.shape[0] > 0:
                ch2 = book.add_chart({"type": "column"})
                ch2.add_series({
                    "name": "Num papers",
                    "categories": [sh, r_citdata + 1, 0, r_citdata + df_cit_hist.shape[0], 0],
                    "values":     [sh, r_citdata + 1, 1, r_citdata + df_cit_hist.shape[0], 1],
                })
                ch2.set_title({"name": "Citation histogram"})
                ch2.set_x_axis({"name": "Citation bin"})
                ch2.set_y_axis({"name": "Num papers", "min": 0})
                ws_ev.insert_chart("H22", ch2, {"x_scale": 1.2, "y_scale": 1.2})

            # Journal distribution (pie)
            if df_journal_dist.shape[0] > 0:
                ch3 = book.add_chart({"type": "pie"})
                ch3.add_series({
                    "name": "Journal distribution (top 10)",
                    "categories": [sh, r_jrdata + 1, 0, r_jrdata + df_journal_dist.shape[0], 0],
                    "values":     [sh, r_jrdata + 1, 1, r_jrdata + df_journal_dist.shape[0], 1],
                })
                ch3.set_title({"name": "Journal distribution (top 10)"})
                ws_ev.insert_chart("H41", ch3, {"x_scale": 1.2, "y_scale": 1.2})

        else:
            # openpyxl chart insertion (fallback)
            try:
                from openpyxl.chart import BarChart, Reference, PieChart

                # Evidence strength
                if df_ev_bar.shape[0] > 0:
                    ch1 = BarChart()
                    ch1.title = "Evidence strength"
                    ch1.y_axis.title = "Level (0–3)"
                    # header is at (r_evdata + 1) zero-based -> +1 for 1-based Excel
                    cat = Reference(ws_ev, min_col=1, min_row=r_evdata + 2, max_row=r_evdata + 1 + df_ev_bar.shape[0])
                    val = Reference(ws_ev, min_col=2, min_row=r_evdata + 2, max_row=r_evdata + 1 + df_ev_bar.shape[0])
                    ch1.add_data(val, titles_from_data=False)
                    ch1.set_categories(cat)
                    ws_ev.add_chart(ch1, "H3")

                # Citation histogram
                if df_cit_hist.shape[0] > 0:
                    ch2 = BarChart()
                    ch2.title = "Citation histogram"
                    ch2.x_axis.title = "Citation bin"
                    ch2.y_axis.title = "Num papers"
                    cat = Reference(ws_ev, min_col=1, min_row=r_citdata + 2, max_row=r_citdata + 1 + df_cit_hist.shape[0])
                    val = Reference(ws_ev, min_col=2, min_row=r_citdata + 2, max_row=r_citdata + 1 + df_cit_hist.shape[0])
                    ch2.add_data(val, titles_from_data=False)
                    ch2.set_categories(cat)
                    ws_ev.add_chart(ch2, "H22")

                # Journal distribution
                if df_journal_dist.shape[0] > 0:
                    ch3 = PieChart()
                    ch3.title = "Journal distribution (top 10)"
                    cat = Reference(ws_ev, min_col=1, min_row=r_jrdata + 2, max_row=r_jrdata + 1 + df_journal_dist.shape[0])
                    val = Reference(ws_ev, min_col=2, min_row=r_jrdata + 2, max_row=r_jrdata + 1 + df_journal_dist.shape[0])
                    ch3.add_data(val, titles_from_data=False)
                    ch3.set_categories(cat)
                    ws_ev.add_chart(ch3, "H41")
            except Exception:
                # If charts fail in openpyxl, we still deliver the merged sheet with table + chart data.
                pass

        # Basic filters for readability
        try:
            if _engine == "xlsxwriter":
                # add autofilter lines (header rows)
                ws_ev.autofilter(r_tbl, 0, r_tbl, max(0, df_evidence_table.shape[1]-1))
                writer.sheets["articles"].autofilter(0, 0, max(0, len(df_articles.index)), max(0, df_articles.shape[1]-1))
        except Exception:
            pass

        writer.close()
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        raise

    return excel_path

# ============================================================================
# 10) PIPELINE (with pre-HTML step + Excel export at the end + SNAPSHOT)
# ============================================================================

def run_pipeline(query: str, gene: str, output: str, fresh: str = "Y"):
    """
    End-to-end pipeline with snapshot support:
      fresh=Y:
        1) Write pre-HTML placeholder
        2) Generate search terms
        3) Google search (PMC), pick first template that returns links
        4) Filter by citation count (>= CITATION_MIN)
        5) Download full texts
        6) Extract conclusions & experiments (LLM)
        7) Build final interactive HTML report (overwrite the placeholder)
        8) Export Excel supplementary file (same path as HTML)
        9) Save snapshot (results + statistics + summaries)
      fresh=N:
        A) Load snapshot for (query,gene) if exists
        B) Replay step-by-step by SPLITTING snapshot into stage data
        C) Combine into final HTML + Excel (no LLM/search/network)
    """
    # Common output path logic
    out_dir = "./temp"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(output)
    name, ext = os.path.splitext(base)
    fname = base if ext.lower() in (".html", ".htm", ".xhtml") else base + ".html"
    out_path = os.path.join(out_dir, fname)

    # Resolve snapshot path
    snap_path = get_snapshot_path(query, gene)
    use_snapshot = (fresh or "Y").upper() == "N" and os.path.exists(snap_path)

    # ===============================
    # SNAPSHOT SHORT PATH (replay)
    # ===============================
    if use_snapshot:
        # --- SNAPSHOT REPLAY (split → assemble) ---
        AG = AgentProgress(run="Skill2", total_steps=9)

        # 1) Pre-HTML placeholder
        AG.start_step("Write placeholder HTML", stage="prehtml")
        placeholder_msg = "Replaying from snapshot: reconstructing step outputs…"
        Path(out_path).write_text(build_pre_html_page(placeholder_msg), encoding="utf-8")
        AG.done(f"Placeholder written → {out_path}", path=out_path)
        _maybe_sleep()

        # 2) Load snapshot file
        AG.start_step("Load snapshot", stage="snapshot", detail=os.path.basename(snap_path))
        snap = load_snapshot(snap_path)
        parts = split_snapshot_for_replay(snap)
        n_articles = len(parts.get("results", []))
        n_with_evid = sum(1 for r in parts.get("results", []) if r.get("experiments"))
        AG.done(f"Snapshot loaded (articles={n_articles}, with_evidence={n_with_evid})")
        _maybe_sleep()

        # 3) Restore candidate links
        AG.start_step("Restore candidate links", stage="search")
        cand = parts.get("candidate_links", [])
        AG.update(f"Candidates enumerated: {len(cand)}", stage="search", perc=20, detail=str(min(5, len(cand))))
        AG.done(f"{len(cand)} candidates restored")
        _maybe_sleep()

        # 4) Rebuild citation filter
        AG.start_step("Rebuild citation filter", stage="filter", detail=f"Keep ≥{CITATION_MIN} citations")
        passed = parts.get("passed_filter_links", [])
        AG.done(f"{len(passed)}/{len(cand)} passed filter")
        _maybe_sleep()

        # 5) Rebuild article metadata
        AG.start_step("Rebuild full-text meta", stage="download")
        arts_meta = parts.get("articles_meta", [])
        AG.done(f"Meta restored for {len(arts_meta)} articles")
        _maybe_sleep()

        # 6) Rebuild extraction artifacts
        AG.start_step("Rebuild evidence (conclusions & experiments)", stage="gpt")
        ev_counts = parts.get("evidence_counts", {})
        n_evid_art = sum(1 for url, cnt in ev_counts.items() if (cnt.get('experiments_n', 0) or 0) > 0)
        AG.done(f"Evidence present in {n_evid_art}/{len(arts_meta)} articles")
        _maybe_sleep()

        # 7) Compose statistics/summaries
        AG.start_step("Compose statistics & summaries", stage="stats")
        stats = parts.get("statistics") or {
            "molecular_mechanism_evidence": "No Evidence",
            "cellular_function_evidence": "No Evidence",
            "animal_function_evidence": "No Evidence",
            "clinical_information_evidence": "No Evidence",
            "bioinformatic_analysis_evidence": "No Evidence",
            "query_response": "No relevant experimental data found.",
            "molecular_mechanism_details": "",
            "cellular_function_details": "",
            "animal_function_details": "",
            "clinical_information_details": "",
            "bioinformatic_analysis_details": "",
        }
        sums = parts.get("summaries") or {}
        AG.done("Stats/summaries ready")
        _maybe_sleep()

        # 8) Build HTML (no LLM; reuse stats/summaries)
        AG.start_step("Build interactive HTML report", stage="html")
        payload = generate_html_report(
            results=parts["results"],
            gene=gene,
            query=query,
            output_path=out_path,
            statistics=stats,
            summaries=sums,
            allow_llm=False
        )
        AG.done(f"Report ready → {out_path}", path=out_path)
        _maybe_sleep()

        # 9) Export Excel (no LLM)
        AG.start_step("Export Excel supplement", stage="excel")
        try:
            excel_path = create_excel_supplement_from_results(
                results=parts["results"],
                gene=gene,
                query=query,
                output_html_path=out_path,
                statistics=payload.get("statistics") if payload else stats,
                summaries=payload.get("summaries") if payload else sums,
                allow_llm=False
            )
            AG.done(f"Excel saved → {excel_path}", path=excel_path)
        except Exception as e:
            AG.error(f"Excel export failed (snapshot): {e}")
        return

    # ===============================
    # FRESH FULL PIPELINE (unchanged)
    # ===============================
    AG = AgentProgress(run="Skill2", total_steps=9)

    # 1) Pre-HTML placeholder
    AG.start_step("Write placeholder HTML", stage="prehtml")
    placeholder_msg = "Your task is accepted. The report will update automatically when the analysis finishes."
    Path(out_path).write_text(build_pre_html_page(placeholder_msg), encoding="utf-8")
    AG.done(f"Placeholder written → {out_path}", path=out_path)

    # 2) Search terms
    AG.start_step("Generate search terms", stage="prepare",
                  detail=f"query={query}, gene={gene}")
    terms = generate_search_terms(query, gene)
    log(f"[SEARCH] generated terms: {terms}")
    AG.done(f"{len(terms)} terms ready")

    # 3) Find PMC links (Google first-match policy across templates)
    AG.start_step("Search PMC for candidate papers", stage="search",
                  detail="Google Custom Search (pmc.ncbi.nlm.nih.gov)")
    all_links: List[str] = []
    for idx, term in enumerate(terms, 1):
        AG.update(f"Trying term {idx}/{len(terms)}", stage="search",
                  detail=term, perc=10 + int(idx/len(terms)*15))
        links = list(dict.fromkeys(google_pmc_search(term)))
        if links:
            all_links = links
            break
        else:
            log(f"[{term}] yielded no links, trying next template...")
    if not all_links:
        AG.error("No PMC links found with any template")
        log("[WARN] No PMC links found with any template. Abort.")
        return
    AG.done(f"{len(all_links)} links")

    # 4) Citation filter
    AG.start_step("Filter by citation count", stage="filter",
                  detail=f"Keep papers with ≥{CITATION_MIN} citations")
    citation_counts: Dict[str, int] = {}
    filtered_links: List[str] = []
    for i, link in enumerate(all_links, 1):
        AG.update(f"Fetching metadata {i}/{len(all_links)}", stage="filter", perc=30 + int(i/len(all_links)*15))
        article_data = fetch_full_text(link)
        pmid = article_data.get("pmid") if article_data else None
        if not pmid:
            log(f"[FILTER] Could not extract PMID from {link}, discarded.")
            continue
        citations = get_citation_count(pmid)
        citation_counts[link] = citations
        if citations >= CITATION_MIN:
            filtered_links.append(link)
            log(f"[FILTER] PMID {pmid} accepted (citations={citations}).")
        else:
            log(f"[FILTER] PMID {pmid} discarded (citations={citations} < {CITATION_MIN}).")
    if not filtered_links:
        AG.error("No articles passed the citation filter")
        log("[WARN] No articles passed citation filter. Abort.")
        return
    AG.done(f"{len(filtered_links)} passed filter")

    # 5) Full text fetch
    AG.start_step("Download full texts", stage="download")
    articles = [a for l in filtered_links if (a := fetch_full_text(l))]
    if not articles:
        AG.error("No full text fetched")
        log("[WARN] No full text fetched.")
        return
    AG.done(f"Fetched {len(articles)}")

    # 6) GPT extraction
    AG.start_step("Extract conclusions & experiments", stage="gpt",
                  detail="o4-mini: conclusions → experiments per article")
    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=min(3, len(articles))) as executor:
        futs = [executor.submit(process_article, art, gene, query) for art in articles]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            if r["meta"]["pmc_url"] in citation_counts:
                r["meta"]["citation_count"] = citation_counts[r["meta"]["pmc_url"]]
            results.append(r)
            AG.update(f"Processed {i}/{len(articles)}", stage="gpt",
                      detail=r["meta"]["pmc_url"], perc=65 + int(i/len(articles)*20))
    AG.done(f"{len(results)} articles parsed")

    # 7) Final HTML (overwrite the placeholder at the same path)
    AG.start_step("Build interactive HTML report", stage="html")
    payload = generate_html_report(results, gene, query, out_path, allow_llm=True)
    AG.done(f"Report ready → {out_path}", path=out_path)
    log(f"HTML report: {out_path}")

    # 8) Excel supplementary (same base path)
    AG.start_step("Export Excel supplement", stage="excel")
    try:
        excel_path = create_excel_supplement_from_results(
            results=results,
            gene=gene,
            query=query,
            output_html_path=out_path,
            statistics=payload.get("statistics") if payload else None,
            summaries=payload.get("summaries") if payload else None,
            allow_llm=False
        )
        AG.done(f"Excel saved → {excel_path}", path=excel_path)
        log(f"Excel supplement: {excel_path}")
    except Exception as e:
        AG.error(f"Excel export failed: {e}")

    # 9) Save snapshot
    AG.start_step("Save snapshot", stage="snapshot")
    try:
        save_snapshot(
            path=snap_path,
            query=query,
            gene=gene,
            results=results,
            statistics=(payload or {}).get("statistics"),
            summaries=(payload or {}).get("summaries")
        )
        AG.done(f"Snapshot saved → {snap_path}")
    except Exception as e:
        AG.error(f"Snapshot save failed: {e}")


# ============================================================================
# 11) CLI
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Skill 2: check RBP evidence with citation filtering, experiment extraction, and snapshot reuse"
    )
    ap.add_argument("--query",  required=True, help="User query")
    ap.add_argument("--gene",   required=True, help="Gene symbol")
    ap.add_argument("--output", required=True, help="Output file name (will be written to ./temp)")
    ap.add_argument("--fresh",  choices=["Y","N"], default="N",
                    help="Y = rerun from scratch; N = load snapshot (if exists) and skip all search/API/LLM")
    args = ap.parse_args()
    run_pipeline(args.query, args.gene, args.output, args.fresh)
