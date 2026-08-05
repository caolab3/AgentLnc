#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 3 — RBP/lncRNA tissue-aware phenotype inference
======================================================

This is a **major refactor** based on the new workflow requirements:

Inputs (CLI)
------------
- --rbp (required): RBP gene symbol (used to search Gene_Assay / PPI / GWAS).
- --lncrna (optional): lncRNA gene symbol (can be empty).
- --tissue (required): GTEx tissue name or keyword (used to pick a column from GTEx_Tissue.txt).
- --regulation_type (optional): one of {"all","up_regulation","down_regulation"}; default "all".
- --append_info (optional): free-form user notes appended into hypothesis prompt.
- --output (required): output stem (writes to ./temp/<stem>.html and ./temp/<stem>.xlsx)
- --fresh (optional): Y = run fresh, N = load snapshot if available (default N)

Pipeline (high level)
---------------------
1) Base data collection:
   - RBP Gene_Assay (local table)
   - RBP GWAS traits (RBPbase)
   - RBP PPI partners (BioGRID/HINT merged table)
   - Tissue expression (GTEx_Tissue.txt; tissue column selected by keyword)
   - If tissue is liver-like: include DRS regulation signals (p<0.05) for involved genes

2) Hypothesis proposal (GPT):
   - Build a structured prompt including: RBP, lncRNA (if any), tissue, user notes,
     GWAS traits, assay summary, PPI summary, and tissue expression.
   - Based on regulation_type, propose one integrated mechanism per direction:
       * up_regulation: 1 hypothesis
       * down_regulation: 1 hypothesis
       * all: 2 hypotheses (1 up + 1 down)
   - Each hypothesis includes a GWAS trait most likely connected to the direction of RBP regulation.

3) Evidence retrieval (unchanged core):
   - Build Google/Vertex (PMC) queries per hypothesis and collect PMC URLs.
   - **Also read the literature already referenced by Gene_Assay (PMCID/PMID)**.
   - Fetch full text from PMC and use GPT extraction to pull experiments.

4) Synthesis:
   - For each hypothesis, select the DB rows actually used, rate (High/Medium/Low),
     and compose a narrative.

5) Output:
   - HTML report (hypothesis cards)
   - Excel results file

Notes
-----
- DRS is only used when the tissue input contains "liver" (case-insensitive).
- GTEx expression is not filtered; DRS regulation keeps only p <= 0.05.
"""

from __future__ import annotations

from pathlib import Path
import os, json, re, sys, math, time, argparse, datetime, html as _html, ast, gzip, hashlib
from typing import List, Dict, Any, Tuple, Set, Optional
from numbers import Number

import pandas as pd
from bs4 import BeautifulSoup
import requests

# =========================
# 0) Configuration & Constants
# =========================
def _load_profile(profile_path: str | None = None) -> dict:
    path = Path(profile_path or os.getenv("PROFILE_JSON", Path(__file__).with_name("profile.json")))
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"[Config] Profile file not found: {path}\n"
            f"You may set env var PROFILE_JSON to override path."
        )
    except json.JSONDecodeError as e:
        raise SystemExit(f"[Config] Failed to parse {path}: {e}")
    for k in list(cfg.keys()):
        if os.getenv(k):
            cfg[k] = os.getenv(k)
    return cfg

_CFG = _load_profile()

# --- Google Web Search via Vertex AI Discovery Engine (searchLite) ---
VERTEX_PROJECT = (_CFG.get("VERTEX_PROJECT") or "").strip()
VERTEX_ENGINE_ID = (_CFG.get("VERTEX_ENGINE_ID") or "").strip()
VERTEX_API_KEY   = (_CFG.get("VERTEX_API_KEY") or "").strip()

VERTEX_LOCATION       = (_CFG.get("VERTEX_LOCATION") or "global").strip()
VERTEX_COLLECTION     = (_CFG.get("VERTEX_COLLECTION") or "default_collection").strip()
VERTEX_SERVING_CONFIG = (_CFG.get("VERTEX_SERVING_CONFIG") or "default_search").strip()

# Backward compatible aliases
GOOGLE_API_KEY = VERTEX_API_KEY
GOOGLE_CX      = VERTEX_ENGINE_ID

# LLM provider: prefer an OpenAI key when supplied, otherwise retain the
# original Azure o4-mini deployment.
from openai import OpenAI, AzureOpenAI
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or _CFG.get("OPENAI_API_KEY") or "").strip()
OPENAI_REASONING_MODEL = (
    os.getenv("OPENAI_REASONING_MODEL")
    or _CFG.get("OPENAI_REASONING_MODEL")
    or "o4-mini"
).strip()
USE_OPENAI = bool(OPENAI_API_KEY)

AZURE_ENDPOINT     = (_CFG.get("AZURE_O4MINI_ENDPOINT") or "").strip()
AZURE_API_KEY      = (_CFG.get("AZURE_O4MINI_API_KEY") or "").strip()
AZURE_API_VERSION  = _CFG.get("AZURE_O4MINI_API_VERSION", "2025-01-01-preview")
AZURE_MODEL_DEPLOY = "o4-mini"

if USE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = OPENAI_REASONING_MODEL
    LLM_PROVIDER = "openai"
else:
    missing = [
        name for name, value in (
            ("AZURE_O4MINI_ENDPOINT", AZURE_ENDPOINT),
            ("AZURE_O4MINI_API_KEY", AZURE_API_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "[Config] Set OPENAI_API_KEY, or provide the Azure fallback values: "
            + ", ".join(missing)
        )
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    MODEL_NAME = AZURE_MODEL_DEPLOY
    LLM_PROVIDER = "azure"

# Entrez (PubMed→PMC)
from Bio import Entrez
Entrez.email = _CFG.get("ENTREZ_EMAIL", "you@example.com")

# Paths (relative to this script)
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "database")

# DRS (used only when tissue is liver-like)
DRS_DIR          = os.path.join(DATA_DIR, "DRS_regulation")
HUMAN_DRS        = os.path.join(DRS_DIR, "Human_DRS_RNAseq_Regulation.txt")
MOUSE_DRS        = os.path.join(DRS_DIR, "Mouse_DRS_RNAseq_Regulation.txt")

# Local DBs
PPI_TABLE_LOCAL  = "BIOGRID_HINT_Merge_PPI"
GENE_ASSAY_FILE  = os.path.join(DATA_DIR, "Gene_Assay.txt")
RBPBASE_FILE     = os.path.join(DATA_DIR, "RBPbase.txt")

# NEW: GTEx tissue expression table
GTEX_TISSUE_FILE = os.path.join(DATA_DIR, "GTEx_Tissue.txt")

TEMP_DIR         = os.path.join(SCRIPT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Snapshot system ---
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "snapshot")
SNAPSHOT_VERSION = "P4SNAPv3_MECHANISM_CLUSTER"  # kept for snapshot-format compatibility
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

SNAPSHOT_REPLAY_DELAY = float(
    os.getenv(
        "SKILL3_SNAPSHOT_REPLAY_DELAY",
        os.getenv("P4_SNAPSHOT_REPLAY_DELAY", os.getenv("P2_SNAPSHOT_REPLAY_DELAY", "0.5")),
    )
)
def _maybe_sleep():
    if SNAPSHOT_REPLAY_DELAY > 0:
        time.sleep(SNAPSHOT_REPLAY_DELAY)

def _slugify(s: str, maxlen: int=40) -> str:
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    s = re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()
    return s[:maxlen] or "na"

def _canonical_inputs_for_snapshot(rbp: str,
                                  lncrna: str,
                                  tissue: str,
                                  regulation_type: str,
                                  append_info: str) -> Dict[str, str]:
    return {
        "rbp": (rbp or '').strip(),
        "lncrna": (lncrna or '').strip(),
        "tissue": re.sub(r'\s+', ' ', (tissue or '').strip()),
        "regulation_type": (regulation_type or 'all').strip(),
        "append_info": re.sub(r'\s+', ' ', (append_info or '').strip()),
    }

def _snapshot_key(ci: Dict[str, str]) -> str:
    payload = {"v": SNAPSHOT_VERSION, "inputs": ci}
    j = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:16]

def get_snapshot_path(rbp: str, lncrna: str, tissue: str, regulation_type: str, append_info: str) -> str:
    ci = _canonical_inputs_for_snapshot(rbp, lncrna, tissue, regulation_type, append_info)
    key = _snapshot_key(ci)
    fname = (
        f"{SNAPSHOT_VERSION}_{key}"
        f"__rbp={_slugify(ci['rbp'])}"
        f"__lnc={_slugify(ci['lncrna'])}"
        f"__tissue={_slugify(ci['tissue'])}"
        f"__reg={_slugify(ci['regulation_type'])}.json.gz"
    )
    return os.path.join(SNAPSHOT_DIR, fname)

def save_snapshot(path: str,
                  rbp: str,
                  lncrna: str,
                  tissue: str,
                  regulation_type: str,
                  append_info: str,
                  hypotheses: List[Dict[str, Any]],
                  registry: Dict[str, Dict[str, Any]],
                  per_pair: Dict[str, Dict[str, Any]],
                  statistics: Dict[str, Any] | None = None) -> None:
    snap = {
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": _canonical_inputs_for_snapshot(rbp, lncrna, tissue, regulation_type, append_info),
        "runtime_meta": {"python": sys.version.split()[0]},
        "hypotheses": hypotheses,
        "registry": registry,
        "per_pair": per_pair,
        "statistics": statistics or {},
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)
    log(f"[SNAPSHOT] Saved → {path}")

def load_snapshot(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("hypotheses", [])
    data.setdefault("registry", {})
    data.setdefault("per_pair", {})
    data.setdefault("statistics", {})
    return data

VERBOSE = False

def log(msg: str):
    print(f"[LOG] {msg}", file=sys.stderr, flush=True)

def log_debug(msg: str):
    if VERBOSE:
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def _esc(s: Any) -> str:
    return _html.escape(str(s)) if s is not None else ""

# =========================
# 1) Lightweight Progress Helper
# =========================
import json as _json, time as _time
class AgentProgress:
    def __init__(self, run: str, total_steps: int):
        self.run = run
        self.total = max(1, int(total_steps))
        self.step = 0
        self.t0 = _time.time()
    def _emit(self, payload: dict):
        payload.setdefault("run", self.run)
        payload.setdefault("t", round(_time.time()-self.t0, 2))
        s = "@@PROGRESS " + _json.dumps(payload, ensure_ascii=False)
        print(s, flush=True)
    def start_step(self, title: str, stage: str, detail: str=""):
        self.step += 1
        self._emit({"kind":"start","stage":stage,"step":self.step,"total":self.total,
                    "perc": int((self.step-1)/self.total*100),"title":title,"detail":detail})
    def update(self, msg: str, stage: str="", perc: int | None=None, **extra):
        self._emit({"kind":"update","stage":stage or "","step":self.step,"total":self.total,
                    "perc": int(self.step/self.total*100) if perc is None else int(perc),
                    "title":msg,"detail":extra.get("detail",""),"extra":extra})
    def done(self, msg: str="", **extra):
        self._emit({"kind":"done","stage":"done","step":self.step,"total":self.total,
                    "perc": int(self.step/self.total*100),"title":msg,"detail":extra.get("detail",""),
                    "extra":extra})
    def error(self, err: Exception | str, **extra):
        self._emit({"kind":"error","stage":"error","step":self.step,"total":self.total,
                    "perc": int(self.step/self.total*100), "title":"Error",
                    "detail": str(err), "extra": extra})

# =========================
# 2) Local Data Loading
# =========================
def is_liver_tissue(tissue_query: str) -> bool:
    return "liver" in (tissue_query or "").strip().lower()

def load_drs_table(path: str) -> pd.DataFrame:
    """Load DRS table; set index to uppercase 'Gene Name' if present."""
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if "Gene Name" in df.columns:
        df["Gene Name"] = df["Gene Name"].astype(str).str.strip().str.upper()
        df = df.set_index("Gene Name")
    return df

def _numeric(val):
    if val is None:
        return None
    if isinstance(val, Number):
        num = float(val)
        return None if math.isnan(num) else num
    if isinstance(val, dict):
        for k in ("p","pval","p_value"):
            if k in val and isinstance(val[k], Number):
                num = float(val[k])
                return None if math.isnan(num) else num
    try:
        num = float(val)
        return None if math.isnan(num) else num
    except Exception:
        return None

def filter_significant_regulation(raw: Dict[str, Any], alpha: float=0.05) -> Dict[str, Any]:
    """
    Keep only regulators whose '*_p' value ≤ alpha; include paired '*_log2fc' and
    optional expression magnitudes if present (log10tpm/tpm/expr/expression).
    """
    if not raw:
        return {}
    keep: Dict[str, Any] = {}
    lower_map = {str(k).lower(): k for k in raw.keys()}
    def _get(key: str):
        return raw.get(lower_map.get(key.lower(), key))
    for k, v in raw.items():
        if str(k).endswith("_p"):
            p = _numeric(v)
            if p is None or p > alpha:
                continue
            base = str(k)[:-2]
            keep[f"{base}_p"] = p
            logk = f"{base}_log2fc"
            if logk in raw:
                try:
                    keep[logk] = float(raw[logk])
                except Exception:
                    pass
            for suf in ["_log10tpm", "_tpm", "_expr", "_expression"]:
                cand = _get(base + suf)
                if _numeric(cand) is not None:
                    keep[base + suf] = float(cand)
    return keep

def convert_to_mouse_symbol(gene: str) -> str:
    return gene[:1].upper() + gene[1:].lower() if gene else gene

def gather_drs_for_gene(gene: str,
                        drs_human: pd.DataFrame,
                        drs_mouse: pd.DataFrame,
                        alpha: float=0.05) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    """
    Return (human_sig, mouse_sig). If DRS tables are empty, returns ({},{}).
    """
    if drs_human is None or drs_human.empty:
        info_h = {}
    else:
        g_up = gene.upper().strip()
        info_h_raw = drs_human.loc[g_up].to_dict() if g_up in drs_human.index else {}
        info_h = filter_significant_regulation(info_h_raw, alpha=alpha)

    if drs_mouse is None or drs_mouse.empty:
        info_m = {}
    else:
        g_up = gene.upper().strip()
        g_mouse = convert_to_mouse_symbol(gene)
        if g_up in drs_mouse.index:
            info_m_raw = drs_mouse.loc[g_up].to_dict()
        elif g_mouse.upper() in drs_mouse.index:
            info_m_raw = drs_mouse.loc[g_mouse.upper()].to_dict()
        else:
            info_m_raw = {}
        info_m = filter_significant_regulation(info_m_raw, alpha=alpha)

    return info_h, info_m

# --- GTEx tissue expression ---
def _normalize_tissue_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s

def choose_gtex_tissue_column(file_path: str, tissue_query: str) -> Tuple[str | None, List[str]]:
    """
    Choose a GTEx tissue column based on user tissue query (keyword match).
    Returns: (selected_column, matched_columns)
    Selection rules:
      1) exact match (case-insensitive, normalized) if exists
      2) substring match (col contains query or query contains col)
      3) if multiple matches: choose shortest column name, then alphabetical
    """
    if not os.path.isfile(file_path):
        return None, []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    if not header:
        return None, []
    cols = header.split("\t")
    if not cols:
        return None, []
    # GTEx format: Name, Description, <tissues...>
    tissue_cols = cols[2:] if len(cols) > 2 else cols
    qn = _normalize_tissue_token(tissue_query)
    if not qn:
        return None, []
    norm_map = {c: _normalize_tissue_token(c) for c in tissue_cols}
    exact = [c for c in tissue_cols if norm_map[c] == qn]
    if exact:
        return exact[0], exact
    matches = []
    for c in tissue_cols:
        cn = norm_map[c]
        if qn in cn or cn in qn:
            matches.append(c)
    if not matches:
        # fallback: token overlap
        qset = set(qn.split("_"))
        for c in tissue_cols:
            cset = set(norm_map[c].split("_"))
            if qset & cset:
                matches.append(c)
    if not matches:
        return None, []
    # deterministic pick: shortest then alphabetical
    matches_sorted = sorted(matches, key=lambda x: (len(x), x))
    return matches_sorted[0], matches_sorted

def load_gtex_expression_dict(file_path: str, tissue_query: str) -> Tuple[Dict[str, float], str | None, List[str]]:
    """
    Load GTEx_Tissue.txt and build {GENE_SYMBOL: expression_value} for the selected tissue column.
    Uses 'Description' column as gene symbol. Returns (expr_map, selected_col, matched_cols).
    """
    selected_col, matched_cols = choose_gtex_tissue_column(file_path, tissue_query)
    if selected_col is None:
        return {}, None, matched_cols
    try:
        usecols = ["Description", selected_col]
        df = pd.read_csv(file_path, sep="\t", usecols=usecols, dtype={"Description": str})
    except Exception as e:
        log(f"[GTEx] load failed: {e}")
        return {}, selected_col, matched_cols
    df = df.fillna(0)
    # Ensure numeric conversion
    try:
        df[selected_col] = pd.to_numeric(df[selected_col], errors="coerce").fillna(0.0)
    except Exception:
        pass
    expr: Dict[str, float] = {}
    for _, row in df.iterrows():
        sym = str(row.get("Description", "")).strip()
        if not sym:
            continue
        val = row.get(selected_col, 0.0)
        try:
            fval = float(val)
        except Exception:
            fval = 0.0
        key = sym.upper()
        # If duplicates exist, keep max (more permissive)
        if key not in expr or fval > expr[key]:
            expr[key] = fval
    return expr, selected_col, matched_cols

def get_gtex_expr_for_genes(expr_map: Dict[str, float], genes: List[str]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for g in genes:
        if not g:
            continue
        out[g.upper()] = expr_map.get(g.upper())
    return out

# --- Gene_Assay / RBPbase ---
def load_gene_assay_df(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()
    if "gene_symbol" in df.columns:
        df["gene_symbol"] = df["gene_symbol"].astype(str).str.upper()
    return df

def filter_gene_assay_for_gene(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    if df.empty or "gene_symbol" not in df.columns:
        return pd.DataFrame()
    sub = df[df["gene_symbol"] == gene.upper()].copy()
    return sub

def load_rbpbase_df(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()
    for c in ("UNIQUE","gene","gene_symbol","symbol"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    return df

def filter_rbpbase_for_gene(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    gene = gene.upper()
    for c in ("UNIQUE","gene","gene_symbol","symbol"):
        if c in df.columns:
            sub = df[df[c] == gene]
            if not sub.empty:
                return sub
    return pd.DataFrame()

# --- PPI query (prefer local_db_handler, fallback to TSV/CSV) ---
def _fallback_ppi_scan(table_base: str, alias: str) -> Tuple[List[str], List[List[str]]]:
    cands = [
        os.path.join(DATA_DIR, f"{table_base}.tsv"),
        os.path.join(DATA_DIR, f"{table_base}.txt"),
        os.path.join(DATA_DIR, f"{table_base}.csv"),
    ]
    for p in cands:
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p, sep="\t" if p.endswith((".tsv",".txt")) else ",", dtype=str).fillna("")
                cols = df.columns.tolist()
                m = df[(df.get("Gene Name A","")==alias) | (df.get("Gene Name B","")==alias)]
                return cols, m.values.tolist()
            except Exception:
                continue
    return [], []

def query_ppi_partners(gene: str) -> List[Dict[str, str]]:
    aliases = {
        gene.upper(),
        convert_to_mouse_symbol(gene).upper(),
        gene,
        convert_to_mouse_symbol(gene),
    }
    try:
        from local_db_handler import query_database_local  # type: ignore
        cols_ref, all_rows = None, []
        for al in aliases:
            search_conditions = [
                ('Gene Name A', lambda _g, a=al: a),
                ('Gene Name B', lambda _g, a=al: a),
            ]
            cols, rows = query_database_local(al, PPI_TABLE_LOCAL, search_conditions)  # noqa: F405
            if cols and rows:
                cols_ref = cols if cols_ref is None else cols_ref
                all_rows.extend(rows)
        uniq = {tuple(r) for r in all_rows}
        return [dict(zip(cols_ref, r)) for r in uniq] if cols_ref else []
    except Exception:
        cols_ref, rows = None, []
        for al in aliases:
            c, r = _fallback_ppi_scan(PPI_TABLE_LOCAL, al)
            if c and r:
                cols_ref = c if cols_ref is None else cols_ref
                rows.extend(r)
        uniq = {tuple(r) for r in rows}
        return [dict(zip(cols_ref, r)) for r in uniq] if cols_ref else []

def summarize_ppi_partners(ppi_records: List[Dict[str, str]], rbp: str, max_genes: int = 50) -> List[Dict[str, Any]]:
    """
    Summarize PPI partners as [{gene: 'TP53', n_pairs: 3, example_pairs:[...]}], limited.
    """
    rbpU = rbp.upper()
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    for r in ppi_records:
        a = str(r.get("Gene Name A","")).upper()
        b = str(r.get("Gene Name B","")).upper()
        if not a or not b:
            continue
        partner = b if a == rbpU else (a if b == rbpU else "")
        if not partner or partner == rbpU:
            continue
        counts[partner] = counts.get(partner, 0) + 1
        examples.setdefault(partner, [])
        if len(examples[partner]) < 3:
            examples[partner].append(f"{a}–{b}")
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_genes]
    out = []
    for g, n in top:
        out.append({"gene": g, "n_pairs": n, "example_pairs": examples.get(g, [])})
    return out

# =========================
# 3) Google / PMC (rate-limited)
# =========================
class _RateLimiter:
    def __init__(self, qps: float = 6.0):
        self.qps = max(0.1, float(qps))
        self.min_interval = 1.0 / self.qps
        self._last_t = 0.0
    def wait(self):
        now = time.time()
        dt = now - self._last_t
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_t = time.time()

_GOOGLE_LIMITER = _RateLimiter(qps=6.0)

def google_pmc_search(term: str, limit: int = 10, quota_user: str | None = None) -> List[str]:
    """
    Vertex AI Discovery Engine searchLite to find PMC links.
    Return: List[str] of PMC URLs (deduplicated), up to `limit`.
    """
    _GOOGLE_LIMITER.wait()
    project   = (VERTEX_PROJECT or "").strip()
    engine_id = ((GOOGLE_CX or "") or (VERTEX_ENGINE_ID or "")).strip()
    key       = ((GOOGLE_API_KEY or "") or (VERTEX_API_KEY or "")).strip()
    location  = (VERTEX_LOCATION or "global").strip()
    collection = (VERTEX_COLLECTION or "default_collection").strip()
    serving_config = (VERTEX_SERVING_CONFIG or "default_search").strip()

    if not project or not engine_id or not key:
        log("[VertexSearch] Missing VERTEX_PROJECT / VERTEX_ENGINE_ID / VERTEX_API_KEY; returning empty list.")
        return []

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
        for item in (resp.get("results", []) or []):
            d = (item.get("document", {}).get("derivedStructData") or {})
            link = (d.get("link") or "").split("#")[0]
            if link:
                yield {"link": link}

    target = max(1, int(limit or 0))
    page_size = 10
    max_results = max(target, page_size)

    links: List[str] = []
    seen = set()

    for offset in range(0, max_results, page_size):
        payload = {
            "servingConfig": serving_cfg_path,
            "query": term,
            "pageSize": int(page_size),
            "offset": int(offset),
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
        if len(links) >= target or len(items) < page_size:
            break
    return links[:target]

def fetch_full_text(pmc_url: str) -> Dict[str, Any] | None:
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/117.0 Safari/537.36'
    }
    try:
        resp = requests.get(pmc_url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        log_debug(f"[PMC fetch] {pmc_url} error: {e}")
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')

    def _meta(name):
        tag = soup.find("meta", {"name": name})
        return tag.get("content", "").strip() if tag and tag.get("content") else ""

    pmid    = _meta("citation_pmid")
    date    = _meta("citation_publication_date")
    journal = _meta("citation_journal_title")
    title   = _meta("citation_title")

    article_section = soup.find("section", attrs={"aria-label": "Article content"})
    full_text = (
        article_section.get_text(separator=" ", strip=True) if article_section
        else soup.get_text(separator=" ", strip=True)
    )
    full_text = re.sub(r"\s+", " ", full_text)
    return {
        "pmid": pmid,
        "pmc_url": pmc_url,
        "date": date,
        "journal": journal,
        "title": title,
        "text": full_text,
    }

def pmid_to_pmcid(pmid: str) -> str | None:
    """PubMed PMID → PMCID using Entrez elink."""
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc")
        rec = Entrez.read(handle)
        handle.close()
        lst = rec[0].get("LinkSetDb", [])
        if lst and lst[0].get("Link"):
            pmcid = lst[0]["Link"][0].get("Id")
            if pmcid and not str(pmcid).upper().startswith("PMC"):
                pmcid = "PMC" + str(pmcid)
            return pmcid
    except Exception as e:
        log_debug(f"[Entrez elink] PMID→PMC failed: {e}")
    return None

def pmcid_to_pmc_url(pmcid: str) -> str:
    pmc = (pmcid or "").strip()
    if not pmc:
        return ""
    if not pmc.upper().startswith("PMC"):
        pmc = "PMC" + pmc
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"

def collect_articles_from_terms(terms: List[str], per_term_limit: int = 6) -> List[str]:
    links: List[str] = []
    for t in terms:
        ls = google_pmc_search(t, limit=per_term_limit)
        for l in ls:
            if l not in links:
                links.append(l)
    return links

# =========================
# 4) GPT Subtasks & Helpers
# =========================
FIG_RE = re.compile(r'\b(?:Figure|Fig\.?|Table)\b', re.I)
TRAIT_SPLIT_RE = re.compile(r"[;,/|、；]+")

def clean_json_response(raw: str) -> str:
    m = re.search(r'(\{.*?\}|\[.*?\])', raw, re.S)
    return m.group(0) if m else raw.strip()

def canonicalize_verdict(v: str) -> str:
    s = (v or "").strip().lower()
    if s in {"high","strong","strongly supported","strong support","very strong","robust"}:
        return "High"
    if s in {"medium","moderate","supported","partial","partially supported","some support"}:
        return "Medium"
    if s in {"low","weak","not supported","little","insufficient","inconclusive","minimal"}:
        return "Low"
    return "Low"

def extract_figure_table_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[\.\?!。！？])\s+', text)
    keep = [s.strip() for s in sentences if FIG_RE.search(s)]
    return keep or [text[:4000]]

def gpt_generate_regulation_hypotheses(payload: Dict[str, Any],
                                       regulation_type: str = "all",
                                       n_each: int = 1) -> List[Dict[str, Any]]:
    """
    Mechanism-first hypothesis generation (NEW workflow)
    ---------------------------------------------------
    Instead of generating many hypotheses each tied to a single GWAS trait, we generate
    **ONE integrated mechanistic hypothesis per direction** (UP vs DOWN).

    - regulation_type='up_regulation'  -> exactly 1 item (UP1)
    - regulation_type='down_regulation'-> exactly 1 item (DOWN1)
    - regulation_type='all'            -> exactly 2 items (UP1 then DOWN1)

    Backward-compatible keys kept:
      - id, direction, trait, hypothesis, rationale, key_genes

    New keys added:
      - axis_label: short label for the mechanism/trait-cluster
      - mechanism: longer mechanistic hypothesis (2–5 sentences)
      - gwas_traits: list of GWAS traits (subset) that the mechanism could explain
      - assays: key validation assays + expected outputs
      - pmc_queries: short PMC/Google queries for literature retrieval (2–4)
      - mechanism_keywords: 3–8 keywords useful for search/extraction focus
    """
    regulation_type = (regulation_type or "all").strip().lower()
    if regulation_type not in {"all", "up_regulation", "down_regulation"}:
        regulation_type = "all"

    # Always force 1 per direction for the new workflow (n_each kept only for backward compatibility)
    n_each = 1

    sys_prompt = (
        "You are a biomedical domain expert (RNA biology, RBP/lncRNA regulation, GWAS interpretation).\n"
        "Task: Given ALL provided evidence (GWAS traits, assay evidence, PPI partners, tissue expression, optional DRS),\n"
        "infer the MOST PLAUSIBLE unifying MOLECULAR MECHANISM for each requested direction of RBP regulation.\n\n"
        "Key framing:\n"
        "  - Traits are downstream phenotypes. Multiple GWAS traits may arise from ONE shared mechanism.\n"
        "  - If lncRNA is provided, hypotheses must be framed as: lncRNA -> (UP/DOWN) RBP -> mechanism -> trait cluster.\n"
        "  - You must explicitly list which GWAS traits (from provided candidates) are consistent with the inferred mechanism.\n"
        "  - Propose a small set of KEY assays (3–6) to validate the mechanism, with expected results.\n\n"
        "Output requirements:\n"
        "  - If regulation_type='up_regulation': output exactly 1 item, direction='up_regulation', id='UP1'.\n"
        "  - If regulation_type='down_regulation': output exactly 1 item, direction='down_regulation', id='DOWN1'.\n"
        "  - If regulation_type='all': output exactly 2 items: first UP1 then DOWN1.\n"
        "  - axis_label must be short (≤ 12 words) and describe the mechanism/phenotype cluster.\n"
        "  - gwas_traits must be a list of 6–20 traits (pick the most relevant subset), each with optional score.\n"
        "  - pmc_queries must be 2–4 short queries (≤ 10 words) to find mechanistic PMC papers.\n"
        "  - key_genes should include RBP, lncRNA (if any), and up to ~6 relevant PPI/pathway genes.\n\n"
        "Return STRICT JSON OBJECT only (no markdown):\n"
        "{\n"
        '  \"hypotheses\": [\n'
        "    {\n"
        '      \"id\": \"UP1\",\n'
        '      \"direction\": \"up_regulation\",\n'
        '      \"axis_label\": \"short mechanism / trait-cluster label\",\n'
        '      \"mechanism\": \"2-5 sentences; unifying mechanism\",\n'
        '      \"gwas_traits\": [ {\"trait\":\"...\",\"score\":0.0,\"why\":\"optional\"} ],\n'
        '      \"mechanism_keywords\": [\"keyword1\",\"keyword2\"],\n'
        '      \"assays\": [ {\"assay\":\"...\",\"readout\":\"...\",\"expected\":\"...\"} ],\n'
        '      \"pmc_queries\": [\"...\",\"...\"],\n'
        '      \"key_genes\": [\"RBP\",\"LNC\",\"PPI1\"],\n'
        '      \"rationale\": \"1-3 sentences, cite only evidence types present in payload\"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user = json.dumps(
        {
            "regulation_type": regulation_type,
            "payload": payload,
        },
        ensure_ascii=False
    )

    arr: List[Any] = []
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            arr = data.get("hypotheses") or data.get("items") or []
        elif isinstance(data, list):
            arr = data
        else:
            arr = []
        if not isinstance(arr, list):
            arr = []
    except Exception as e:
        log(f"[gpt_generate_regulation_hypotheses] {e}")
        arr = []

    # -------------------------
    # Fallback (never return empty)
    # -------------------------
    if not arr:
        gwas = payload.get("gwas_candidates") or []
        # sort by score desc (None/NaN last)
        def _score(it):
            sc = it.get("score")
            try:
                return float(sc)
            except Exception:
                return float("-inf")
        gwas_sorted = sorted(
            [it for it in gwas if str(it.get("trait", "")).strip()],
            key=_score,
            reverse=True
        )
        top_traits = []
        seen = set()
        for it in gwas_sorted:
            t = str(it.get("trait", "")).strip()
            k = normalize_text(t)
            if not t or k in seen:
                continue
            seen.add(k)
            top_traits.append({"trait": t, "score": it.get("score")})
            if len(top_traits) >= 12:
                break
        if not top_traits:
            top_traits = [{"trait": "(no GWAS trait provided)", "score": None}]

        rbp = str(payload.get("rbp", "")).upper() or "RBP"
        lnc = str(payload.get("lncrna", "")).upper().strip()
        tissue = str(payload.get("tissue_query", "")).strip() or "tissue"

        def _mk(direction: str) -> Dict[str, Any]:
            if direction == "up_regulation":
                hid = "UP1"
                axis = "RBP-up: post-transcriptional program"
                mech = (
                    f"{lnc + ' ' if lnc else ''}may increase {rbp} expression in {tissue}, "
                    f"altering a shared post-transcriptional regulatory program (RNA binding/splicing/stability), "
                    f"which could drive a cluster of GWAS phenotypes."
                )
            else:
                hid = "DOWN1"
                axis = "RBP-down: loss of RNA processing control"
                mech = (
                    f"{lnc + ' ' if lnc else ''}may decrease {rbp} expression in {tissue}, "
                    f"reducing RNA binding/splicing/stability control and shifting downstream gene expression, "
                    f"which could drive a cluster of GWAS phenotypes."
                )
            assays = [
                {"assay": "RBP eCLIP/CLIP-seq (± lncRNA perturbation)", "readout": "binding sites/targets", "expected": "UP: increased occupancy on key targets; DOWN: reduced occupancy"},
                {"assay": "RNA-seq (splicing + DE) after RBP OE/KD", "readout": "isoform usage / expression", "expected": "Directional changes consistent with mechanism keywords"},
                {"assay": "RIP-qPCR for candidate target RNAs", "readout": "enrichment", "expected": "Binding/enrichment tracks with RBP level"},
                {"assay": "Reporter assay (3'UTR / splice reporter)", "readout": "reporter activity", "expected": "RBP-dependent change in reporter output"},
            ]
            pmc_q = [f"{rbp} RNA binding mechanism", f"{rbp} splicing regulation"]
            if lnc:
                pmc_q = [f"{lnc} {rbp} regulation", pmc_q[0], pmc_q[1]]
            return {
                "id": hid,
                "direction": direction,
                "axis_label": axis,
                "mechanism": mech,
                "gwas_traits": top_traits,
                "mechanism_keywords": ["RNA binding", "splicing", "mRNA stability"],
                "assays": assays[:5],
                "pmc_queries": pmc_q[:3],
                "key_genes": [rbp] + ([lnc] if lnc else []),
                "rationale": "Fallback: uses provided GWAS candidates and generic RBP mechanism framing.",
            }

        out_fb: List[Dict[str, Any]] = []
        if regulation_type in {"all", "up_regulation"}:
            out_fb.append(_mk("up_regulation"))
        if regulation_type in {"all", "down_regulation"}:
            out_fb.append(_mk("down_regulation"))

        # add backward-compatible aliases
        for it in out_fb:
            it["trait"] = it.get("axis_label", "")
            it["hypothesis"] = it.get("mechanism", "")
        return out_fb

    # -------------------------
    # Normalize & enforce schema
    # -------------------------
    normed: List[Dict[str, Any]] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        direction = str(it.get("direction", "")).strip().lower()
        if direction not in {"up_regulation", "down_regulation"}:
            # infer from id
            direction = "up_regulation" if str(it.get("id", "")).upper().startswith("UP") else "down_regulation"

        axis_label = re.sub(r"\s+", " ", str(it.get("axis_label") or it.get("trait") or "").strip())
        mechanism = re.sub(r"\s+", " ", str(it.get("mechanism") or it.get("hypothesis") or "").strip())
        rationale = re.sub(r"\s+", " ", str(it.get("rationale", "")).strip())

        # key_genes
        key_genes = it.get("key_genes", [])
        if not isinstance(key_genes, list):
            key_genes = []
        key_genes = [re.sub(r"\s+", " ", str(g).strip()).upper() for g in key_genes if str(g).strip()]

        # gwas_traits
        gwas_traits = it.get("gwas_traits") or it.get("traits") or it.get("associated_traits") or []
        if isinstance(gwas_traits, list):
            gt_norm = []
            seen_t = set()
            for x in gwas_traits:
                if isinstance(x, str):
                    tname = re.sub(r"\s+", " ", x).strip()
                    if not tname:
                        continue
                    k = normalize_text(tname)
                    if k in seen_t:
                        continue
                    seen_t.add(k)
                    gt_norm.append({"trait": tname, "score": None, "why": ""})
                elif isinstance(x, dict):
                    tname = re.sub(r"\s+", " ", str(x.get("trait") or x.get("name") or "").strip())
                    if not tname:
                        continue
                    k = normalize_text(tname)
                    if k in seen_t:
                        continue
                    seen_t.add(k)
                    sc = x.get("score")
                    try:
                        sc2 = float(sc) if sc is not None else None
                    except Exception:
                        sc2 = None
                    gt_norm.append({"trait": tname, "score": sc2, "why": re.sub(r"\s+", " ", str(x.get("why","")).strip())})
                if len(gt_norm) >= 25:
                    break
            gwas_traits = gt_norm
        else:
            gwas_traits = []

        # assays
        assays = it.get("assays") or it.get("predicted_assays") or []
        if not isinstance(assays, list):
            assays = []
        assays_norm = []
        for a in assays:
            if isinstance(a, str):
                assays_norm.append({"assay": a.strip(), "readout": "", "expected": ""})
            elif isinstance(a, dict):
                assays_norm.append({
                    "assay": re.sub(r"\s+", " ", str(a.get("assay","")).strip()),
                    "readout": re.sub(r"\s+", " ", str(a.get("readout","")).strip()),
                    "expected": re.sub(r"\s+", " ", str(a.get("expected") or a.get("expected_result") or "").strip()),
                })
            if len(assays_norm) >= 10:
                break

        # queries
        pmc_queries = it.get("pmc_queries") or it.get("queries") or []
        if not isinstance(pmc_queries, list):
            pmc_queries = []
        pmc_queries = [re.sub(r"\s+", " ", str(q)).strip() for q in pmc_queries if str(q).strip()][:6]

        keywords = it.get("mechanism_keywords") or it.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [re.sub(r"\s+", " ", str(k)).strip() for k in keywords if str(k).strip()][:12]

        hid = str(it.get("id") or ("UP1" if direction == "up_regulation" else "DOWN1")).strip()
        hid = ("UP1" if direction == "up_regulation" else "DOWN1") if regulation_type != "all" else hid

        normed.append({
            "id": hid,
            "direction": direction,
            "axis_label": axis_label or "(unspecified axis)",
            "mechanism": mechanism or "",
            "gwas_traits": gwas_traits,
            "mechanism_keywords": keywords,
            "assays": assays_norm,
            "pmc_queries": pmc_queries,
            "key_genes": key_genes,
            "rationale": rationale,
            # backward-compat keys
            "trait": axis_label or "(unspecified axis)",
            "hypothesis": mechanism or "",
        })

    # Enforce count/order constraints strictly
    up_items = [x for x in normed if x["direction"] == "up_regulation"]
    down_items = [x for x in normed if x["direction"] == "down_regulation"]

    out: List[Dict[str, Any]] = []
    if regulation_type == "up_regulation":
        out = (up_items[:1] or [{"id":"UP1","direction":"up_regulation","axis_label":"(unspecified axis)","mechanism":"","gwas_traits":[],"mechanism_keywords":[],"assays":[],"pmc_queries":[],"key_genes":[],"rationale":"","trait":"(unspecified axis)","hypothesis":""}])
        out[0]["id"] = "UP1"
    elif regulation_type == "down_regulation":
        out = (down_items[:1] or [{"id":"DOWN1","direction":"down_regulation","axis_label":"(unspecified axis)","mechanism":"","gwas_traits":[],"mechanism_keywords":[],"assays":[],"pmc_queries":[],"key_genes":[],"rationale":"","trait":"(unspecified axis)","hypothesis":""}])
        out[0]["id"] = "DOWN1"
    else:
        up = up_items[:1] or [{"id":"UP1","direction":"up_regulation","axis_label":"(unspecified axis)","mechanism":"","gwas_traits":[],"mechanism_keywords":[],"assays":[],"pmc_queries":[],"key_genes":[],"rationale":"","trait":"(unspecified axis)","hypothesis":""}]
        down = down_items[:1] or [{"id":"DOWN1","direction":"down_regulation","axis_label":"(unspecified axis)","mechanism":"","gwas_traits":[],"mechanism_keywords":[],"assays":[],"pmc_queries":[],"key_genes":[],"rationale":"","trait":"(unspecified axis)","hypothesis":""}]
        up[0]["id"] = "UP1"
        down[0]["id"] = "DOWN1"
        out = [up[0], down[0]]

    return out

def gpt_propose_two_terms(rbp: str,
                          lncrna: str,
                          direction: str,
                          trait: str,
                          context: str,
                          top_ppi: List[str],
                          drs_hint: str) -> List[str]:
    """
    Build TWO short literature search queries for PMC.
    NOTE: Some model deployments do not support response_format.type="json_array".
    We therefore request a json_object with a top-level "queries" array and parse it.
    """
    sys_prompt = (
        "You are a biomedical information specialist. Build TWO short Google queries "
        "(≤ 8 words each) intended to retrieve mechanistic papers from PubMed Central (PMC).\n"
        "Avoid quotes. Prefer including gene symbols and 1 phenotype keyword.\n"
        'Return a JSON OBJECT only: {"queries": ["...","..."]}.'
    )
    ctx = {
        "rbp": rbp,
        "lncrna": lncrna or "",
        "direction": direction or "",
        "trait": trait,
        "context": context,
        "ppi_partners": top_ppi[:3],
        "drs_hint": drs_hint[:200],
    }
    user = json.dumps(ctx, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            arr = data.get("queries") or data.get("items") or []
        elif isinstance(data, list):
            arr = data
        else:
            arr = []
        arr = [re.sub(r"\s+", " ", str(x)).strip() for x in arr if str(x).strip()]
        out, seen = [], set()
        for t in arr:
            if len(t.split()) <= 12 and t.lower() not in seen:
                out.append(t)
                seen.add(t.lower())
            if len(out) >= 2:
                break
        if not out:
            out = [f"{rbp} {trait}", f"{lncrna} {rbp}"] if lncrna else [f"{rbp} {trait}", f"{rbp} {context}"]
        return out[:2]
    except Exception as e:
        log_debug(f"[gpt_propose_two_terms] {e}")
        if lncrna:
            return [f"{lncrna} {rbp} {trait}", f"{rbp} {trait} mechanism"][:2]
        return [f"{rbp} {trait}", f"{rbp} {context}"][:2]

def gpt_select_ppi_for_hypothesis(rbp: str,
                                  trait: str,
                                  context: str,
                                  ppi_rows: List[Dict[str, str]],
                                  max_return: int = 12) -> Dict[str, Any]:
    """
    Ask GPT to pick PPI partners relevant to the hypothesis.
    Returns: {"partners": [{"gene":"TP53","why":"...","pair":"RBP–TP53","db":"..."}]}
    """
    items = []
    rbpU = rbp.upper()
    for r in ppi_rows[:500]:
        ga = str(r.get("Gene Name A","")) or ""
        gb = str(r.get("Gene Name B","")) or ""
        db = str(r.get("Database Name","")) or ""
        if not ga or not gb:
            continue
        pair = f"{ga}–{gb}"
        partner = gb if ga.upper()==rbpU else ga if gb.upper()==rbpU else ""
        if not partner:
            continue
        items.append({"pair": pair, "partner": partner, "db": db})

    sys_prompt = (
        "You are selecting protein–protein interactors that matter for a specific hypothesis.\n"
        "Given a list of pairs involving the query RBP, pick partners that are mechanistically or contextually relevant "
        "to the TRAIT and CONTEXT. Prefer partners with known regulatory roles, complexes, or pathways that plausibly "
        "affect the trait. Avoid generic interactors.\n"
        f"Return strict JSON: {{\"partners\":[{{\"gene\":\"...\",\"why\":\"...\",\"pair\":\"...\",\"db\":\"...\"}}]}} "
        f"Limit to at most {max_return} partners."
    )
    usr = json.dumps({"rbp": rbp, "trait": trait, "context": context, "ppi_pairs": items}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":usr}]
        )
        data = json.loads(resp.choices[0].message.content.strip())
        arr = data.get("partners", [])
        out = []
        seen = set()
        for it in arr:
            g = str(it.get("gene") or it.get("partner") or "").strip()
            if not g:
                continue
            gU = g.upper()
            if gU == rbpU or gU in seen:
                continue
            seen.add(gU)
            out.append({
                "gene": gU,
                "why": re.sub(r"\s+"," ", str(it.get("why",""))).strip(),
                "pair": str(it.get("pair","")),
                "db": str(it.get("db","")),
            })
        return {"partners": out[:max_return]}
    except Exception as e:
        log(f"[gpt_select_ppi_for_hypothesis] {e}")
        # fallback: top by frequency
        counts = {}
        for it in items:
            partner = it["partner"].upper()
            counts[partner] = counts.get(partner, 0) + 1
        top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_return]
        return {"partners": [{"gene": g, "why": "freq-based fallback", "pair": "", "db": ""} for g,_ in top]}

def gpt_extract_conclusions(text: str, rbp: str, query: str) -> List[str]:
    cand = "\n".join(extract_figure_table_sentences(text))
    sys_prompt = (
        "You are a biomedical literature assistant.\n"
        "From SENTENCES, extract EVERY experimental conclusion relevant to the RBP gene or the QUERY.\n"
        'Return JSON: {"conclusions": ["..."]}'
    )
    usr = f"RBP:{rbp}\nQUERY:{query}\nSENTENCES:\n{cand}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":usr}]
        )
        data = json.loads(resp.choices[0].message.content.strip())
        arr = data.get("conclusions", [])
        return arr if isinstance(arr, list) else []
    except Exception as e:
        log(f"[gpt_extract_conclusions] {e}")
        return []

def gpt_extract_experiments(article_text: str,
                            conclusions: List[str],
                            gene_set: List[str],
                            query: str,
                            pmc_url: str) -> List[Dict[str,Any]]:
    """
    Extract experiments only if they directly involve symbols from GENE_SET.
    """
    if not conclusions:
        return []
    system = (
        "You are a biomedical text-mining assistant.\n"
        "Task: For EACH conclusion, find ≤3 supporting experiments and return JSON object with key `experiments`.\n"
        "Only include an experiment if the model or result explicitly involves a symbol from GENE_SET "
        "(e.g., knockdown/overexpression/binding/expression/readout).\n"
        "For each experiment include fields:\n"
        "  pmc_url, conclusion, target_gene, model, intervention, readout,\n"
        "  result_stats, locating_sentence, evidence_type.\n"
        "evidence_type ∈ {Molecular Mechanism Experiments, Cellular Function Experiments, "
        "Animal Function Experiments, Clinical Information Analysis Experiments, Bioinformatic Analysis}."
    )
    user = (
        f"GENE_SET:{json.dumps(gene_set, ensure_ascii=False)}\n"
        f"QUERY:{query}\nPMC_URL:{pmc_url}\n"
        f"CONCLUSIONS:{json.dumps(conclusions, ensure_ascii=False)}\n"
        f"FULLTEXT:{article_text[:60000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}]
        )
        obj = json.loads(resp.choices[0].message.content.strip())
        exps = obj.get("experiments", [])
        for e in exps:
            method_pieces = [e.get("model",""), e.get("intervention",""), e.get("readout","")]
            e["methods"] = "; ".join([p for p in method_pieces if p]).strip()
            e["results"] = e.get("result_stats") or e.get("locating_sentence","")
        return exps
    except Exception as e:
        log(f"[gpt_extract_experiments] {e}")
        return []

def extract_article_evidence_cached(pmc_url: str,
                                   gene_set: List[str],
                                   query: str,
                                   cache: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Cached wrapper: avoid re-fetching/re-extracting the same PMC URL.
    Cache key includes (pmc_url, sorted(gene_set), query) because extraction is gene_set dependent.
    """
    key = hashlib.sha1(
        (pmc_url + "||" + ",".join(sorted({g.upper() for g in gene_set})) + "||" + query).encode("utf-8")
    ).hexdigest()
    if key in cache:
        return cache[key]
    meta = fetch_full_text(pmc_url)
    if not meta:
        cache[key] = None
        return None
    concl = gpt_extract_conclusions(meta["text"], gene_set[0] if gene_set else "", query)
    exps  = gpt_extract_experiments(meta["text"], concl, gene_set, query, pmc_url)

    # strict gene_set filter
    if gene_set:
        pat = re.compile(r"\b(" + "|".join(re.escape(x) for x in gene_set) + r")\b", re.I)
        exps2 = []
        for e in exps:
            blob = " ".join([e.get("target_gene",""), e.get("methods",""), e.get("results",""), e.get("locating_sentence","")])
            if pat.search(blob or ""):
                exps2.append(e)
    else:
        exps2 = exps

    if not concl and not exps2:
        cache[key] = None
        return None
    out = {
        "meta": {k: meta.get(k,"") for k in ("pmid","pmc_url","date","journal","title")},
        "conclusions": concl,
        "experiments": exps2,
    }
    cache[key] = out
    return out

# =========================
# 5) GWAS parsing & registry helpers
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", s.lower()).strip()

def parse_rbp_traits_with_scores(row: pd.Series) -> List[Tuple[str,float]]:
    pairs: List[Tuple[str,float]] = []
    cand_cols = [c for c in row.index if re.search(r"(gwas|trait|phenotype|disease)", c, re.I)]
    for c in cand_cols:
        v = str(row.get(c, "")).strip()
        if not v:
            continue
        for seg in TRAIT_SPLIT_RE.split(v):
            s = re.sub(r"\s+", " ", seg).strip()
            if not s:
                continue
            m = re.match(r"^(.*?)[\s]*\(([-+]?\d*\.?\d+)\)\s*$", s)
            if m:
                name = m.group(1).strip()
                try:
                    score = float(m.group(2))
                except Exception:
                    score = float("nan")
                if name:
                    pairs.append((name, score))
            else:
                pairs.append((s, float("nan")))
    seen = {}
    for t,sc in pairs:
        key = t.lower()
        if key not in seen:
            seen[key] = sc
    return [(t, seen[t.lower()]) for t,_ in pairs if t.lower() in seen]

def best_trait_match_for_phenotype(traits_scored: List[Tuple[str,float]], phenotype: str) -> Tuple[str, float] | None:
    if not traits_scored or not phenotype:
        return None
    p_norm = normalize_text(phenotype)
    for t, sc in traits_scored:
        if normalize_text(t) == p_norm:
            return (t, sc)
    for t, sc in traits_scored:
        tn = normalize_text(t)
        if p_norm in tn or tn in p_norm:
            return (t, sc)
    p_set = set(p_norm.split())
    best = None
    best_iou = 0.0
    for t, sc in traits_scored:
        tn = normalize_text(t)
        t_set = set(tn.split())
        if not p_set or not t_set:
            continue
        inter = len(p_set & t_set)
        union = len(p_set | t_set)
        iou = inter/union if union else 0.0
        if iou > best_iou:
            best_iou = iou
            best = (t, sc)
    return best

def coerce_to_str_phenotype(item: Any) -> str:
    """Robustly convert hypothesis-like object into a label string."""
    if isinstance(item, str):
        s = item.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                return coerce_to_str_phenotype(parsed)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                    return coerce_to_str_phenotype(parsed)
                except Exception:
                    return s
        return s
    if isinstance(item, dict):
        if "trait" in item and str(item["trait"]).strip():
            return str(item["trait"]).strip()
        if "phenotype" in item and str(item["phenotype"]).strip():
            return str(item["phenotype"]).strip()
        for k in ("name","label","title"):
            if k in item and str(item[k]).strip():
                return str(item[k]).strip()
        joined = " ".join(str(v) for v in item.values() if str(v).strip())
        return joined.strip() or "(unknown phenotype)"
    if isinstance(item, (list, tuple)):
        parts = [coerce_to_str_phenotype(x) for x in item if x is not None]
        joined = " ".join(p for p in parts if p.strip())
        return joined.strip() or "(unknown phenotype)"
    try:
        return str(item).strip()
    except Exception:
        return "(unknown phenotype)"

def safe_axis_label(rbp: str, trait: str, direction: str) -> str:
    d = "UP" if direction=="up_regulation" else "DOWN"
    return f"{rbp.upper()} - {trait} ({d})"

# =========================
# 6) Registry (ID tagging)
# =========================
def build_db_registry(rbp: str,
                      rbp_gene_df: pd.DataFrame,
                      assay_gene_df: pd.DataFrame,
                      ppi_records: List[Dict[str,str]],
                      drs_human_df: pd.DataFrame,
                      drs_mouse_df: pd.DataFrame,
                      include_drs: bool) -> Dict[str, Dict[str, Any]]:
    """
    Return registry: db_id -> {source, text, payload, ...}
    source ∈ {RBP, ASSAY, PPI, DRS-H, DRS-M}
    """
    reg: Dict[str, Dict[str, Any]] = {}

    # --- RBPbase rows ---
    if not rbp_gene_df.empty:
        for i, row in rbp_gene_df.reset_index(drop=True).iterrows():
            rid = f"RBP:{i+1:04d}"
            traits_scored = parse_rbp_traits_with_scores(row)
            traits = [t for t,_ in traits_scored]
            digest = f"RBPbase GWAS: {', '.join(traits[:8])}" if traits else "RBPbase row"
            reg[rid] = {
                "id": rid,
                "source": "RBP",
                "text": digest,
                "payload": row.to_dict(),
                "traits": traits,
                "traits_scored": traits_scored,
            }

    # --- Gene_Assay rows (trim payload) ---
    if not assay_gene_df.empty:
        for i, row in assay_gene_df.reset_index(drop=True).iterrows():
            rid = f"ASSAY:{i+1:04d}"
            payload = {}
            for col in ["gene_symbol","ensg_id","pmid","pmcid","readout","explanation"]:
                if col in assay_gene_df.columns:
                    payload[col] = str(row.get(col, "")).strip()
            pmcid = payload.get("pmcid","")
            if pmcid and not pmcid.upper().startswith("PMC"):
                payload["pmcid"] = "PMC"+pmcid
            title = ""
            for c in ["pubmed_title","title","pub_title"]:
                if str(row.get(c,"")).strip():
                    title = str(row.get(c,"")).strip(); break
            concl = str(payload.get("explanation","")).strip()
            reg[rid] = {
                "id": rid,
                "source": "ASSAY",
                "text": title or (concl[:120] if concl else "Gene_Assay row"),
                "payload": payload,
                "title": title,
                "pmcid": payload.get("pmcid",""),
                "pmid": payload.get("pmid",""),
                "conclusion": concl,
            }

    # --- PPI rows ---
    if ppi_records:
        for i, r in enumerate(ppi_records, start=1):
            a = (r.get("Gene Name A") or "").upper()
            b = (r.get("Gene Name B") or "").upper()
            partner = b if a==rbp.upper() else (a if b==rbp.upper() else (b or a))
            rid = f"PPI:{i:04d}"
            reg[rid] = {
                "id": rid,
                "source": "PPI",
                "text": f"{a} – {b}",
                "payload": r,
                "partner": partner,
            }

    # --- DRS: main gene + PPI genes (only if include_drs=True) ---
    if include_drs and (not drs_human_df.empty or not drs_mouse_df.empty):
        h, m = gather_drs_for_gene(rbp, drs_human_df, drs_mouse_df)
        if h:
            reg[f"DRS:H:{rbp.upper()}"] = {
                "id": f"DRS:H:{rbp.upper()}",
                "source": "DRS-H",
                "text": f"Human DRS significant: {len([k for k in h.keys() if k.endswith('_p')])} p-keys",
                "payload": h,
                "gene": rbp.upper(),
            }
        if m:
            reg[f"DRS:M:{rbp.upper()}"] = {
                "id": f"DRS:M:{rbp.upper()}",
                "source": "DRS-M",
                "text": f"Mouse DRS significant: {len([k for k in m.keys() if k.endswith('_p')])} p-keys",
                "payload": m,
                "gene": rbp.upper(),
            }
        # PPI genes
        ppi_genes = sorted({
            (r.get("Gene Name A") or "").upper() for r in ppi_records
        } | {
            (r.get("Gene Name B") or "").upper() for r in ppi_records
        })
        ppi_genes = [g for g in ppi_genes if g and g != rbp.upper()]
        for g in ppi_genes[:80]:
            hh, mm = gather_drs_for_gene(g, drs_human_df, drs_mouse_df)
            if hh:
                reg[f"DRS:H:{g}"] = {"id":f"DRS:H:{g}","source":"DRS-H","text":f"Human DRS significant for {g}","payload":hh,"gene":g}
            if mm:
                reg[f"DRS:M:{g}"] = {"id":f"DRS:M:{g}","source":"DRS-M","text":f"Mouse DRS significant for {g}","payload":mm,"gene":g}
    return reg

def build_gtex_registry(expr_map: Dict[str, float],
                        tissue_col: str,
                        genes: List[str]) -> Dict[str, Dict[str, Any]]:
    reg: Dict[str, Dict[str, Any]] = {}
    t = tissue_col or "Tissue"
    for g in sorted({x.upper() for x in genes if x}):
        val = expr_map.get(g.upper())
        rid = f"GTEX:{t}:{g.upper()}"
        reg[rid] = {
            "id": rid,
            "source": "GTEX",
            "text": f"{g.upper()} {t} expr={val if val is not None else 'NA'}",
            "payload": {"gene": g.upper(), "tissue": t, "expression": val},
            "gene": g.upper(),
            "tissue": t,
        }
    return reg

def registry_digest_for_hypothesis(registry: Dict[str, Dict[str,Any]],
                                   target_traits: Any,
                                   include_ids: Optional[Set[str]] = None,
                                   max_hits_per_row: int = 6) -> List[Dict[str,str]]:
    """
    Build a compact digest of registry rows for downstream GPT selection/rating.

    NEW: `target_traits` can be a *list* of GWAS traits (trait cluster) rather than a single trait.
    For RBPbase rows we annotate any matched traits as [HITS: ...] to help the model quickly
    see which GWAS phenotypes are relevant to the current mechanism hypothesis.

    If include_ids is provided, only those ids are included (in that order).
    """
    # Normalize target trait list
    traits: List[str] = []
    if isinstance(target_traits, str):
        traits = [target_traits]
    elif isinstance(target_traits, (list, tuple, set)):
        traits = [str(x) for x in target_traits]
    else:
        traits = [str(target_traits)] if target_traits is not None else []
    # de-dup, keep order
    seen_tt = set()
    traits2: List[str] = []
    for t in traits:
        t = re.sub(r"\s+", " ", str(t or "")).strip()
        if not t:
            continue
        k = normalize_text(t)
        if k in seen_tt:
            continue
        seen_tt.add(k)
        traits2.append(t)
    traits = traits2

    out: List[Dict[str,str]] = []
    items = []
    if include_ids is None:
        items = list(registry.items())
    else:
        for rid in include_ids:
            if rid in registry:
                items.append((rid, registry[rid]))

    def _score_val(sc: Any) -> float:
        try:
            f = float(sc)
            if f != f:
                return float("-inf")
            return f
        except Exception:
            return float("-inf")

    for _, v in items:
        txt = v.get("text","")
        if v.get("source") == "RBP" and traits:
            tsc = v.get("traits_scored") or []
            hits = []
            for tt in traits:
                best = best_trait_match_for_phenotype(tsc, tt)
                if best:
                    tr, sc = best
                    hits.append((tr, sc))
            # de-dup hits by trait name
            uniq = {}
            for tr, sc in hits:
                k = normalize_text(tr)
                if k not in uniq or _score_val(sc) > _score_val(uniq[k]):
                    uniq[k] = sc
            hits2 = sorted([(tr, uniq[normalize_text(tr)]) for tr in {h[0] for h in hits}],
                           key=lambda x: _score_val(x[1]), reverse=True)
            hits2 = hits2[:max_hits_per_row]
            if hits2:
                pieces = []
                for tr, sc in hits2:
                    if sc is None or (isinstance(sc, float) and sc != sc):
                        pieces.append(f"{tr}(score=NA)")
                    else:
                        try:
                            pieces.append(f"{tr}(score={float(sc):.3g})")
                        except Exception:
                            pieces.append(f"{tr}(score={sc})")
                txt = f"{txt} [HITS: " + "; ".join(pieces) + "]"
        out.append({"id": v["id"], "source": v["source"], "text": txt})
    return out

# =========================
# 7) Rating + narrative (LLM)
# =========================
EXPLICIT_RATING_RULES = """Rating rubric (apply exactly; do not improvise):
- HIGH: Must satisfy any of the following:
  (H1) ≥1 direct functional experiment on the query gene (Gene_Assay or extracted literature) showing a change in the hypothesis trait/phenotype in an appropriate model, AND at least one of:
       • consistent DRS regulation for the gene or a relevant PPI partner (if available); or
       • independent replication in a different model/dataset.
  (H2) ≥2 independent experimental modalities among {molecular mechanism, cellular function, animal function, clinical/observational} that converge on the same direction/effect; PPI evidence is used mechanistically (not only as a list).
- MEDIUM: Evidence suggests the hypothesis but lacks replication or breadth:
  (M1) exactly one functional experiment with plausible effect but limited scope; OR
  (M2) consistent DRS signal and/or strong GWAS association that aligns with the trait but without direct functional manipulation; OR
  (M3) mechanistic link via PPI plus one supportive line (DRS or expression) but no direct assay.
- LOW: Signals are weak, indirect, or inconsistent:
  (L1) only association-level evidence (GWAS/RBPbase) without supporting functional data; OR
  (L2) PPI-only hints or pathway speculation; OR
  (L3) conflicting or null experimental results, or phenotype not specific to the hypothesis.

Tie-breakers:
- Prefer appropriate tissue/context, concordant directionality, and higher-quality studies.
Return the chosen label and a short rule-by-rule justification referencing (H1/H2/M1/M2/M3/L1/L2/L3).
"""

def gpt_select_db_evidence_and_rate(rbp: str,
                                   axis_label: str,
                                   context: str,
                                   db_digest: List[Dict[str, str]],
                                   evidence_counts: Dict[str, int],
                                   has_assay_hits: bool,
                                   selected_ppi_genes: Optional[List[str]] = None,
                                   trait_cluster: Optional[List[str]] = None,
                                   mechanism: str = "") -> Dict[str, Any]:
    """
    Select ONLY the database rows truly USED and assign a rating (High/Medium/Low)
    for a *mechanism-first* hypothesis.

    axis_label: short mechanism/phenotype-cluster label.
    trait_cluster: list of GWAS traits (downstream phenotypes) that the mechanism aims to explain.
    mechanism: mechanistic hypothesis text (optional, helps anchoring).

    Output JSON keys:
      - hypothesis_axis (string)
      - verdict ("High"|"Medium"|"Low")
      - used_db_refs (list of ids)
      - rule_justification (string)
      - notes (string)
    """
    ppi_clause = ""
    if selected_ppi_genes:
        ppi_list = "; ".join(sorted({str(x).strip() for x in selected_ppi_genes if str(x).strip()}))
        ppi_clause = (
            "\nUse PPI rows ONLY when the partner is in this preselected set and it helps mechanism: "
            + ppi_list + ".\n"
        )

    sys_prompt = (
        "You are consolidating evidence for a single biological *mechanism-first* hypothesis.\n"
        "Input: database row digests (with ids) + experiment-type counts from literature extraction.\n"
        "Goal: select ONLY those database rows that are truly USED in final reasoning.\n\n"
        "Important framing:\n"
        "  - axis_label is a short description of the mechanism/phenotype-cluster.\n"
        "  - trait_cluster is a list of downstream GWAS phenotypes that might share a common mechanism.\n"
        "  - Prefer mechanistic/functional evidence (Gene_Assay, extracted experiments) when available.\n"
        "  - Use GWAS (RBPbase) to motivate relevance, not as proof of mechanism.\n"
        + ppi_clause +
        "\nApply the rubric below to assign High/Medium/Low.\n\n"
        + EXPLICIT_RATING_RULES +
        "\nReturn STRICT JSON object with keys: hypothesis_axis, verdict, used_db_refs, rule_justification, notes."
    )

    payload = {
        "rbp": rbp,
        "axis_label": axis_label,
        "mechanism": mechanism[:1200] if mechanism else "",
        "trait_cluster": list(trait_cluster or [])[:50],
        "context": context,
        "has_assay_hits": bool(has_assay_hits),
        "experiment_type_counts": evidence_counts,
        "db_digest": db_digest[:140],
        "selected_ppi_genes": list(selected_ppi_genes or []),
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
            ]
        )
        obj = json.loads(clean_json_response(resp.choices[0].message.content))
        if not isinstance(obj, dict):
            raise ValueError("not dict")
        obj.setdefault("hypothesis_axis", f"{rbp} ↔ {axis_label}")
        obj.setdefault("verdict", "Low")
        obj.setdefault("used_db_refs", [])
        obj.setdefault("rule_justification", "")
        obj.setdefault("notes", "")
        obj["verdict"] = canonicalize_verdict(obj.get("verdict","Low"))
        u = obj.get("used_db_refs", [])
        if not isinstance(u, list):
            u = []
        obj["used_db_refs"] = [str(x).strip() for x in u if str(x).strip()]
        return obj
    except Exception as e:
        log(f"[gpt_select_db_evidence_and_rate] fallback due to {type(e).__name__}: {e}")
        c = evidence_counts or {}
        has_mol = c.get("Molecular Mechanism Experiments",0)>0
        has_cell = c.get("Cellular Function Experiments",0)>0
        has_anm = c.get("Animal Function Experiments",0)>0
        verdict = "High" if (has_cell or has_anm) and has_mol else ("Medium" if (has_cell or has_anm or has_mol) else "Low")
        return {
            "hypothesis_axis": f"{rbp} ↔ {axis_label}",
            "verdict": canonicalize_verdict(verdict),
            "used_db_refs": [],
            "rule_justification": "fallback",
            "notes": "fallback",
        }

def gpt_compose_hypothesis_narrative(rbp: str,
                                     trait: str,
                                     direction: str,
                                     context: str,
                                     hypothesis_statement: str,
                                     used_evidence: Dict[str, Any],
                                     ev_counts: Dict[str, int],
                                     article_details: List[Dict[str, Any]] | None,
                                     verdict: str,
                                     rule_justification: str) -> Dict[str, Any]:
    """
    Compose narrative paragraph + steps integrating DB + literature evidence.
    Returns: {"paragraph": str, "why": str, "steps": [str,...]}
    """
    sys_prompt = (
        "You are summarizing scientific evidence for an RBP regulation hypothesis.\n"
        "Use ALL evidence below (DB + literature extraction).\n"
        "Requirements:\n"
        "  • Write a concise paragraph (2–4 sentences) stating the hypothesis and integrating evidence.\n"
        "  • Provide a short 'why rating' explanation referencing the rubric.\n"
        "  • Provide a numbered list (3–7 bullets). Each bullet MUST end with a source tag like "
        "[Source: Gene_Assay], [Source: GTEx], [Source: DRS-Human], [Source: PPI], [Source: PMID x], etc.\n"
        "Return strict JSON: {\"paragraph\":\"...\",\"why\":\"...\",\"steps\":[\"...\"]}."
    )
    payload = {
        "rbp": rbp,
        "trait": trait,
        "direction": direction,
        "context": context,
        "hypothesis_statement": hypothesis_statement,
        "rating": canonicalize_verdict(verdict),
        "rating_rules_applied": rule_justification,
        "evidence_type_counts": ev_counts,
        "used_evidence": used_evidence,
        "literature_details": article_details or [],
    }
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
        )
        obj = json.loads(resp.choices[0].message.content.strip())
        para = str(obj.get("paragraph","")).strip()
        why  = str(obj.get("why","")).strip()
        steps = obj.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        return {"paragraph": para, "why": why, "steps": steps[:7]}
    except Exception as e:
        log(f"[gpt_compose_hypothesis_narrative] {e}")
        para = f"{rbp} ({direction}) ↔ {trait}. Evidence types: " + ", ".join(f"{k}:{v}" for k,v in (ev_counts or {}).items())
        steps = [f"Rating: {canonicalize_verdict(verdict)} — {rule_justification} [Source: rules]"]
        return {"paragraph": para, "why": "fallback narrative", "steps": steps[:5]}

# =========================
# 8) Gene_Assay literature harvesting
# =========================
def extract_assay_pmc_urls(assay_gene_df: pd.DataFrame, max_urls: int = 25) -> List[str]:
    """
    From Gene_Assay rows, collect PMC URLs. Uses PMCID if present, else PMID→PMCID conversion.
    """
    urls: List[str] = []
    if assay_gene_df is None or assay_gene_df.empty:
        return urls

    # Detect columns
    cols = [c.lower() for c in assay_gene_df.columns]
    pmcid_col = None
    pmid_col = None
    for c in assay_gene_df.columns:
        if re.search(r"pmcid|pmc_id", c, re.I):
            pmcid_col = c; break
    for c in assay_gene_df.columns:
        if re.search(r"\bpmid\b", c, re.I):
            pmid_col = c; break

    for _, row in assay_gene_df.iterrows():
        pmcid = str(row.get(pmcid_col, "")).strip() if pmcid_col else ""
        pmid = str(row.get(pmid_col, "")).strip() if pmid_col else ""
        if pmcid:
            if not pmcid.upper().startswith("PMC"):
                pmcid = "PMC" + pmcid
            url = pmcid_to_pmc_url(pmcid)
            if url and url not in urls:
                urls.append(url)
        elif pmid:
            pmcid2 = pmid_to_pmcid(pmid)
            if pmcid2:
                url = pmcid_to_pmc_url(pmcid2)
                if url and url not in urls:
                    urls.append(url)
        if len(urls) >= max_urls:
            break
    return urls

# =========================
# 9) Search-term building
# =========================
def drs_brief_from_items(drs_items: List[Dict[str, Any]]) -> str:
    if not drs_items:
        return ""
    # count total significant bases across genes
    total = 0
    for it in drs_items:
        payload = it.get("payload") or {}
        total += len([k for k in payload.keys() if str(k).endswith("_p")])
    return f"{len(drs_items)} genes; {total} sig p-keys"

def build_search_terms_for_hypothesis(rbp: str,
                                      lncrna: str | None,
                                      direction: str,
                                      axis_label: str,
                                      gwas_trait_terms: Optional[List[str]] = None,
                                      mechanism_keywords: Optional[List[str]] = None,
                                      pmc_queries: Optional[List[str]] = None,
                                      key_genes: Optional[List[str]] = None,
                                      context: str = "",
                                      drs_items: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Build a small set of PMC search queries for a *mechanism-first* hypothesis.

    Priority order:
      1) model-proposed pmc_queries (if provided by hypothesis generator)
      2) lncRNA–RBP regulation phrases
      3) mechanism keywords
      4) representative GWAS traits (from the trait cluster)
      5) direction-specific perturbation terms (overexpression/knockdown)

    Returns ≤ 6 de-duplicated query strings.
    """
    rbp = (rbp or "").strip()
    lncrna = (lncrna or "").strip() if lncrna else ""
    axis_label = re.sub(r"\s+", " ", str(axis_label or "")).strip()

    gwas_trait_terms = [re.sub(r"\s+", " ", str(t)).strip() for t in (gwas_trait_terms or []) if str(t).strip()]
    mechanism_keywords = [re.sub(r"\s+", " ", str(k)).strip() for k in (mechanism_keywords or []) if str(k).strip()]
    pmc_queries = [re.sub(r"\s+", " ", str(q)).strip() for q in (pmc_queries or []) if str(q).strip()]
    key_genes = [re.sub(r"\s+", " ", str(g)).strip() for g in (key_genes or []) if str(g).strip()]

    terms: List[str] = []

    # 1) LLM-proposed queries (best starting point)
    for q in pmc_queries:
        terms.append(q)

    # 2) core gene regulation pair
    if lncrna:
        terms.append(f"{lncrna} {rbp} regulation")
        terms.append(f"{lncrna} {rbp} lncRNA")
    terms.append(f"{rbp} {axis_label}" if axis_label else f"{rbp} mechanism")

    # 3) mechanism keywords
    for kw in mechanism_keywords[:3]:
        terms.append(f"{rbp} {kw}")

    # 4) representative GWAS traits
    for t in gwas_trait_terms[:2]:
        terms.append(f"{rbp} {t}")

    # 5) direction perturbation hints
    if direction == "up_regulation":
        terms.append(f"{rbp} overexpression")
    else:
        terms.append(f"{rbp} knockdown")

    # 6) add one key interactor if available (boost mechanistic papers)
    extras = [g for g in key_genes if g.upper() not in {rbp.upper(), (lncrna or '').upper()}]
    if extras:
        terms.append(f"{rbp} {extras[0]} interaction")

    # If nothing worked (shouldn't happen), call the old helper as a last resort
    if not any(t.strip() for t in terms):
        try:
            drs_hint = drs_brief_from_items(drs_items or [])
            ppi3 = extras[:3]
            extra2 = gpt_propose_two_terms(
                rbp=rbp,
                lncrna=lncrna,
                direction=direction,
                trait=(gwas_trait_terms[0] if gwas_trait_terms else axis_label or "mechanism"),
                context=context,
                top_ppi=ppi3,
                drs_hint=drs_hint
            )
            terms.extend(extra2)
        except Exception:
            terms.extend([f"{rbp} RNA binding", f"{rbp} splicing"])

    # Deduplicate preserving order + keep reasonably short
    out: List[str] = []
    seen = set()
    for t in terms:
        tt = re.sub(r"\s+", " ", t).strip()
        if not tt:
            continue
        # cap words (keep queries terse)
        if len(tt.split()) > 14:
            tt = " ".join(tt.split()[:14])
        key = tt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tt)
        if len(out) >= 6:
            break
    return out

# =========================
# 10) HTML report (Hypothesis cards)
# =========================
def _linkify_pmcid(val: str) -> str:
    if not val:
        return "-"
    pmc = val.strip()
    if not pmc.upper().startswith("PMC"):
        pmc = "PMC"+pmc
    return f"<a target='_blank' href='https://pmc.ncbi.nlm.nih.gov/articles/{_esc(pmc)}/'>{_esc(pmc)}</a>"

def _linkify_pmid(val: str) -> str:
    if not val:
        return "-"
    return f"<a target='_blank' href='https://pubmed.ncbi.nlm.nih.gov/{_esc(val)}/'>{_esc(val)}</a>"

def make_chart_js() -> str:
    # Reuse existing chart helpers (Chart.js + datalabels)
    return """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<script>
if (typeof ChartDataLabels !== 'undefined' && typeof Chart !== 'undefined') {
  Chart.register(ChartDataLabels);
}
function pieOnce(canvasId, dataObj) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: Object.keys(dataObj || {}),
      datasets: [{ data: Object.values(dataObj || {}) }]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
  });
}
function barDRSGroupedOnce(canvasId, genes, datasets, yTitle) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const ds = (datasets || []).map(function(d) {
    return { label: d.label, data: d.data, pvals: d.pvals || [], categoryPercentage: 0.85, barPercentage: 0.88 };
  });
  new Chart(ctx, {
    type: 'bar',
    data: { labels: genes, datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { top: 8 } },
      scales: {
        x: { stacked: false, ticks: { autoSkip: false, maxRotation: 45, minRotation: 0 } },
        y: { stacked: false, beginAtZero: true, title: { display: true, text: yTitle || 'log2FC' } }
      },
      plugins: {
        legend: { position: 'top' },
        tooltip: { callbacks: { label: function(context) {
          const pv = context.dataset.pvals ? context.dataset.pvals[context.dataIndex] : undefined;
          let ptxt = (pv!=null && !isNaN(pv)) ? (pv < 1e-4 ? pv.toExponential(2) : pv.toPrecision(2)) : 'NA';
          return context.dataset.label + ': ' + context.formattedValue + ' | p=' + ptxt;
        }}},
        datalabels: {
          anchor: 'end', align: 'top', offset: 2,
          formatter: function(value, ctx) {
            const pv = (ctx.dataset && ctx.dataset.pvals) ? ctx.dataset.pvals[ctx.dataIndex] : undefined;
            if (pv==null || isNaN(pv)) return '';
            return 'p=' + (pv < 1e-4 ? pv.toExponential(2) : pv.toPrecision(2));
          },
          font: { size: 9 }
        }
      }
    }
  });
}
function toggle(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = (el.style.display==='none') ? 'block' : 'none';
}
</script>
"""

def build_html_report(out_html: str,
                      rbp: str,
                      lncrna: str,
                      tissue_query: str,
                      gtex_tissue_col: str | None,
                      regulation_type: str,
                      append_info: str,
                      hypotheses: List[Dict[str, Any]],
                      registry: Dict[str, Dict[str,Any]],
                      per_pair: Dict[str, Dict[str,Any]]):
    css = """
<style>
:root { --gap:2%; --left:58%; --right:40%; }
* { box-sizing: border-box; }
body { font-family: Inter, Arial, Helvetica, sans-serif; background:#f5f7fb; margin:0; }
header, footer { background:linear-gradient(90deg,#00274d,#5a9bd6); color:#fff; padding:1.2% 0; text-align:center; }
main { width:92%; margin:2% auto; background:#fff; padding:1.6%; border:0.08rem solid #e5e7eb; border-radius:0.8rem; }
.hdr { font-size:0.95rem; color:#334155; margin:0 0 1%; display:flex; flex-wrap:wrap; gap:1%; }
.hdr b { color:#0f172a; }
.card { border:0.08rem solid #e5e7eb; border-radius:0.8rem; padding:1.2%; margin:1.6% 0; background:#ffffff; }
.card h3 { margin:0 0 0.8%; color:#0f172a; }
.axis { font-weight:700; font-size:1.05rem; }
.badge { display:inline-block; margin-left:0.6rem; padding:0.15rem 0.6rem; border-radius:9999rem; font-size:0.8rem; }
.badge-high { background:#ef4444; color:#fff; }
.badge-medium { background:#f59e0b; color:#111827; }
.badge-low { background:#3b82f6; color:#fff; }
.grid { display:grid; grid-template-columns: var(--left) var(--right); gap: var(--gap); align-items:start; }
.tbl { border-collapse: collapse; width:100%; font-size:0.9rem; table-layout: fixed; }
.tbl th, .tbl td { border:0.08rem solid #cbd5e1; padding:0.5rem 0.6rem; vertical-align:top; word-break: break-word; }
.tbl th { background:#eef4ff; }
.section-title { margin:1% 0 0.6%; color:#1f2937; font-weight:600; }
.kv { color:#475569; font-size:0.85rem; }
.details, .scrollbox { border-radius:0.6rem; }
.details { border:0.08rem dashed #cbd5e1; background:#f8fafc; padding:1%; }
.scrollbox { max-height:40vh; overflow:auto; }
.right-pane { display:flex; flex-direction:column; gap:1.2rem; }
.summary-panel { background:linear-gradient(180deg,#f8fafc,#eef2ff); border:0.08rem solid #e2e8f0; padding:1.2rem; border-radius:0.8rem; }
.summary { font-size:0.95rem; color:#0f172a; line-height:1.5; margin:0; }
ol.steps { margin:0.8rem 0 0 1.2rem; }
.muted { color:#64748b; font-size:0.85rem; }
.chart-wrap { width:100%; height:38vh; overflow: hidden; position: relative; }
.drs-card { background:#f8fafc; border:0.08rem solid #e2e8f0; border-radius:0.6rem; padding:1%; position:relative; overflow:hidden; }
.drs-title { font-size:0.85rem; color:#3730a3; font-weight:600; margin-bottom:0.4rem; }
a { color:#0ea5e9; text-decoration:none; }
a:hover { text-decoration:underline; }
.tag { display:inline-block; padding:0.2rem 0.5rem; background:#eef2ff; color:#3730a3; border-radius:0.4rem; font-size:0.8rem; }
.section-spacer { margin: 1.4rem 0; clear: both; }
</style>
"""
    cards: List[str] = []
    js_blocks = [make_chart_js()]

    def tbl(items: List[Dict[str,Any]], headers: List[str], row_builder, wrap_scroll: bool=True):
        if not items:
            return "<div class='muted'>(None)</div>"
        rows = []
        for it in items:
            rows.append("<tr>"+ "".join(f"<td>{x}</td>" for x in row_builder(it)) +"</tr>")
        table_html = "<table class='tbl'><thead><tr>" + "".join(f"<th>{_esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        return f"<div class='scrollbox'>{table_html}</div>" if wrap_scroll else table_html

    for idx, h in enumerate(hypotheses, start=1):
        hid = str(h.get("id", f"H{idx}"))
        trait = str(h.get("trait","")).strip()
        direction = str(h.get("direction","")).strip()
        hyp_stmt = str(h.get("hypothesis","")).strip()

        payload = per_pair.get(hid, {}) or {}
        sel = payload.get("select_and_rate") or {}
        verdict = canonicalize_verdict(sel.get("verdict","Low"))

        axis_text = safe_axis_label(rbp, trait, direction)
        used_ids = set(sel.get("used_db_refs", []))
        ev_counts = payload.get("ev_counts", {})
        articles = payload.get("articles", [])
        narrative = payload.get("narrative", {"paragraph":"", "why":"", "steps":[]})

        # Group USED items only
        groups = {"ASSAY":[], "RBP":[], "DRS":[], "PPI":[], "GTEX":[]}
        for uid in used_ids:
            item = registry.get(uid)
            if not item:
                continue
            src = item.get("source")
            if src=="ASSAY": groups["ASSAY"].append(item)
            elif src=="RBP": groups["RBP"].append(item)
            elif src in ("DRS-H","DRS-M"): groups["DRS"].append(item)
            elif src=="PPI": groups["PPI"].append(item)
            elif src=="GTEX": groups["GTEX"].append(item)

        # ---- DRS grouped chart data ----
        drs_items = payload.get("drs_selected") or groups["DRS"]
        genes = []
        per_gene = {}
        cond_stats = {}
        for item in drs_items:
            g = str(item.get("gene","")).upper()
            if not g:
                continue
            if g not in genes:
                genes.append(g)
            p = item.get("payload", {}) or {}
            lower_map = {str(k).lower(): k for k in p.keys()}
            def _get(name): return p.get(lower_map.get(name.lower(), name))
            local = {}
            for k in p.keys():
                if str(k).endswith("_p"):
                    base = str(k)[:-2]
                    pv = _numeric(_get(base + "_p"))
                    lfc = _numeric(_get(base + "_log2fc"))
                    l10 = _numeric(_get(base + "_log10tpm"))
                    tpm = _numeric(_get(base + "_tpm"))
                    y = None
                    if lfc is not None:
                        y = float(lfc)
                    elif l10 is not None:
                        y = float(l10)
                    elif tpm is not None:
                        try:
                            y = math.log10(float(tpm)+1.0)
                        except Exception:
                            y = None
                    if y is None:
                        continue
                    local[base] = {"y": y, "p": pv if pv is not None else float("nan")}
                    if pv is not None and pv == pv:
                        s = -math.log10(max(pv, 1e-300))
                    else:
                        s = 0.0
                    cond_stats.setdefault(base, {"n":0, "score":0.0})
                    cond_stats[base]["n"] += 1
                    cond_stats[base]["score"] += s
            if local:
                per_gene[g] = local

        conditions_sorted = sorted(cond_stats.keys(), key=lambda c: (-cond_stats[c]["n"], -cond_stats[c]["score"], c))[:8]
        datasets = []
        for cond in conditions_sorted:
            data_row, p_row = [], []
            for g in genes:
                cell = (per_gene.get(g) or {}).get(cond)
                if cell:
                    data_row.append(cell["y"])
                    p_row.append(cell["p"] if cell["p"]==cell["p"] else None)
                else:
                    data_row.append(None)
                    p_row.append(None)
            datasets.append({"label": cond, "data": data_row, "pvals": p_row})

        drs_canvas_id = f"drs_all_{idx}"
        drs_js = f"barDRSGroupedOnce('{drs_canvas_id}', {json.dumps(genes)}, {json.dumps(datasets)}, {json.dumps('log2FC')});"

        pie_id = f"pie_ev_{idx}"
        pie_js = f"pieOnce('{pie_id}', {json.dumps(ev_counts)});"

        # Article details
        detail_id = f"details_{idx}"
        art_rows = []
        for art in articles:
            exps = [e for e in art.get("experiments",[]) if e]
            if not exps:
                continue
            meta = art.get("meta",{})
            url  = meta.get("pmc_url","")
            ttl  = meta.get("title","")
            pmid = meta.get("pmid","")
            jn   = meta.get("journal",""); dt = meta.get("date","")
            art_rows.append("<tr><td>"
                            f"<a href='{_esc(url)}' target='_blank'>{_esc(ttl)}</a>"
                            f"<div class='muted'>{_esc(jn)} · {_esc(dt)} · PMID {_esc(pmid)}</div>"
                            "</td><td>")
            sub = []
            for e in exps:
                et = e.get("evidence_type","")
                sub.append(
                    "<div style='margin:0.6rem 0'>"
                    f"<div><b>{_esc(et)}</b></div>"
                    f"<div class='kv'>methods: {_esc(e.get('methods',''))}</div>"
                    f"<div class='kv'>results: {_esc(e.get('results',''))}</div>"
                    f"<div class='kv'>locating sentence: {_esc(e.get('locating_sentence',''))}</div>"
                    "</div>"
                )
            art_rows.append("".join(sub))
            art_rows.append("</td></tr>")
        details_html = (
            f"<div class='details'><div>"
            f"<a href='javascript:void(0)' onclick=\"toggle('{detail_id}')\">Show/Hide Details</a></div>"
            f"<div id='{detail_id}' class='scrollbox' style='display:none'>"
            f"<table class='tbl'><thead><tr><th>Article</th><th>Experiments</th></tr></thead>"
            f"<tbody>{''.join(art_rows) or '<tr><td colspan=2>(None)</td></tr>'}</tbody></table>"
            f"</div></div>"
        )

        # Hypothesis statement block (always shown)
        # Hypothesis block (mechanism-first)
        axis_label = str(payload.get("axis_label") or payload.get("trait") or "").strip()
        top_gwas = []
        for gt in (payload.get("gwas_traits") or []):
            if isinstance(gt, dict) and str(gt.get("trait","")).strip():
                top_gwas.append(str(gt["trait"]).strip())
            elif isinstance(gt, str) and gt.strip():
                top_gwas.append(gt.strip())
            if len(top_gwas) >= 8:
                break
        top_gwas_str = ", ".join(top_gwas)
        hyp_bits = []
        if axis_label:
            hyp_bits.append(f"<div class='kv'><span class='tag'>Axis</span> {_esc(axis_label)}</div>")
        if hyp_stmt:
            hyp_bits.append(f"<div class='kv'><span class='tag'>Mechanism hypothesis</span> {_esc(hyp_stmt)}</div>")
        if top_gwas_str:
            hyp_bits.append(f"<div class='kv'><span class='tag'>GWAS traits (examples)</span> {_esc(top_gwas_str)}</div>")
        hyp_block = "".join(hyp_bits)

        # ASSAY block
        ASSAY_DISPLAY_COLUMNS = ["gene_symbol", "ensg_id", "pmid", "pmcid", "readout", "explanation"]
        def assay_row_builder(it: dict) -> list:
            payload2 = it.get("payload", it)
            cells: list = []
            for col in ASSAY_DISPLAY_COLUMNS:
                val = str(payload2.get(col, "")).strip()
                if not val:
                    cells.append("-")
                    continue
                if col == "pmcid":
                    cells.append(_linkify_pmcid(val))
                elif col == "pmid":
                    cells.append(_linkify_pmid(val))
                else:
                    cells.append(_esc(val))
            return cells
        assay_block = tbl(groups["ASSAY"], ASSAY_DISPLAY_COLUMNS, assay_row_builder, wrap_scroll=True)

        # RBP block (only used trait)
        def rbp_row_builder(it: dict) -> list:
            tsc = it.get("traits_scored") or []
            hit = best_trait_match_for_phenotype(tsc, trait)
            if hit:
                tr, sc = hit
                score = ("NA" if sc!=sc else f"{sc:.2f}")
                return [_esc(tr), _esc(score)]
            return ["-", "-"]
        # GWAS trait cluster block (mechanism-first)
        gwas_list = payload.get("gwas_traits") or []
        if not isinstance(gwas_list, list):
            gwas_list = []
        def gwas_row_builder(it: dict) -> list:
            if isinstance(it, dict):
                tr = str(it.get("trait","")).strip()
                sc = it.get("score")
                why = str(it.get("why","")).strip()
            else:
                tr = str(it).strip()
                sc = None
                why = ""
            if sc is None or (isinstance(sc, float) and sc != sc):
                score = "NA"
            else:
                try:
                    score = f"{float(sc):.3g}"
                except Exception:
                    score = str(sc)
            return [_esc(tr), _esc(score), _esc(why)]
        rbp_block = tbl(gwas_list[:30], ["Trait","Score","Why"], gwas_row_builder, wrap_scroll=True)

        # GTEx expression block (use per_pair expression table if present; else used GTEX items)
        expr_dict = payload.get("gtex_expr", {}) or {}
        expr_rows = [{"gene": g, "expr": expr_dict.get(g)} for g in sorted(expr_dict.keys())] if expr_dict else []
        def expr_row_builder(it: dict) -> list:
            return [_esc(it.get("gene","")), _esc(tissue_query), _esc(it.get("expr","NA"))]
        gtex_block = tbl(expr_rows, ["Gene","Tissue","GTEx expression"], expr_row_builder, wrap_scroll=True)

        # PPI block
        PPI_HEADERS = ["Selected by GPT?", "Gene Name A","Gene Name B","Uniprot ID A","Uniprot ID B","Species A","Species B","Database Name"]
        selected_genes = { (p.get("gene") or "").upper() for p in (payload.get("ppi_selected") or []) if (p.get("gene") or "").strip() }
        def ppi_row_builder(it: dict) -> list:
            payload2 = it.get("payload", it)
            ga = str(payload2.get("Gene Name A","")).upper()
            gb = str(payload2.get("Gene Name B","")).upper()
            sel_mark = "✓" if (ga in selected_genes or gb in selected_genes) else "–"
            cells = [sel_mark]
            for col in ["Gene Name A","Gene Name B","Uniprot ID A","Uniprot ID B","Species A","Species B","Database Name"]:
                v = str(payload2.get(col,"")).strip()
                cells.append(_esc(v) if v else "-")
            return cells
        ppi_items = payload.get("ppi_selected_records") or groups["PPI"]
        ppi_block = tbl(ppi_items, PPI_HEADERS, ppi_row_builder, wrap_scroll=True)

        # DRS block
        if datasets and any(any(v is not None for v in ds["data"]) for ds in datasets):
            drs_block = (
                f"<div class='drs-card'>"
                f"<div class='drs-title'>DRS – grouped by gene (bars = conditions; p≤0.05 filtered)</div>"
                f"<div class='chart-wrap'><canvas id='{drs_canvas_id}'></canvas></div>"
                f"</div>"
            )
        else:
            drs_block = "<div class='muted'>(None)</div>"

        badge_cls = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(verdict,"badge-low")
        card: List[str] = []
        card.append("<div class='card'>")
        card.append(f"<h3 class='axis'>{_esc(axis_text)} <span class='badge {badge_cls}'>{_esc(verdict)}</span></h3>")
        card.append(hyp_block)
        card.append("<div class='grid'>")
        # Left: database evidence
        card.append("<div>")
        card.append("<div class='section-title'>Database evidence used</div>")
        card.append("<div class='kv'>Gene_Assay (RBP)</div>"+assay_block)
        card.append("<div class='kv' style='margin-top:1rem'>GWAS (RBPbase)</div>"+rbp_block)
        card.append("<div class='kv' style='margin-top:1rem'>GTEx expression</div>"+gtex_block)
        card.append("<div class='kv' style='margin-top:1rem'>DRS (liver only)</div>"+f"<div class='section-spacer'>{drs_block}</div>")
        card.append("<div class='kv' style='margin-top:1rem'>PPI</div>"+ppi_block)
        card.append("</div>")
        # Right: summary + pie chart
        card.append("<div class='right-pane'>")
        card.append("<div class='summary-panel'>")
        card.append("<div class='section-title'>How the evidence fits together</div>")
        if narrative.get("paragraph"):
            card.append("<p class='summary'>{}</p>".format(_esc(narrative.get('paragraph', ''))))
        if narrative.get("why"):
            card.append("<p class='summary'><b>Why rating:</b> {}</p>".format(_esc(narrative.get('why', ''))))
        if narrative.get("steps"):
            steps_li = "".join(f"<li>{_esc(s)}</li>" for s in narrative["steps"])
            card.append(f"<ol class='steps'>{steps_li}</ol>")
        # Predicted assays & expected outputs (from mechanism-first hypothesis step)
        pred_assays = payload.get("assays") or []
        if isinstance(pred_assays, list) and pred_assays:
            def pred_assay_row_builder(it: dict) -> list:
                if not isinstance(it, dict):
                    return [_esc(str(it)), "", ""]
                return [_esc(str(it.get("assay",""))), _esc(str(it.get("readout",""))), _esc(str(it.get("expected","")))]
            card.append("<div class='section-title' style='margin-top:1rem'>Predicted assays & expected outputs</div>")
            card.append(tbl(pred_assays[:10], ["Assay","Readout","Expected"], pred_assay_row_builder, wrap_scroll=True))
        card.append("</div>")
        card.append("<div>")
        card.append("<div class='section-title'>Evidence-type breakdown</div>")
        card.append(f"<div class='chart-wrap'><canvas id='{pie_id}' style='width:100%; height:28vh;'></canvas></div>")
        card.append("</div>")
        card.append("</div>")
        card.append("</div>")
        card.append("<div style='margin-top:1rem'>"+details_html+"</div>")
        card.append("</div>")

        cards.append("".join(card))
        js_blocks.append("<script>"+pie_js+"</script>")
        js_blocks.append("<script>"+drs_js+"</script>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Skill 3 – {_esc(rbp)} / {_esc(tissue_query)} (phenotype inference)</title>
{css}
</head>
<body>
<header><h1>RBP–lncRNA phenotype inference report</h1></header>
<main>
  <div class='hdr'>
    <b>RBP:</b> {_esc(rbp.upper())}
    &nbsp; | &nbsp; <b>lncRNA:</b> {_esc((lncrna or '').upper() or 'None')}
    &nbsp; | &nbsp; <b>Tissue:</b> {_esc(tissue_query)} (GTEx column: {_esc(gtex_tissue_col or 'NA')})
    &nbsp; | &nbsp; <b>regulation_type:</b> {_esc(regulation_type)}
    {'&nbsp; | &nbsp; <b>user append:</b> ' + _esc(append_info) if append_info else ''}
    &nbsp; | &nbsp; <span class='muted'>Only evidence actually used in conclusions is highlighted.</span>
  </div>
  {"".join(cards) if cards else "<p>(No hypothesis cards)</p>"}
</main>
<footer>&copy; {datetime.datetime.now().year} LNC Research</footer>
{''.join(js_blocks)}
</body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# =========================
# 11) Excel export
# =========================
def export_excel(out_xlsx: str,
                 rbp: str,
                 lncrna: str,
                 tissue_query: str,
                 gtex_tissue_col: str | None,
                 regulation_type: str,
                 append_info: str,
                 hypotheses: List[Dict[str, Any]],
                 per_pair: Dict[str, Dict[str, Any]]):
    meta = pd.DataFrame([
        {"Key":"RBP","Value":rbp.upper()},
        {"Key":"lncRNA","Value":(lncrna or "").upper() or "None"},
        {"Key":"Tissue query","Value":tissue_query},
        {"Key":"GTEx tissue column","Value":gtex_tissue_col or "NA"},
        {"Key":"regulation_type","Value":regulation_type},
        {"Key":"user append info","Value":append_info or ""},
        {"Key":"Hypotheses (N)","Value":len(hypotheses)},
        {"Key":"Generated At","Value":datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
    ])

    rows_cards: List[Dict[str, Any]] = []
    rows_exps: List[Dict[str, Any]] = []

    for h in hypotheses:
        hid = str(h.get("id",""))
        trait = str(h.get("trait",""))
        direction = str(h.get("direction",""))
        hyp_stmt = str(h.get("hypothesis",""))
        pp = per_pair.get(hid, {}) or {}
        sel = pp.get("select_and_rate", {}) or {}
        # Flatten GWAS trait cluster + predicted assays for Excel
        gwas_list = pp.get("gwas_traits") or []
        gwas_flat = []
        for gt in gwas_list:
            if isinstance(gt, dict) and str(gt.get("trait","")).strip():
                gwas_flat.append(str(gt["trait"]).strip())
            elif isinstance(gt, str) and gt.strip():
                gwas_flat.append(gt.strip())

        assays_list = pp.get("assays") or []
        assay_flat = []
        for a in assays_list:
            if isinstance(a, dict):
                a_name = str(a.get("assay","")).strip()
                a_exp  = str(a.get("expected","")).strip()
                if a_name and a_exp:
                    assay_flat.append(f"{a_name} => {a_exp}")
                elif a_name:
                    assay_flat.append(a_name)
            elif isinstance(a, str) and a.strip():
                assay_flat.append(a.strip())

        rows_cards.append({
            "Hypothesis ID": hid,
            "Direction": direction,
            "Axis label": pp.get("axis_label", trait),
            "Trait/cluster label": trait,
            "Axis": safe_axis_label(rbp, trait, direction),
            "Mechanism hypothesis": (pp.get("mechanism") or hyp_stmt),
            "GWAS traits (cluster)": "; ".join(gwas_flat),
            "Predicted assays (expected)": "; ".join(assay_flat),
            "PMC queries": "; ".join([str(x) for x in (pp.get("pmc_queries") or []) if str(x).strip()]),
            "Mechanism keywords": "; ".join([str(x) for x in (pp.get("mechanism_keywords") or []) if str(x).strip()]),
            "Verdict": canonicalize_verdict(sel.get("verdict","Low")),
            "Rule justification": sel.get("rule_justification",""),
            "Why": (pp.get("narrative") or {}).get("why",""),
            "Used DB Refs": ";".join(sel.get("used_db_refs", [])),
            "Selected PPI Genes": ", ".join([p.get("gene","") for p in (pp.get("ppi_selected") or [])]),
        })

        for art in pp.get("articles", []):
            meta_a = art.get("meta", {})
            exps = [e for e in art.get("experiments", []) if e]
            for e in exps:
                rows_exps.append({
                    "Hypothesis ID": hid,
                    "Direction": direction,
                    "Trait": trait,
                    "PMCID/URL": meta_a.get("pmc_url",""),
                    "PMID": meta_a.get("pmid",""),
                    "Title": meta_a.get("title",""),
                    "Evidence Type": e.get("evidence_type",""),
                    "Methods": e.get("methods",""),
                    "Results": e.get("results",""),
                    "Locating Sentence": e.get("locating_sentence",""),
                })

    df_cards = pd.DataFrame(rows_cards)
    df_exps  = pd.DataFrame(rows_exps)

    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
            meta.to_excel(w, sheet_name="readme", index=False)
            df_cards.to_excel(w, sheet_name="hypotheses", index=False)
            df_exps.to_excel(w, sheet_name="experiments", index=False)
    except Exception:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            meta.to_excel(w, sheet_name="readme", index=False)
            df_cards.to_excel(w, sheet_name="hypotheses", index=False)
            df_exps.to_excel(w, sheet_name="experiments", index=False)
    return out_xlsx

# =========================
# 12) Main Integration Flow
# =========================
def build_hypothesis_prompt_payload(rbp: str,
                                   lncrna: str,
                                   tissue_query: str,
                                   gtex_tissue_col: str | None,
                                   regulation_type: str,
                                   append_info: str,
                                   rbp_gene_df: pd.DataFrame,
                                   assay_gene_df: pd.DataFrame,
                                   ppi_summary: List[Dict[str, Any]],
                                   gtex_expr: Dict[str, float | None],
                                   drs_sig_by_gene: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a structured payload for the *mechanism-first* GPT hypothesis step.

    Key change vs previous versions:
      - We pass a (near) complete GWAS trait list (deduplicated with best scores) so the model can
        infer a unifying mechanism and then select the subset of traits explained by that mechanism.
    """
    # --- GWAS trait candidates (dedup with best score) ---
    gwas_pairs: List[Tuple[str, float | None]] = []
    if rbp_gene_df is not None and not rbp_gene_df.empty:
        for _, row in rbp_gene_df.iterrows():
            pairs = parse_rbp_traits_with_scores(row)
            for t, sc in pairs:
                gwas_pairs.append((t, None if sc != sc else float(sc)))

    best: Dict[str, float | None] = {}
    name_map: Dict[str, str] = {}
    for t, sc in gwas_pairs:
        key = normalize_text(t)
        if not key:
            continue
        name_map.setdefault(key, t)
        prev = best.get(key)
        if prev is None:
            best[key] = sc
        else:
            # keep higher score if both are numeric
            if sc is not None and (prev is None or sc > prev):
                best[key] = sc

    def _sort_key(k: str):
        sc = best.get(k)
        # scored traits first
        if sc is None:
            return (1, 0.0, name_map.get(k, k))
        return (0, -float(sc), name_map.get(k, k))

    keys_sorted = sorted(best.keys(), key=_sort_key)

    # cap to avoid token blowups; still "all-ish" for typical genes
    MAX_GWAS_TRAITS = int(os.getenv("SKILL3_MAX_GWAS_TRAITS", os.getenv("P4_MAX_GWAS_TRAITS", "200")))
    keys_sorted = keys_sorted[:MAX_GWAS_TRAITS]

    gwas_candidates = [{"trait": name_map[k], "score": best.get(k)} for k in keys_sorted]

    # --- Gene_Assay evidence summary (trim) ---
    assay_rows: List[Dict[str, str]] = []
    if assay_gene_df is not None and not assay_gene_df.empty:
        for _, row in assay_gene_df.head(40).iterrows():
            assay_rows.append({
                "gene_symbol": str(row.get("gene_symbol","")).strip(),
                "ensg_id": str(row.get("ensg_id","")).strip(),
                "pmid": str(row.get("pmid","")).strip(),
                "pmcid": str(row.get("pmcid","")).strip(),
                "readout": str(row.get("readout","")).strip(),
                "explanation": str(row.get("explanation","")).strip(),
            })

    payload = {
        "rbp": rbp.upper(),
        "lncrna": (lncrna or "").upper(),
        "tissue_query": tissue_query,
        "gtex_tissue_col": gtex_tissue_col or "",
        "regulation_type": regulation_type,
        "user_append_info": append_info or "",
        "gwas_candidates": gwas_candidates,
        "gene_assay_rows": assay_rows,
        "ppi_partner_summary": ppi_summary,
        "gtex_expression": gtex_expr,          # gene->expr
        "drs_significant": drs_sig_by_gene,    # gene->(sig dict)
        "meta": {
            "gwas_trait_n": len(gwas_candidates),
            "assay_row_n": len(assay_rows),
            "ppi_partner_n": len(ppi_summary),
        }
    }
    return payload

def main(rbp: str,
         tissue: str,
         output: str,
         lncrna: str = "",
         regulation_type: str = "all",
         append_info: str = "",
         fresh: str = "Y"):
    rbp = (rbp or "").strip().upper()
    lncrna = (lncrna or "").strip().upper()
    tissue_query = (tissue or "").strip()
    regulation_type = (regulation_type or "all").strip().lower()
    if regulation_type not in {"all","up_regulation","down_regulation"}:
        regulation_type = "all"

    base = os.path.basename(output)
    stem = os.path.splitext(base)[0]
    out_html = os.path.join(TEMP_DIR, stem + ".html")
    out_xlsx = os.path.join(TEMP_DIR, stem + ".xlsx")

    snap_path = get_snapshot_path(rbp, lncrna, tissue_query, regulation_type, append_info)
    use_snapshot = (fresh or "Y").upper() == "N" and os.path.exists(snap_path)

    if use_snapshot:
        AG = AgentProgress(run="Skill3", total_steps=3)
        AG.start_step("Load snapshot", "snapshot", detail=os.path.basename(snap_path))
        _maybe_sleep()
        snap = load_snapshot(snap_path)
        hypotheses = snap.get("hypotheses", [])
        registry = snap.get("registry", {})
        per_pair = snap.get("per_pair", {})
        gtex_tissue_col = snap.get("statistics", {}).get("gtex_tissue_col")
        AG.done(f"snapshot: {len(hypotheses)} hypotheses, {len(registry)} registry items")
        _maybe_sleep()

        AG.start_step("Build HTML", "html")
        html_path = build_html_report(out_html, rbp, lncrna, tissue_query, gtex_tissue_col, regulation_type, append_info,
                                      hypotheses, registry, per_pair)
        AG.done(f"HTML → {html_path}", path=html_path); _maybe_sleep()

        AG.start_step("Export Excel", "excel")
        xlsx_path = export_excel(out_xlsx, rbp, lncrna, tissue_query, gtex_tissue_col, regulation_type, append_info,
                                 hypotheses, per_pair)
        AG.done(f"Excel → {xlsx_path}", path=xlsx_path)
        return {"html": html_path, "xlsx": xlsx_path}

    # Fresh run
    AG = AgentProgress(run="Skill3", total_steps=12)

    include_drs = is_liver_tissue(tissue_query)
    MAX_ARTICLES_PER_HYP = int(
        os.getenv("SKILL3_MAX_ARTICLES_PER_HYP", os.getenv("P4_MAX_ARTICLES_PER_HYP", "3"))
    )

    # 1) Load GTEx expression for tissue
    AG.start_step("Load GTEx tissue expression", "prepare")
    gtex_expr_map, gtex_tissue_col, gtex_matches = load_gtex_expression_dict(GTEX_TISSUE_FILE, tissue_query)
    if gtex_tissue_col is None:
        AG.update(f"GTEx tissue column not found for query '{tissue_query}'", stage="prepare",
                  matched=gtex_matches[:10])
    AG.done(f"GTEx loaded: tissue_col={gtex_tissue_col}, genes={len(gtex_expr_map)}")

    # 2) Load local DBs (Gene_Assay / RBPbase / PPI / optional DRS)
    AG.start_step("Load local DBs", "prepare")
    drs_human_df = load_drs_table(HUMAN_DRS) if include_drs else pd.DataFrame()
    drs_mouse_df = load_drs_table(MOUSE_DRS) if include_drs else pd.DataFrame()

    gene_assay_all = load_gene_assay_df(GENE_ASSAY_FILE)
    rbp_all = load_rbpbase_df(RBPBASE_FILE)

    rbp_gene_df = filter_rbpbase_for_gene(rbp_all, rbp)
    assay_gene_df = filter_gene_assay_for_gene(gene_assay_all, rbp)
    ppi_records = query_ppi_partners(rbp)
    ppi_summary = summarize_ppi_partners(ppi_records, rbp, max_genes=50)

    AG.done(f"DB OK: assay_rows={len(assay_gene_df)}, rbpbase_rows={len(rbp_gene_df)}, PPI_rows={len(ppi_records)}, include_drs={include_drs}")

    # 3) Base gene set for expression/DRS in hypothesis prompt
    AG.start_step("Assemble expression/DRS for base genes", "prepare")
    base_genes: List[str] = [rbp] + ([lncrna] if lncrna else []) + [x["gene"] for x in ppi_summary]
    base_genes = [g.upper() for g in base_genes if g]
    # GTEx expression subset
    gtex_expr_subset = get_gtex_expr_for_genes(gtex_expr_map, base_genes)
    # DRS significant subset (liver only)
    drs_sig_by_gene: Dict[str, Dict[str, Any]] = {}
    if include_drs:
        for g in base_genes:
            h_sig, m_sig = gather_drs_for_gene(g, drs_human_df, drs_mouse_df, alpha=0.05)
            merged = {}
            if h_sig:
                merged["human"] = h_sig
            if m_sig:
                merged["mouse"] = m_sig
            if merged:
                drs_sig_by_gene[g] = merged
    AG.done(f"Base genes={len(base_genes)}; GTEx subset={len(gtex_expr_subset)}; DRS genes={len(drs_sig_by_gene)}")

    # 4) GPT: propose hypotheses based on regulation_type
    AG.start_step("Propose hypotheses (GPT)", "reason")
    prompt_payload = build_hypothesis_prompt_payload(
        rbp=rbp,
        lncrna=lncrna,
        tissue_query=tissue_query,
        gtex_tissue_col=gtex_tissue_col,
        regulation_type=regulation_type,
        append_info=append_info,
        rbp_gene_df=rbp_gene_df,
        assay_gene_df=assay_gene_df,
        ppi_summary=ppi_summary,
        gtex_expr=gtex_expr_subset,
        drs_sig_by_gene=drs_sig_by_gene,
    )
    hypotheses = gpt_generate_regulation_hypotheses(prompt_payload, regulation_type=regulation_type, n_each=1)
    AG.done(f"Hypotheses generated: {len(hypotheses)}")

    # 5) Build global registry (DB rows + GTEx expression records)
    AG.start_step("Build evidence registry", "prepare")
    registry = build_db_registry(rbp, rbp_gene_df, assay_gene_df, ppi_records, drs_human_df, drs_mouse_df, include_drs=include_drs)
    # Add GTEx expression rows for base genes (and later we will use per-hypothesis expr from per_pair)
    gtex_reg = build_gtex_registry(gtex_expr_map, gtex_tissue_col or tissue_query, base_genes)
    registry.update(gtex_reg)
    AG.done(f"Registry items: {len(registry)}")

        # 6) Per-hypothesis setup (mechanism-first): genes/expr/DRS + search terms + article links
    AG.start_step("Per-hypothesis setup (genes/expr/DRS/search terms)", "search")
    per_pair: Dict[str, Dict[str, Any]] = {}

    # Gene_Assay referenced literature is always considered first (then we fill the remaining slots with search hits)
    assay_urls = extract_assay_pmc_urls(assay_gene_df, max_urls=25)
    AG.update(f"Assay literature URLs collected: {len(assay_urls)}", stage="search")

    for h in hypotheses:
        hid = str(h.get("id"))
        direction = str(h.get("direction","")).strip()
        axis_label = str(h.get("axis_label") or h.get("trait") or "").strip()
        mechanism = str(h.get("mechanism") or h.get("hypothesis") or "").strip()
        rationale = str(h.get("rationale","")).strip()

        gwas_traits = h.get("gwas_traits") or []
        if not isinstance(gwas_traits, list):
            gwas_traits = []
        # representative GWAS terms for search/query focus
        gwas_trait_terms = []
        for gt in gwas_traits:
            if isinstance(gt, dict) and str(gt.get("trait","")).strip():
                gwas_trait_terms.append(str(gt["trait"]).strip())
            elif isinstance(gt, str) and gt.strip():
                gwas_trait_terms.append(gt.strip())
        gwas_trait_terms = gwas_trait_terms[:5]

        assays = h.get("assays") or []
        if not isinstance(assays, list):
            assays = []

        pmc_queries = h.get("pmc_queries") or []
        if not isinstance(pmc_queries, list):
            pmc_queries = []

        mech_keywords = h.get("mechanism_keywords") or []
        if not isinstance(mech_keywords, list):
            mech_keywords = []

        # key genes (from the hypothesis step). These replace the old per-trait PPI-selection step.
        key_genes_raw = h.get("key_genes") or []
        if not isinstance(key_genes_raw, list):
            key_genes_raw = []
        key_genes = [str(x).strip().upper() for x in key_genes_raw if str(x).strip()]

        # If model didn't return any partners, fall back to the most frequent PPI partners
        if not key_genes:
            key_genes = [rbp] + ([lncrna] if lncrna else [])
        partner_genes = [g for g in key_genes if g and g not in {rbp.upper(), (lncrna or '').upper()}]
        if not partner_genes:
            partner_genes = [x.get("gene","").upper() for x in ppi_summary[:6] if str(x.get("gene","")).strip()]

        # context for downstream steps (kept compact to avoid prompt bloat later)
        top_traits_str = ", ".join(gwas_trait_terms[:8])
        context = (
            f"Tissue: {tissue_query} (GTEx column: {gtex_tissue_col or 'NA'}). "
            f"Direction: {direction}. "
            f"RBP: {rbp}. "
            f"lncRNA: {lncrna or 'None'}. "
            + (f"Axis: {axis_label}. " if axis_label else "")
            + (f"GWAS trait cluster (examples): {top_traits_str}. " if top_traits_str else "")
            + (f"Mechanism: {mechanism[:500]}" if mechanism else "")
        )
        if append_info:
            context = context + f" User info: {append_info}."

        # Materialize selected PPI rows for display (RBP ↔ partner_genes)
        sel_set = {g.upper() for g in partner_genes if g}
        ppi_rows_selected = []
        for r in ppi_records:
            ga = str(r.get("Gene Name A","")).upper()
            gb = str(r.get("Gene Name B","")).upper()
            if (ga == rbp.upper() and gb in sel_set) or (gb == rbp.upper() and ga in sel_set):
                ppi_rows_selected.append(r)

        ppi_selected = [{"gene": g, "why": "hypothesis-key-gene"} for g in sorted(sel_set)]

        # genes to show expression/DRS for this hypothesis
        hyp_genes = [rbp] + ([lncrna] if lncrna else []) + sorted(sel_set)
        # de-dup preserve order
        seen_g = set()
        hyp_genes = [g.upper() for g in hyp_genes if g and not (g.upper() in seen_g or seen_g.add(g.upper()))]

        gtex_expr_h = get_gtex_expr_for_genes(gtex_expr_map, hyp_genes)

        # DRS selection (liver only)
        drs_items = []
        if include_drs:
            for g in hyp_genes:
                hh, mm = gather_drs_for_gene(g, drs_human_df, drs_mouse_df, alpha=0.05)
                if hh:
                    drs_items.append({"id": f"DRS:H:{g}", "source":"DRS-H", "text": f"Human DRS significant for {g}", "payload": hh, "gene": g})
                if mm:
                    drs_items.append({"id": f"DRS:M:{g}", "source":"DRS-M", "text": f"Mouse DRS significant for {g}", "payload": mm, "gene": g})

        terms = build_search_terms_for_hypothesis(
            rbp=rbp,
            lncrna=lncrna or None,
            direction=direction,
            axis_label=axis_label,
            gwas_trait_terms=gwas_trait_terms,
            mechanism_keywords=mech_keywords,
            pmc_queries=pmc_queries,
            key_genes=hyp_genes,
            context=context,
            drs_items=drs_items,
        )

        # With only 1 hypothesis per direction, we keep search shallow.
        search_urls = collect_articles_from_terms(terms, per_term_limit=3)

        # Ensure Gene_Assay referenced literature is ALWAYS included (up to MAX_ARTICLES_PER_HYP)
        assay_urls_local = [u for u in assay_urls if u]
        links_to_read: List[str] = []
        seen_u = set()
        for u in assay_urls_local:
            if u not in seen_u:
                links_to_read.append(u); seen_u.add(u)
            if len(links_to_read) >= MAX_ARTICLES_PER_HYP:
                break
        if len(links_to_read) < MAX_ARTICLES_PER_HYP:
            for u in search_urls:
                if u not in seen_u:
                    links_to_read.append(u); seen_u.add(u)
                if len(links_to_read) >= MAX_ARTICLES_PER_HYP:
                    break

        per_pair[hid] = {
            # labeling
            "axis_label": axis_label,
            "trait": axis_label,  # backward-compat: used in axis label, Excel etc.
            "direction": direction,
            # hypothesis content
            "mechanism": mechanism,
            "hypothesis": mechanism,  # backward-compat
            "rationale": rationale,
            "gwas_traits": gwas_traits,
            "assays": assays,
            "pmc_queries": pmc_queries,
            "mechanism_keywords": mech_keywords,
            "key_genes": key_genes,
            # evidence tables
            "ppi_selected": ppi_selected,
            "ppi_selected_records": ppi_rows_selected,
            "gtex_expr": gtex_expr_h,
            "drs_selected": drs_items,
            # literature search
            "context": context,
            "terms": terms,
            "assay_urls": assay_urls_local,
            "search_urls": search_urls,
            "links": links_to_read,
        }
    AG.done("Per-hypothesis setup done")

    # 7) Fetch/extract articles (search + assay)
    AG.start_step("Fetch articles & extract experiments", "extract")
    article_cache: Dict[str, Any] = {}
    for h in hypotheses:
        hid = str(h.get("id"))
        pp = per_pair.get(hid, {})
        links = pp.get("links", [])
        trait = pp.get("trait","")
        direction = pp.get("direction","")
        # gene set for extraction = RBP + lncRNA + selected PPI genes + hypothesis key genes (if any)
        sel_ppi_genes = [str(it.get("gene","")).upper() for it in (pp.get("ppi_selected") or []) if str(it.get("gene","")).strip()]
        hyp_key_genes = [str(x).strip().upper() for x in (pp.get("key_genes") or []) if str(x).strip()]
        gene_set = [rbp] + ([lncrna] if lncrna else []) + sel_ppi_genes + hyp_key_genes
        # de-dup preserve order
        seen_g = set()
        gene_set = [g for g in gene_set if g and not (g.upper() in seen_g or seen_g.add(g.upper()))]
        # query string for extraction (short)
        query = f"{rbp} {trait} {direction}"
        arts = []
        for u in links:
            art = extract_article_evidence_cached(u, gene_set, query, cache=article_cache)
            if art:
                arts.append(art)
        # evidence type counts
        counts = {
            "Molecular Mechanism Experiments": 0,
            "Cellular Function Experiments": 0,
            "Animal Function Experiments": 0,
            "Clinical Information Analysis Experiments": 0,
            "Bioinformatic Analysis": 0,
        }
        for a in arts:
            for e in a.get("experiments", []):
                et = e.get("evidence_type","")
                if et in counts:
                    counts[et] += 1
        pp["articles"] = arts
        pp["ev_counts"] = {k:v for k,v in counts.items() if v>0}
        per_pair[hid] = pp
    AG.done("Articles parsed")

    # 8) Select USED DB rows & rate
    AG.start_step("Select USED evidence & rate hypotheses", "reason")
    for h in hypotheses:
        hid = str(h.get("id"))
        pp = per_pair.get(hid, {})
        trait = pp.get("trait","")
        direction = pp.get("direction","")
        context = pp.get("context","")
        # Build subset of registry ids relevant to this hypothesis:
        include_ids: Set[str] = set()
        # always include all RBP and ASSAY rows
        for rid, item in registry.items():
            if item.get("source") in {"RBP","ASSAY"}:
                include_ids.add(rid)
        # include selected PPI rows
        sel_genes = {str(it.get("gene","")).upper() for it in (pp.get("ppi_selected") or []) if str(it.get("gene","")).strip()}
        for rid, item in registry.items():
            if item.get("source") == "PPI":
                pay = item.get("payload") or {}
                ga = str(pay.get("Gene Name A","")).upper()
                gb = str(pay.get("Gene Name B","")).upper()
                if ga in sel_genes or gb in sel_genes:
                    include_ids.add(rid)
        # include DRS selected ids
        for it in pp.get("drs_selected", []) or []:
            if it.get("id"):
                include_ids.add(str(it["id"]))
            # ensure registry has this DRS row (hypothesis-specific genes)
            if it.get("id") and it.get("id") not in registry:
                registry[str(it["id"])] = {
                    "id": str(it["id"]),
                    "source": it.get("source","DRS-H"),
                    "text": it.get("text",""),
                    "payload": it.get("payload",{}),
                    "gene": it.get("gene",""),
                }
        # include GTEx rows for genes in this hypothesis
        for g in (pp.get("gtex_expr") or {}).keys():
            rid = f"GTEX:{gtex_tissue_col or tissue_query}:{g}"
            if rid in registry:
                include_ids.add(rid)
            else:
                # create on the fly
                registry[rid] = {
                    "id": rid,
                    "source": "GTEX",
                    "text": f"{g} {gtex_tissue_col or tissue_query} expr={pp['gtex_expr'].get(g)}",
                    "payload": {"gene": g, "tissue": gtex_tissue_col or tissue_query, "expression": pp["gtex_expr"].get(g)},
                    "gene": g,
                    "tissue": gtex_tissue_col or tissue_query,
                }
                include_ids.add(rid)

        # Provide the GWAS trait *cluster* (not a single trait) to help annotate RBPbase rows
        trait_cluster = []
        for gt in (pp.get("gwas_traits") or []):
            if isinstance(gt, dict) and str(gt.get("trait","")).strip():
                trait_cluster.append(str(gt["trait"]).strip())
            elif isinstance(gt, str) and gt.strip():
                trait_cluster.append(gt.strip())
        digest = registry_digest_for_hypothesis(registry, trait_cluster or trait, include_ids=include_ids)

        has_assay = any(item.get("source")=="ASSAY" for rid, item in registry.items() if rid in include_ids)
        selected_ppi_genes = [str(it.get("gene","")) for it in (pp.get("ppi_selected") or [])]
        sel = gpt_select_db_evidence_and_rate(
            rbp=rbp,
            axis_label=trait,
            trait_cluster=trait_cluster,
            mechanism=pp.get("mechanism",""),
            context=context,
            db_digest=digest,
            evidence_counts=pp.get("ev_counts", {}),
            has_assay_hits=has_assay,
            selected_ppi_genes=selected_ppi_genes,
        )
        pp["select_and_rate"] = sel
        per_pair[hid] = pp
    AG.done("Rating done")

    # 9) Compose narrative
    AG.start_step("Compose narratives", "reason")
    for h in hypotheses:
        hid = str(h.get("id"))
        pp = per_pair.get(hid, {})
        sel = pp.get("select_and_rate", {}) or {}
        used = set(sel.get("used_db_refs", []))

        used_summary: Dict[str, Any] = {"assay": [], "gwas": [], "ppi": [], "drs": [], "gtex": []}
        for uid in used:
            it = registry.get(uid)
            if not it:
                continue
            src = it.get("source")
            if src == "ASSAY":
                payload = it.get("payload", it)
                used_summary["assay"].append({k: payload.get(k,"") for k in ("gene_symbol","ensg_id","pmid","pmcid","readout","explanation")})
            elif src == "RBP":
                tsc = it.get("traits_scored") or []
                hit = best_trait_match_for_phenotype(tsc, pp.get("trait",""))
                if hit:
                    t, sc = hit
                    used_summary["gwas"].append({"trait": t, "score": None if sc!=sc else float(sc)})
            elif src == "PPI":
                used_summary["ppi"].append({"pair": it.get("text",""), "partner": it.get("partner","")})
            elif src in ("DRS-H","DRS-M"):
                regs = []
                for k,v in (it.get("payload") or {}).items():
                    if str(k).endswith("_log2fc"):
                        base = k.replace("_log2fc","")
                        try:
                            fc = float(v)
                        except Exception:
                            fc = None
                        if fc is not None:
                            regs.append({"condition": base, "log2fc": fc})
                used_summary["drs"].append({"gene": it.get("gene",""), "source": src, "signals": regs[:25]})
            elif src == "GTEX":
                pay = it.get("payload") or {}
                used_summary["gtex"].append({"gene": pay.get("gene"), "tissue": pay.get("tissue"), "expression": pay.get("expression")})

        narrative = gpt_compose_hypothesis_narrative(
            rbp=rbp,
            trait=pp.get("trait",""),
            direction=pp.get("direction",""),
            context=pp.get("context",""),
            hypothesis_statement=pp.get("hypothesis",""),
            used_evidence=used_summary,
            ev_counts=pp.get("ev_counts", {}),
            article_details=pp.get("articles", []),
            verdict=sel.get("verdict","Low"),
            rule_justification=sel.get("rule_justification",""),
        )
        pp["narrative"] = narrative
        per_pair[hid] = pp
    AG.done("Narratives done")

    # 10) Build HTML
    AG.start_step("Build HTML", "html")
    html_path = build_html_report(out_html, rbp, lncrna, tissue_query, gtex_tissue_col, regulation_type, append_info,
                                  hypotheses, registry, per_pair)
    AG.done(f"HTML → {html_path}", path=html_path)

    # 11) Export Excel
    AG.start_step("Export Excel", "excel")
    xlsx_path = export_excel(out_xlsx, rbp, lncrna, tissue_query, gtex_tissue_col, regulation_type, append_info,
                             hypotheses, per_pair)
    AG.done(f"Excel → {xlsx_path}", path=xlsx_path)

    # 12) Save snapshot
    AG.start_step("Save snapshot", "snapshot")
    try:
        stats = {
            "n_hypotheses": len(hypotheses),
            "n_registry": len(registry),
            "gtex_tissue_col": gtex_tissue_col,
            "gtex_matches": gtex_matches[:25],
            "include_drs": include_drs,
        }
        save_snapshot(snap_path, rbp, lncrna, tissue_query, regulation_type, append_info,
                      hypotheses, registry, per_pair, statistics=stats)
        AG.done(f"Snapshot → {snap_path}")
    except Exception as e:
        AG.update(f"Snapshot save failed: {e}", stage="snapshot")

    return {"html": html_path, "xlsx": xlsx_path}

# =========================
# 13) CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Skill 3: tissue-aware RBP/lncRNA phenotype inference")
    # New arguments
    ap.add_argument("--rbp", help="RBP gene symbol (required unless --gene is provided)")
    ap.add_argument("--lncrna", default="", help="lncRNA gene symbol (optional)")
    ap.add_argument("--tissue", required=True, help="GTEx tissue name or keyword")
    ap.add_argument("--regulation_type", choices=["all","up_regulation","down_regulation"], default="all",
                    help="Analyze RBP up/down effects, or both (default all)")
    ap.add_argument("--append_info", default="", help="Optional user notes appended into hypothesis prompt")
    ap.add_argument("--output", required=True, help="output file stem (writes to ./temp/<stem>.html/.xlsx)")
    ap.add_argument("--fresh", choices=["Y","N"], default="N", help="Y = run fresh; N = load snapshot if available")

    # Backward compatible aliases (deprecated)
    ap.add_argument("--gene", help="(Deprecated) alias of --rbp")
    ap.add_argument("--function", help="(Deprecated) mapped into --append_info")

    args = ap.parse_args()

    rbp = args.rbp or args.gene
    if not rbp:
        raise SystemExit("Missing required --rbp (or legacy --gene).")

    tissue = args.tissue
    if not tissue:
        raise SystemExit("Missing required --tissue.")

    append_info = args.append_info or (args.function or "")

    main(
        rbp=rbp,
        lncrna=args.lncrna or "",
        tissue=tissue,
        regulation_type=args.regulation_type,
        append_info=append_info,
        output=args.output,
        fresh=args.fresh,
    )
