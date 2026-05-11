#!/usr/bin/env python3
"""
Map AI model dependencies across enterprise vendors.

Reads a CSV of (vendor_name, website), scrapes each vendor's site + GitHub org,
calls Anthropic's Batch API with Haiku 4.5 to extract foundation model dependencies,
and writes results to SQLite. Low-confidence extractions are flagged for review.

Usage:
    python extract_dependencies.py vendors.csv --db deps.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urljoin, urlparse

import anthropic
import requests
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from bs4 import BeautifulSoup

MODEL = "claude-haiku-4-5"
USER_AGENT = "ai-concentration-research/1.0 (+research; contact admin)"
HTTP_TIMEOUT = 20
PAGE_CHAR_LIMIT = 40_000  # per page after extraction; keeps batch payloads sane
MAX_GITHUB_REPOS = 5
CONFIDENCE_THRESHOLD = 0.8
DEFAULT_PATHS = ["/", "/blog", "/partners", "/docs", "/about", "/technology"]

ALLOWED_RELATIONSHIPS = {
    "core_dependency",
    "multi_model_support",
    "optional_connector",
    "proprietary_model",
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "models": {
            "type": "array",
            "description": "Foundation model dependencies found in the source material.",
            "items": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Canonical model name (e.g. 'GPT-4', 'Claude 3.5 Sonnet', 'Llama 3'). Use the most specific version mentioned.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Lab/company that produces the model (e.g. 'OpenAI', 'Anthropic', 'Meta', 'Google', 'Mistral').",
                    },
                    "cloud_wrapper": {
                        "type": ["string", "null"],
                        "description": "Cloud platform mediating access if mentioned (e.g. 'Azure OpenAI', 'AWS Bedrock', 'GCP Vertex AI'); null if direct API or unspecified.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0 to 1.0. 0.9+ for explicit named dependencies. 0.7-0.9 for clear but indirect mentions. <0.7 for weak inference.",
                    },
                    "source_excerpt": {
                        "type": "string",
                        "description": "Exact quote (<= 300 chars) from the source supporting this extraction.",
                    },
                },
                "required": ["model_name", "provider", "cloud_wrapper", "confidence", "source_excerpt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["models"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You analyze enterprise vendor materials (websites, docs, GitHub READMEs) to identify which foundation/AI models the vendor depends on.

Rules:
- Only extract foundation models the vendor *uses or integrates with*. Do NOT extract models the vendor merely lists as competitors, mentions in passing, or supports as one option among many in unused integrations.
- A "foundation model" is a general-purpose LLM, vision model, or embedding model from a lab like OpenAI, Anthropic, Meta, Google, Mistral, Cohere, AI21, etc. Skip narrow ML models (e.g. fraud-detection classifiers).
- Distinguish direct API use from cloud-mediated access. If text says "we use GPT-4 via Azure OpenAI", set cloud_wrapper="Azure OpenAI". If just "we use GPT-4", set cloud_wrapper=null.
- Be conservative with confidence. If a vendor lists an SDK as a dependency in package.json but doesn't describe using it, that's <0.7. Marketing copy claiming "powered by GPT-4" is 0.9+.
- Deduplicate: if the same model appears in multiple sources, return ONE entry with the strongest excerpt.
- Return an empty array if no foundation model dependencies are evident. Do not invent.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Source:
    url: str
    content_type: str  # "homepage", "blog", "docs", "partners", "github_readme", etc.
    text: str
    fetched_at: str


@dataclass
class Vendor:
    name: str
    website: str
    github_org: str | None = None
    sources: list[Source] = field(default_factory=list)
    db_id: int | None = None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS vendors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    website TEXT NOT NULL,
    github_org TEXT,
    scraped_at TEXT
);

CREATE TABLE IF NOT EXISTS foundation_models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    cloud_wrapper TEXT,
    UNIQUE (name, provider, cloud_wrapper)
);

CREATE TABLE IF NOT EXISTS vendor_model_links (
    id INTEGER PRIMARY KEY,
    vendor_id INTEGER NOT NULL REFERENCES vendors(id),
    model_id INTEGER NOT NULL REFERENCES foundation_models(id),
    confidence REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('auto', 'review', 'confirmed', 'rejected')),
    source_excerpt TEXT,
    extracted_at TEXT NOT NULL,
    relationship_type TEXT CHECK (
        relationship_type IS NULL
        OR relationship_type IN ('core_dependency', 'multi_model_support', 'optional_connector', 'proprietary_model')
    ),
    UNIQUE (vendor_id, model_id)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    vendor_id INTEGER NOT NULL REFERENCES vendors(id),
    url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE (vendor_id, url)
);

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    submitted_at TEXT NOT NULL,
    request_count INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('submitted', 'processed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_links_status ON vendor_model_links(status);
CREATE INDEX IF NOT EXISTS idx_sources_vendor ON sources(vendor_id);
"""


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_vendor(conn: sqlite3.Connection, vendor: Vendor) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO vendors (name, website, github_org, scraped_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               website = excluded.website,
               github_org = excluded.github_org,
               scraped_at = excluded.scraped_at
           RETURNING id""",
        (vendor.name, vendor.website, vendor.github_org, now),
    )
    return cur.fetchone()[0]


def insert_sources(conn: sqlite3.Connection, vendor_id: int, sources: list[Source]) -> None:
    conn.executemany(
        """INSERT INTO sources (vendor_id, url, content_type, content, fetched_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(vendor_id, url) DO UPDATE SET
               content_type = excluded.content_type,
               content = excluded.content,
               fetched_at = excluded.fetched_at""",
        [(vendor_id, s.url, s.content_type, s.text, s.fetched_at) for s in sources],
    )


def load_sources_from_db(conn: sqlite3.Connection, vendor_id: int) -> list[Source]:
    rows = conn.execute(
        "SELECT url, content_type, content, fetched_at FROM sources WHERE vendor_id = ? ORDER BY id",
        (vendor_id,),
    ).fetchall()
    return [Source(url=r[0], content_type=r[1], text=r[2], fetched_at=r[3]) for r in rows]


def lookup_vendor_id(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT id FROM vendors WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def upsert_model(conn: sqlite3.Connection, name: str, provider: str, cloud_wrapper: str | None) -> int:
    cur = conn.execute(
        "SELECT id FROM foundation_models WHERE name = ? AND provider = ? AND IFNULL(cloud_wrapper, '') = IFNULL(?, '')",
        (name, provider, cloud_wrapper),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO foundation_models (name, provider, cloud_wrapper) VALUES (?, ?, ?) RETURNING id",
        (name, provider, cloud_wrapper),
    )
    return cur.fetchone()[0]


def insert_link(
    conn: sqlite3.Connection,
    vendor_id: int,
    model_id: int,
    confidence: float,
    status: str,
    excerpt: str,
) -> None:
    conn.execute(
        """INSERT INTO vendor_model_links (vendor_id, model_id, confidence, status, source_excerpt, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(vendor_id, model_id) DO UPDATE SET
               confidence = excluded.confidence,
               status = CASE
                   WHEN vendor_model_links.status IN ('confirmed', 'rejected') THEN vendor_model_links.status
                   ELSE excluded.status
               END,
               source_excerpt = excluded.source_excerpt,
               extracted_at = excluded.extracted_at""",
        (vendor_id, model_id, confidence, status, excerpt, datetime.now(timezone.utc).isoformat()),
    )


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def http_get(session: requests.Session, url: str) -> str | None:
    try:
        r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and "text" in r.headers.get("content-type", "").lower():
            return r.text
    except requests.RequestException as e:
        logging.debug("fetch failed %s: %s", url, e)
    return None


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:PAGE_CHAR_LIMIT]


def find_github_org_in_html(html: str, vendor_domain: str) -> str | None:
    """Look for github.com/<org> links in homepage HTML; reject user-style paths."""
    matches = re.findall(r"https?://github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})(?:/|$|\")", html)
    blocklist = {"login", "join", "features", "pricing", "about", "explore"}
    counts: dict[str, int] = {}
    for m in matches:
        m = m.lower()
        if m in blocklist or len(m) < 2:
            continue
        counts[m] = counts.get(m, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.lower()).strip("-")
    return s


def scrape_vendor_site(session: requests.Session, vendor: Vendor) -> None:
    base = vendor.website.rstrip("/")
    domain = urlparse(base).netloc

    homepage_html = http_get(session, base)
    homepage_text = ""
    if homepage_html:
        homepage_text = extract_text(homepage_html)
        vendor.sources.append(Source(
            url=base, content_type="homepage", text=homepage_text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        ))

    for path in DEFAULT_PATHS[1:]:
        url = urljoin(base + "/", path.lstrip("/"))
        html = http_get(session, url)
        if not html:
            continue
        text = extract_text(html)
        if len(text) < 200:
            continue
        vendor.sources.append(Source(
            url=url,
            content_type=path.strip("/") or "root",
            text=text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        ))
        time.sleep(0.5)

    # GitHub discovery: scrape homepage links first, fall back to slug guess
    if homepage_html:
        scraped_org = find_github_org_in_html(homepage_html, domain)
        if scraped_org:
            vendor.github_org = scraped_org

    if not vendor.github_org:
        slug = slugify(vendor.name)
        if slug:
            probe = http_get(session, f"https://github.com/{slug}")
            if probe and "Page not found" not in probe[:5000]:
                vendor.github_org = slug


def scrape_github_org(session: requests.Session, vendor: Vendor) -> None:
    if not vendor.github_org:
        return
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = session.get(
            f"https://api.github.com/orgs/{vendor.github_org}/repos",
            params={"sort": "updated", "per_page": MAX_GITHUB_REPOS},
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 404:
            r = session.get(
                f"https://api.github.com/users/{vendor.github_org}/repos",
                params={"sort": "updated", "per_page": MAX_GITHUB_REPOS},
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
        if r.status_code != 200:
            logging.debug("github list failed for %s: %s", vendor.github_org, r.status_code)
            return
        repos = r.json()
    except requests.RequestException as e:
        logging.debug("github error for %s: %s", vendor.github_org, e)
        return

    for repo in repos[:MAX_GITHUB_REPOS]:
        full = repo.get("full_name")
        if not full:
            continue
        readme_url = f"https://raw.githubusercontent.com/{full}/HEAD/README.md"
        text = http_get(session, readme_url)
        if not text:
            continue
        vendor.sources.append(Source(
            url=f"https://github.com/{full}",
            content_type="github_readme",
            text=text[:PAGE_CHAR_LIMIT],
            fetched_at=datetime.now(timezone.utc).isoformat(),
        ))
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Anthropic Batch extraction
# ---------------------------------------------------------------------------

def excerpt_supports_model(model_name: str, excerpt: str) -> bool:
    """Lenient check: does the excerpt contain the model name?

    Tries direct, whitespace-collapsed, and hyphen-collapsed matches so
    'GPT-4' matches 'gpt 4', 'GPT4', and 'GPT-4'. Catches fabrications
    where the model self-reports 'no explicit mention found'.
    """
    if not model_name or not excerpt:
        return False
    excerpt_l = excerpt.lower()
    name_l = model_name.lower()
    if name_l in excerpt_l:
        return True
    name_compact = re.sub(r"\s+", "", name_l)
    excerpt_compact = re.sub(r"\s+", "", excerpt_l)
    if name_compact in excerpt_compact:
        return True
    name_nohyp = re.sub(r"[-\s]+", "", name_l)
    excerpt_nohyp = re.sub(r"[-\s]+", "", excerpt_l)
    return name_nohyp in excerpt_nohyp


def excerpt_in_sources(excerpt: str, source_content: str, ngram: int = 5) -> bool:
    """Check that the excerpt actually appears in the scraped source content.

    Catches fabrications where the model invented a plausible-sounding quote.
    Tries direct substring match (whitespace-normalized), then any N-word
    sequence from the excerpt as a substring.
    """
    if not excerpt or not source_content:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    e_norm = norm(excerpt)
    c_norm = norm(source_content)
    if e_norm in c_norm:
        return True
    words = e_norm.split()
    if len(words) < ngram:
        return False
    for i in range(len(words) - ngram + 1):
        if " ".join(words[i:i + ngram]) in c_norm:
            return True
    return False


def validate_extraction(
    model_name: str,
    excerpt: str,
    vendor_source_content: str,
) -> tuple[bool, str]:
    """Two-layer fabrication check. Returns (passes, reason_if_not)."""
    if not excerpt_supports_model(model_name, excerpt):
        return False, "model_name_not_in_excerpt"
    if not excerpt_in_sources(excerpt, vendor_source_content):
        return False, "excerpt_not_in_sources"
    return True, ""


def import_classifications(conn: sqlite3.Connection, path: str) -> dict[str, int]:
    """Read relationship_type values from CSV; update vendor_model_links by id.

    CSV must have at least 'id' and 'relationship_type' columns. Blank values
    are skipped (won't clear existing classifications). Invalid values and
    missing ids are logged and counted, but don't abort the run.
    """
    counts = {"updated": 0, "skipped_blank": 0, "invalid_value": 0, "missing_id": 0}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if "id" not in cols or "relationship_type" not in cols:
            raise SystemExit(
                f"CSV must have 'id' and 'relationship_type' columns; got {reader.fieldnames}"
            )
        for row in reader:
            raw_id = (row.get("id") or "").strip()
            try:
                link_id = int(raw_id)
            except ValueError:
                counts["missing_id"] += 1
                logging.warning("skipping row with non-integer id: %r", raw_id)
                continue
            rel = (row.get("relationship_type") or "").strip()
            if not rel:
                counts["skipped_blank"] += 1
                continue
            if rel not in ALLOWED_RELATIONSHIPS:
                counts["invalid_value"] += 1
                logging.warning("id=%d: invalid relationship_type %r (allowed: %s)",
                                link_id, rel, sorted(ALLOWED_RELATIONSHIPS))
                continue
            cur = conn.execute(
                "UPDATE vendor_model_links SET relationship_type = ? WHERE id = ?",
                (rel, link_id),
            )
            if cur.rowcount == 0:
                counts["missing_id"] += 1
                logging.warning("id=%d: no matching row in vendor_model_links", link_id)
            else:
                counts["updated"] += 1
    conn.commit()
    return counts


def get_vendor_source_content(conn: sqlite3.Connection, vendor_id: int) -> str:
    rows = conn.execute(
        "SELECT content FROM sources WHERE vendor_id = ?", (vendor_id,)
    ).fetchall()
    return "\n\n".join(r[0] for r in rows if r[0])


def build_user_message(vendor: Vendor) -> str:
    parts = [f"Vendor: {vendor.name}", f"Website: {vendor.website}"]
    if vendor.github_org:
        parts.append(f"GitHub: github.com/{vendor.github_org}")
    parts.append("\n--- SOURCES ---\n")
    for src in vendor.sources:
        parts.append(f"\n## {src.content_type.upper()} ({src.url})\n{src.text}\n")
    parts.append("\nExtract foundation model dependencies as structured JSON.")
    return "\n".join(parts)


def submit_batch(client: anthropic.Anthropic, conn: sqlite3.Connection, vendors: list[Vendor]) -> str:
    """Build batch requests; cache the system prompt across all vendors."""
    requests_list = []
    for v in vendors:
        if not v.sources:
            continue
        requests_list.append(Request(
            custom_id=f"vendor-{v.db_id}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": build_user_message(v)}],
                output_config={"format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
            ),
        ))

    if not requests_list:
        raise RuntimeError("no vendors with scraped content to submit")

    batch = client.messages.batches.create(requests=requests_list)
    conn.execute(
        "INSERT INTO batches (id, submitted_at, request_count, status) VALUES (?, ?, ?, 'submitted')",
        (batch.id, datetime.now(timezone.utc).isoformat(), len(requests_list)),
    )
    conn.commit()
    logging.info("submitted batch %s with %d requests (saved to db)", batch.id, len(requests_list))
    return batch.id


def mark_batch_processed(conn: sqlite3.Connection, batch_id: str) -> None:
    conn.execute("UPDATE batches SET status = 'processed' WHERE id = ?", (batch_id,))
    conn.commit()


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 60) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        logging.info(
            "batch %s status=%s processing=%d succeeded=%d errored=%d",
            batch_id, batch.processing_status,
            batch.request_counts.processing, batch.request_counts.succeeded,
            batch.request_counts.errored,
        )
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def process_results(
    client: anthropic.Anthropic,
    batch_id: str,
    conn: sqlite3.Connection,
    threshold: float,
) -> dict[str, int]:
    counts = {"auto": 0, "review": 0, "errored": 0, "no_models": 0,
              "model_name_not_in_excerpt": 0, "excerpt_not_in_sources": 0}
    source_cache: dict[int, str] = {}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if not custom_id.startswith("vendor-"):
            continue
        vendor_id = int(custom_id.removeprefix("vendor-"))

        if result.result.type != "succeeded":
            counts["errored"] += 1
            logging.warning("vendor_id=%s failed: %s", vendor_id, result.result.type)
            continue

        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            counts["errored"] += 1
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            logging.warning("vendor_id=%s bad JSON: %s", vendor_id, e)
            counts["errored"] += 1
            continue

        models = payload.get("models", [])
        if not models:
            counts["no_models"] += 1
            continue

        if vendor_id not in source_cache:
            source_cache[vendor_id] = get_vendor_source_content(conn, vendor_id)
        for m in models:
            model_id = upsert_model(conn, m["model_name"], m["provider"], m.get("cloud_wrapper"))
            confidence = float(m["confidence"])
            excerpt = m.get("source_excerpt", "") or ""
            valid, reason = validate_extraction(m["model_name"], excerpt, source_cache[vendor_id])
            if not valid:
                status = "review"
                counts[reason] += 1
            else:
                status = "auto" if confidence >= threshold else "review"
            insert_link(conn, vendor_id, model_id, confidence, status, excerpt)
            counts[status] += 1
        conn.commit()

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_vendors(path: str) -> list[Vendor]:
    vendors: list[Vendor] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        name_col = cols.get("name") or cols.get("vendor") or cols.get("vendor_name")
        site_col = cols.get("website") or cols.get("url") or cols.get("site")
        if not name_col or not site_col:
            raise SystemExit("CSV must have name and website columns")
        for row in reader:
            name = (row[name_col] or "").strip()
            site = (row[site_col] or "").strip()
            if not name or not site:
                continue
            if name in seen:
                duplicates.append(name)
                continue
            seen.add(name)
            if not site.startswith("http"):
                site = "https://" + site
            vendors.append(Vendor(name=name, website=site))
    if duplicates:
        logging.warning("dropped %d duplicate vendor name(s) in CSV: %s",
                        len(duplicates), ", ".join(duplicates))
    return vendors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="CSV with vendor name + website columns (omit when using --resume or --list-batches)")
    parser.add_argument("--db", default="dependencies.sqlite", help="SQLite output path")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence below this is flagged for review (default: 0.8)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch status polls")
    parser.add_argument("--resume", metavar="BATCH_ID",
                        help="Skip scraping/submitting; resume polling and processing an existing batch")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Reuse sources already in the DB instead of re-scraping (CSV vendors must already exist in the DB)")
    parser.add_argument("--scrape-only", action="store_true",
                        help="Scrape and store sources, then exit without submitting a batch (use --skip-scrape later to submit)")
    parser.add_argument("--list-batches", action="store_true",
                        help="Show batches recorded in the DB and exit")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-check excerpt-vs-model-name on existing 'auto' rows; demote unsupported ones to 'review' and exit")
    parser.add_argument("--import-classifications", metavar="CSV",
                        help="Import relationship_type values from CSV (must have 'id' and 'relationship_type' columns); blank values skipped")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    conn = init_db(args.db)

    if args.scrape_only and args.skip_scrape:
        parser.error("--scrape-only and --skip-scrape are mutually exclusive")
    if args.scrape_only and args.resume:
        parser.error("--scrape-only and --resume are mutually exclusive")

    if args.import_classifications:
        if any([args.resume, args.scrape_only, args.skip_scrape, args.list_batches, args.revalidate]):
            parser.error("--import-classifications cannot be combined with other mode flags")
        counts = import_classifications(conn, args.import_classifications)
        logging.info(
            "import: updated=%d skipped_blank=%d invalid_value=%d missing_id=%d",
            counts["updated"], counts["skipped_blank"],
            counts["invalid_value"], counts["missing_id"],
        )
        conn.close()
        return 0

    if args.list_batches:
        for row in conn.execute(
            "SELECT id, submitted_at, request_count, status FROM batches ORDER BY submitted_at DESC"
        ):
            print(f"{row[0]}  submitted={row[1]}  requests={row[2]}  status={row[3]}")
        conn.close()
        return 0

    if args.revalidate:
        rows = conn.execute(
            """SELECT l.id, l.vendor_id, m.name, l.source_excerpt
               FROM vendor_model_links l
               JOIN foundation_models m ON m.id = l.model_id
               WHERE l.status = 'auto'"""
        ).fetchall()
        source_cache: dict[int, str] = {}
        demote_reasons = {"model_name_not_in_excerpt": 0, "excerpt_not_in_sources": 0}
        for link_id, vendor_id, model_name, excerpt in rows:
            if vendor_id not in source_cache:
                source_cache[vendor_id] = get_vendor_source_content(conn, vendor_id)
            valid, reason = validate_extraction(model_name, excerpt or "", source_cache[vendor_id])
            if not valid:
                conn.execute(
                    "UPDATE vendor_model_links SET status = 'review' WHERE id = ?",
                    (link_id,),
                )
                demote_reasons[reason] += 1
        conn.commit()
        total_demoted = sum(demote_reasons.values())
        logging.info(
            "revalidate: scanned %d auto rows, demoted %d (%d name-not-in-excerpt, %d excerpt-not-in-sources)",
            len(rows), total_demoted,
            demote_reasons["model_name_not_in_excerpt"],
            demote_reasons["excerpt_not_in_sources"],
        )
        conn.close()
        return 0

    client = anthropic.Anthropic()

    if args.resume:
        batch_id = args.resume
        logging.info("resuming batch %s", batch_id)
    else:
        if not args.csv:
            parser.error("csv argument is required unless --resume or --list-batches is used")
        vendors = load_vendors(args.csv)
        logging.info("loaded %d vendors", len(vendors))

        if args.skip_scrape:
            missing = []
            for v in vendors:
                vid = lookup_vendor_id(conn, v.name)
                if vid is None:
                    missing.append(v.name)
                    continue
                v.db_id = vid
                v.sources = load_sources_from_db(conn, vid)
                logging.info("  %s: loaded %d sources from db", v.name, len(v.sources))
            if missing:
                parser.error(f"--skip-scrape: vendor(s) not in db: {', '.join(missing)}")
        else:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})

            for v in vendors:
                logging.info("scraping %s", v.name)
                scrape_vendor_site(session, v)
                scrape_github_org(session, v)
                v.db_id = upsert_vendor(conn, v)
                insert_sources(conn, v.db_id, v.sources)
                conn.commit()
                logging.info("  %d sources, github=%s", len(v.sources), v.github_org or "-")

        if args.scrape_only:
            thin = [v.name for v in vendors if len(v.sources) < 3]
            logging.info("scrape-only: %d vendors scraped, %d with <3 sources",
                         len(vendors), len(thin))
            if thin:
                logging.warning("thin-source vendors (consider deepening): %s", ", ".join(thin))
            logging.info("re-run with --skip-scrape to submit the batch")
            conn.close()
            return 0

        batch_id = submit_batch(client, conn, vendors)

    wait_for_batch(client, batch_id, poll_interval=args.poll_interval)
    counts = process_results(client, batch_id, conn, args.threshold)
    mark_batch_processed(conn, batch_id)

    logging.info(
        "done: auto=%d review=%d no_models=%d errored=%d "
        "(demoted: %d name-not-in-excerpt, %d excerpt-not-in-sources)",
        counts["auto"], counts["review"], counts["no_models"], counts["errored"],
        counts["model_name_not_in_excerpt"], counts["excerpt_not_in_sources"],
    )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
