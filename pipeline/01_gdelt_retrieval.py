"""
Step 1 — News acquisition from the GDELT Document 2.0 API.
==========================================================
Reference implementation of the retrieval stage described in Section 4.2 of the paper.
It is provided so that the acquisition protocol can be inspected and re-executed; the article
texts it downloads are third-party copyrighted material and are therefore NOT redistributed
in this repository.

Protocol
--------
* One ArtList query per calendar month, per query string, over 2018-01 .. 2025-08.
* `maxrecords = 250` is the per-request ceiling of the API, so the cap applies per month,
  not per week; stored monthly volumes never reached it (maximum 234).
* Client-side filter keeps Turkish-language items only.
* Full text is extracted with `trafilatura`, falling back to `newspaper3k`; items with fewer
  than 80 words are discarded as boilerplate.
* Each article is assigned to the ISO week starting on the preceding Monday, and the records are
  stored as one JSONL file per week of each monthly query.

Stock queries use the ticker plus its common legal/trade names, e.g.
    '("AKBNK" OR "Akbank" OR "Akbank T.A.Ş." OR "Akbank TAS" OR "Akbank Turkish Commercial Bank")'
Macro queries are:
    economy   : '("Ekonomi" OR "Türkiye Ekonomisi")'
    ambiguity : '("seçim" OR "cumhurbaşkanlığı seçimi" OR "geopolitical risk" OR "siyasi belirsizlik")'

Usage
-----
    python 01_gdelt_retrieval.py --query '("AKBNK" OR "Akbank")' --label AKBNK --out ./raw/AKBNK
"""
import argparse                                                # CLI
import calendar                                                # month lengths
import json                                                    # JSONL output
import time                                                    # rate limiting
from datetime import date, datetime, timedelta                 # dates
from pathlib import Path                                       # paths

import requests                                                # HTTP
import trafilatura                                             # main text extractor

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"  # Document 2.0 endpoint
MAX_RECORDS = 250                                              # API ceiling per request
LANG_LABEL = "Turkish"                                         # language filter
MIN_WORDS = 80                                                 # boilerplate threshold
SLEEP_SEC = 0.6                                                # polite pacing


def months(y0, m0, y1, m1):
    """Yield (first_day, last_day) for every calendar month in the sample window."""
    y, m = y0, m0                                              # cursor
    while (y, m) <= (y1, m1):                                  # inclusive end month
        yield date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)           # advance one month


def fetch_month(query, d1, d2):
    """One ArtList request for a single month; returns Turkish-language records."""
    params = {"query": query, "mode": "ArtList", "maxrecords": MAX_RECORDS,
              "sort": "DateDesc", "format": "JSON",
              "startdatetime": d1.strftime("%Y%m%d") + "000000",
              "enddatetime": d2.strftime("%Y%m%d") + "235959"}
    try:
        r = requests.get(GDELT_DOC_URL, params=params, timeout=30)   # request
        r.raise_for_status()                                          # surface HTTP errors
        items = r.json().get("articles", []) or []                    # record list
    except Exception as e:                                            # network/parse failure
        print(f"[WARN] {d1}..{d2}: {e}")                              # log and continue
        return []
    return [a for a in items if LANG_LABEL.lower() in (a.get("language", "") or "").lower()]


def article_date(raw):
    """Parse the several date formats GDELT returns; None if unparseable."""
    if not raw:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%d"):   # known layouts
        try:
            return datetime.strptime(raw, fmt).date()          # first match wins
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()   # ISO fallback
    except Exception:
        return None


def extract_text(url):
    """Full-text extraction; returns '' when the body is missing or too short."""
    try:
        html = trafilatura.fetch_url(url)                      # download
        txt = trafilatura.extract(html) if html else None      # boilerplate removal
    except Exception:
        txt = None
    if txt and len(txt.split()) >= MIN_WORDS:                  # length gate
        return txt
    try:                                                        # secondary extractor
        from newspaper import Article
        art = Article(url, language="tr"); art.download(); art.parse()
        return art.text if len(art.text.split()) >= MIN_WORDS else ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()                             # arguments
    ap.add_argument("--query", required=True)                  # GDELT query string
    ap.add_argument("--label", required=True)                  # ticker code or macro source name
    ap.add_argument("--out", required=True)                    # output directory
    ap.add_argument("--start", default="2018-01")              # first month
    ap.add_argument("--end", default="2025-08")                # last month
    a = ap.parse_args()

    y0, m0 = map(int, a.start.split("-")); y1, m1 = map(int, a.end.split("-"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)  # ensure output dir
    for d1, d2 in months(y0, m0, y1, m1):                      # month by month
        recs, seen = [], set()                                 # buffer + URL de-duplication
        for it in fetch_month(a.query, d1, d2):                # each returned article
            url = it.get("url")                                # identity key
            if not url or url in seen:                         # skip duplicates
                continue
            seen.add(url)
            d = article_date(it.get("seendate"))                # publication date
            txt = extract_text(url)                             # body text
            if not (d and txt):                                 # both are required
                continue
            recs.append({"ticker": a.label, "date": d.isoformat(),
                         "week_start": (d - timedelta(days=d.weekday())).isoformat(),  # Monday
                         "url": url, "source": it.get("domain", ""),
                         "title": it.get("title", ""), "language": it.get("language", ""),
                         "text": txt})
            time.sleep(SLEEP_SEC)                               # be polite to publishers
        weeks = {}                                             # Monday-anchored weekly buckets
        for r in recs:
            weeks.setdefault(r["week_start"], []).append(r)
        for wk, rows in sorted(weeks.items()):                 # one JSONL file per week
            path = out / f"{a.label}_{d1:%Y%m}_{wk.replace('-', '')}.jsonl"
            with path.open("w", encoding="utf-8") as f:         # write JSONL
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{d1:%Y-%m}: {len(recs):3d} articles in {len(weeks)} weekly files")


if __name__ == "__main__":
    main()
