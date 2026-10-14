"""
Step 2 — Multi-LLM enrichment of each article.
==============================================
Reference implementation of the enrichment stage (Sections 3.2 and 4.3 of the paper). Every article is
scored independently by three models accessed through OpenRouter:

    openai/gpt-4.1-nano
    meta-llama/llama-3.3-70b-instruct
    google/gemma-4-26b-a4b-it

Look-ahead control
------------------
Each request carries exactly one `user` message containing one article and nothing else: no
system message, no conversation history, no retrieved context. Information from a different
article -- in particular a later one -- therefore cannot reach the model, by construction. The
macro prompt additionally states the constraint explicitly ([KRİTİK KISITLAMA]).

The prompt templates are read verbatim from ../prompts/ so that the text used in the paper and
the text used in the code cannot drift apart.

Set the API key in the environment; it is never stored in this file:
    export OPENROUTER_API_KEY=...

Usage
-----
    python 02_llm_enrichment.py --kind stock --ticker AKBNK --in ./raw/AKBNK --out ./enriched/AKBNK/gpt \
                                --model openai/gpt-4.1-nano
"""
import argparse                                                # CLI
import asyncio                                                 # concurrency
import json                                                    # JSONL / JSON parsing
import os                                                      # environment
import re                                                      # JSON extraction from replies
from pathlib import Path                                       # paths

from openai import AsyncOpenAI                                 # OpenRouter is OpenAI-compatible

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"      # prompt template directory
TEMPERATURE = 0.1                                              # near-deterministic scoring
MAX_TOKENS = 1000                                              # response ceiling
MIN_CHARS = 50                                                 # skip near-empty bodies


def load_template(kind):
    """Read a prompt template and return only the block between the two ruled lines."""
    fn = "stock_news_prompt_tr.txt" if kind == "stock" else "macro_news_prompt_tr.txt"
    lines = (PROMPTS / fn).read_text(encoding="utf-8").splitlines()        # documented file
    rules = [i for i, l in enumerate(lines) if set(l.strip()) == {"-"} and len(l.strip()) > 20]
    return "\n".join(lines[rules[0] + 1:rules[1]]).strip()                 # the prompt itself


def build_prompt(tpl, kind, ticker, text):
    """Substitute the placeholders; truncation limits follow the paper (5k stock / 15k macro)."""
    limit = 5000 if kind == "stock" else 15000                 # per-kind character cap
    out = tpl.replace("{text[:5000]}", text[:limit])           # stock placeholder
    out = out.replace("{text[:15000]}", text[:limit])          # macro placeholder
    return out.replace("{ticker}", ticker or "")               # ticker placeholder


async def score_one(client, model, prompt):
    """One independent completion; returns the parsed JSON object or None."""
    try:
        r = await client.chat.completions.create(              # single-turn request
            model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])    # NO system message, NO history
        m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)   # tolerate stray prose
        return json.loads(m.group(0)) if m else None           # parsed enrichment
    except Exception as e:
        print(f"[WARN] {e}")                                   # log and drop the record
        return None


async def run(args):
    key = os.environ.get("OPENROUTER_API_KEY")                 # credential from environment
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment.")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    tpl = load_template(args.kind)                             # verbatim template
    sem = asyncio.Semaphore(args.concurrency)                  # bound in-flight requests
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    async def one(rec):
        """Enrich a single article record under the concurrency limit."""
        text = rec.get("text", "")                             # article body
        if len(text) <= MIN_CHARS:                             # nothing to score
            return None
        async with sem:                                        # throttle
            res = await score_one(client, args.model, build_prompt(tpl, args.kind,
                                                                  args.ticker, text))
        return {**{k: v for k, v in rec.items() if k != "text"},   # drop the body from the output
                "enrichment": res} if res else None                # keep only LLM-derived fields

    for src in sorted(Path(args.inp).glob("*.jsonl")):         # file by file
        dst = out_dir / src.name                               # mirrored filename
        if dst.exists():                                       # resumable: skip finished files
            continue
        recs = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        done = [r for r in await asyncio.gather(*(one(r) for r in recs)) if r]
        with dst.open("w", encoding="utf-8") as f:             # write enriched JSONL
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{src.name}: {len(done)}/{len(recs)} enriched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["stock", "macro"], required=True)   # prompt family
    ap.add_argument("--model", required=True)                              # OpenRouter model id
    ap.add_argument("--ticker", default="")                                # stock code (stock only)
    ap.add_argument("--inp", "--in", dest="inp", required=True)            # step-1 output
    ap.add_argument("--out", required=True)                                # per-model output dir
    ap.add_argument("--concurrency", type=int, default=20)                 # parallel requests
    asyncio.run(run(ap.parse_args()))
