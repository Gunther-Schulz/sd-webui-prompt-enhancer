"""V3-specific steps: retrieval-first with LLM as curator.

Pipeline shape (differs from V1):
    prose                   ← same as V1 (Detailed base + shortlist frag)
    concept_enumerate       ← new: LLM lists distinct scene concepts as
                              free-form phrases (no tag-shaped output)
    canonicalize_concepts   ← new: each concept phrase → embedder +
                              retriever → nearest real DB tag (no
                              compound-split heuristic; no invention)
    curate                  ← new: LLM sees (prose + candidate pool)
                              and picks + can drop off-theme candidates
                              (can't invent; removals only)
    rule_layer              ← reuse extension's rule layer (format)
    slot_fill               ← same as V1

The key structural difference: the LLM emits concept *phrases*, not tag
tokens. The embedder/retriever canonicalizes each phrase to a real DB
tag. Invention is impossible because every tag in the output traces to
a DB record.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from experiments.steps.common import call_llm, get_stack, pe


# ── Step: concept_enumerate ──────────────────────────────────────────


_ENUMERATE_SP = """\
You are a scene analyst. Given a prose description of an image, list
the distinct *concepts* an image-generation model would need to know
about — subject(s), pose, clothing (each layer), expression, lighting,
setting, props, mood, era, style, any explicit or adult content.

Output format:
- One concept per line.
- Use short NOUN PHRASES (2-4 words typically). Not sentences.
- Do NOT use underscores or booru-tag formatting.
- Do NOT invent concepts that aren't in the prose.
- Cover every distinct thing the prose describes — favor over-coverage
  over omission.
- Preserve the prose's explicit/adult content literally. Do not soften.

Examples (valid):
  young woman
  long silver hair
  golden eyes
  ornate jewelry
  sun-drenched meadow
  contemplative expression

Examples (invalid — do not output):
  "1girl"           (booru tag — that's the next stage's job)
  "a sense of calm"  (not a concrete concept)
  "She wears a dress" (full sentence)
"""


def concept_enumerate(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: prose (str), seed (int).
    Write: concepts (list[str]).

    LLM pass that extracts distinct concept phrases from the prose.
    Each phrase becomes input to canonicalize_concepts.
    """
    prose = state["prose"]
    seed = state.get("seed", -1)
    raw = call_llm(
        _ENUMERATE_SP,
        f"PROSE:\n{prose}\n\nList concepts, one per line.",
        seed=seed,
        temperature=params.get("temperature", 0.6),
        num_predict=params.get("num_predict", 512),
        model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
    )
    concepts: List[str] = []
    for line in raw.splitlines():
        s = line.strip().lstrip("-•*").strip()
        # Drop numbering (1. 2. etc) and empty lines
        while s and s[0].isdigit():
            s = s[1:]
        s = s.lstrip(".) ").strip().strip("\"'")
        if s and len(s) <= 80:
            concepts.append(s)
    if not concepts:
        raise ValueError(
            f"concept_enumerate got empty list from LLM. raw: {raw[:300]!r}"
        )
    return {**state, "concepts": concepts, "concepts_raw": raw}


# ── Step: canonicalize_concepts ──────────────────────────────────────


def canonicalize_concepts(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: concepts (list[str]), shortlist.
    Write: candidates (list[dict]) — one dict per concept with
           concept, top_k_tags, best_tag, best_score, dropped_reason.

    For each concept phrase, retrieve top-K tags from the DB via the
    embedder+FAISS+rerank. A concept is dropped if even the top result
    scores below `min_score` (embedder cosine) OR below `min_post_count`.
    """
    concepts = state["concepts"]
    stack = get_stack()
    min_score = params.get("min_score", 0.35)
    min_post = params.get("min_post_count", 50)
    top_k = params.get("top_k", 5)

    # Bulk embed all concepts in one pass (fast)
    vecs = stack.embedder.encode(concepts)
    # Retrieve top-K per concept
    scores_mat, ids_mat = stack.index.search(vecs, top_k)

    results: List[Dict[str, Any]] = []
    for concept, row_scores, row_ids in zip(concepts, scores_mat, ids_mat):
        hits = []
        for score, tid in zip(row_scores, row_ids):
            if tid < 0:
                continue
            rec = stack.db.get_by_id(int(tid))
            if not rec:
                continue
            hits.append({
                "name": rec["name"],
                "category": rec["category"],
                "post_count": rec["post_count"],
                "score": float(score),
            })
        # Pick the best — top score that clears both floors
        best = None
        for h in hits:
            if h["score"] >= min_score and h["post_count"] >= min_post:
                best = h
                break
        results.append({
            "concept": concept,
            "top_k": hits,
            "best": best,
            "dropped_reason": None if best else (
                "no_candidate_above_floors"
                if hits else "no_retrieval_hit"
            ),
        })
    return {**state, "candidates": results}


# ── Step: curate ─────────────────────────────────────────────────────


_CURATE_SP = """\
You are a tag curator for an anime image-generation model. You will see:
  (1) a prose description of the intended scene
  (2) a pool of candidate tags, grouped by which concept each came from

Your job: OUTPUT a comma-separated list of the tags you judge best for
this scene. Pick from the candidate pool ONLY — do NOT invent tags not
in the pool.

Rules:
- Output one comma-separated line. Nothing else.
- Drop any candidate that's off-theme for the scene (e.g. if the pool
  has "airplane" but the prose is a speakeasy, drop it).
- Prefer one strong tag per concept. You can drop a concept entirely
  if no candidate fits.
- If the prose has explicit/adult content, KEEP the explicit candidate
  tags. Do not sanitize.
- Use spaces in tag names (not underscores).
"""


def curate(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: prose (str), candidates (list[dict]), seed (int).
    Write: draft (str) — comma-separated tag string.

    LLM sees the prose + candidate pool and picks. Output shape matches
    what `pe._anima_tag_from_draft` expects so downstream validate +
    rule_layer work unchanged.
    """
    prose = state["prose"]
    candidates = state["candidates"]

    # Build pool string — per-concept groups so the LLM sees structure
    pool_lines = []
    for c in candidates:
        if not c["top_k"]:
            continue
        names = [h["name"].replace("_", " ") for h in c["top_k"]]
        pool_lines.append(f"[{c['concept']}] → {', '.join(names)}")
    pool_str = "\n".join(pool_lines)
    if not pool_str:
        raise ValueError("curate got empty candidate pool")

    user_msg = (
        f"PROSE:\n{prose}\n\n"
        f"CANDIDATE POOL (tag → concept it came from):\n{pool_str}\n\n"
        f"Output the curated comma-separated tag list."
    )

    draft = call_llm(
        _CURATE_SP,
        user_msg,
        seed=state.get("seed", -1),
        temperature=params.get("temperature", 0.5),
        num_predict=params.get("num_predict", 512),
        model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
    )
    return {**state, "draft": draft}
