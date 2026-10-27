"""
Fitness & Health Knowledge Lake - Generative AI on the Knowledge Lake graph

Three small LLM tasks on our own Neo4j data, each with a measurable result:

  Task 1  Graph-RAG entity summaries
          Neo4j sub-graph (studies, outcomes, % positive, co-occurring concepts,
          Wikidata facts) -> LLM writes an evidence summary citing PMIDs.
          Baseline: same LLM, same question, NO graph context.
          Metric: share of cited PMIDs that are verifiable in the Knowledge Lake.

  Task 2  Finding classification (re-do of the Part 1.1 zero-shot step)
          LLM labels 40 study conclusions: positive / no_effect / inconclusive / negative.
          Metric: accuracy + macro-F1 vs a manually labelled sample, compared with BART.

  Task 3  Knowledge-graph completion
          Entity pairs that co-occur strongly (PMI) but have no explicit relation
          -> LLM reads the shared conclusions and proposes a typed relation
          (IMPROVES, REDUCES, NO_EFFECT_ON, ...) + confidence + evidence PMIDs.
          Written back to Neo4j as (a)-[:PROPOSED {status:'pending_review'}]->(b).

LLM: Google Gemini (free API key from https://aistudio.google.com/apikey)
     Only Task 3 again (e.g. after a quota error):  python genai_tasks.py --only3
     or a local Ollama model:  python genai_tasks.py --ollama llama3.1

Run:  pip install google-genai neo4j pandas
      python genai_tasks.py
"""

import getpass
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from neo4j import GraphDatabase

OUT = Path("genai")
OUT.mkdir(exist_ok=True)
NEO4J_URI, NEO4J_USER = "neo4j://127.0.0.1:7687", "neo4j"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")      # empty = pick the newest available Flash model
USE_OLLAMA = "--ollama" in sys.argv
OLLAMA_MODEL = sys.argv[sys.argv.index("--ollama") + 1] if USE_OLLAMA and len(sys.argv) > sys.argv.index("--ollama") + 1 else "llama3.1"
MODEL_NAME = f"ollama:{OLLAMA_MODEL}" if USE_OLLAMA else GEMINI_MODEL

RAG_ENTITIES = ["creatine", "sarcopenia", "HIIT", "type 2 diabetes", "squat"]
LABELS = ["positive", "no_effect", "inconclusive", "negative"]
RELATIONS = ["IMPROVES", "REDUCES", "NO_EFFECT_ON", "WORSENS", "ASSOCIATED_WITH", "NONE"]


# =============================================================================
# LLM wrapper (Gemini or Ollama), with retry for free-tier rate limits
# =============================================================================
_client = None


def llm(prompt: str, as_json: bool = False, temperature: float = 0.2) -> str:
    global _client
    for attempt in range(6):
        try:
            if USE_OLLAMA:
                r = requests.post("http://localhost:11434/api/generate", timeout=600, json={
                    "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                    "options": {"temperature": temperature}, **({"format": "json"} if as_json else {})})
                r.raise_for_status()
                return r.json()["response"]
            from google import genai
            from google.genai import types
            if _client is None:
                key = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Gemini API key: ")
                _client = genai.Client(api_key=key)
                pick_gemini_model(_client)
            resp = _client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json" if as_json else "text/plain"))
            time.sleep(4)                      # stay under free-tier requests/minute
            return resp.text
        except RuntimeError:
            raise
        except Exception as ex:                # 429 / transient errors -> back off
            wait = 10 * (attempt + 1)
            print(f"    LLM error ({type(ex).__name__}: {str(ex)[:80]}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM failed after retries")


def pick_gemini_model(client):
    """Choose the newest text 'flash' model this API key can call (model names change often)."""
    global GEMINI_MODEL, MODEL_NAME
    if GEMINI_MODEL:
        return
    names = []
    for m in client.models.list():
        name = m.name.split("/")[-1]
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        if "flash" in name and not any(x in name for x in
                                       ("lite", "image", "tts", "live", "audio", "thinking", "exp", "8b")):
            names.append(name)
    if not names:
        raise RuntimeError("No Gemini flash model available for this key - set GEMINI_MODEL manually")

    def version(n):
        nums = re.findall(r"\d+(?:\.\d+)?", n)
        return (float(nums[0]) if nums else 0, "preview" not in n, n)
    GEMINI_MODEL = sorted(names, key=version)[-1]
    MODEL_NAME = GEMINI_MODEL
    print(f"  using Gemini model: {GEMINI_MODEL}   (available flash models: {', '.join(sorted(names))})")


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    obj = json.loads(text)
    if isinstance(obj, dict):          # e.g. {"results": [...]} -> the list inside
        obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
    return obj


# =============================================================================
# TASK 1 - Graph-RAG summaries vs no-context baseline
# =============================================================================
CONTEXT_QUERIES = {
    "entity": """MATCH (e:Entity {name: $name})
                 RETURN e.entity_type AS type, e.n_abstracts AS studies, e.pagerank AS pagerank,
                        e.community AS community, e.wd_description AS wikidata_description""",
    "outcomes": """MATCH (e:Entity {name: $name})<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(o:Entity)
                   WHERE o:Outcome OR o:Condition
                   RETURN o.name AS concept, count(s) AS studies,
                          sum(CASE s.finding WHEN 'positive' THEN 1 ELSE 0 END) AS positive_findings
                   ORDER BY studies DESC LIMIT 8""",
    "related": """MATCH (e:Entity {name: $name})-[r:CO_OCCURS_WITH]-(x:Entity)
                  RETURN x.name AS concept, x.entity_type AS type, r.weight AS shared_studies, r.pmi AS pmi
                  ORDER BY r.pmi * r.weight DESC LIMIT 8""",
    "wikidata": """MATCH (e:Entity {name: $name})-[r]->(x)
                   WHERE r.source = 'wikidata'
                   RETURN type(r) AS relation, x.name AS value LIMIT 12""",
    "muscles": """MATCH (e:Entity {name: $name})<-[:VARIANT_OF]-(v:ExerciseVariant)-[:TARGETS {role:'primary'}]->(m)
                  RETURN m.name AS muscle, count(v) AS variants ORDER BY variants DESC LIMIT 6""",
    "studies": """MATCH (e:Entity {name: $name})<-[:MENTIONS]-(s:Study)
                  RETURN s.pmid AS pmid, s.year AS year, s.finding AS finding, s.name AS title
                  ORDER BY s.year DESC LIMIT 8""",
}


def graph_context(session, name: str) -> dict:
    ctx = {k: session.run(q, name=name).data() for k, q in CONTEXT_QUERIES.items()}
    ctx["entity"] = ctx["entity"][0] if ctx["entity"] else {}
    return ctx


def add_conclusions(ctx, abstracts):
    concl = abstracts.set_index("pmid")["conclusion"].to_dict()
    for s in ctx["studies"]:
        s["conclusion"] = str(concl.get(str(s["pmid"]), ""))[:450]
    return ctx


RAG_PROMPT = """You are a sports-science research assistant. Using ONLY the knowledge-graph context
below, write an evidence summary (150-200 words) about "{name}" for a fitness professional.
Rules: cite studies inline as [PMID:12345678] using only PMIDs that appear in the context;
say how consistent the evidence is (use the positive_findings counts); mention one limitation;
do not add facts that are not in the context.

KNOWLEDGE-GRAPH CONTEXT (JSON):
{context}"""

BASELINE_PROMPT = """You are a sports-science research assistant. Write an evidence summary (150-200 words)
about "{name}" for a fitness professional. Cite supporting studies inline as [PMID:12345678]."""


def cited_pmids(text):
    return re.findall(r"PMID:\s*(\d{6,9})", text)


def task1(session, abstracts):
    print("\nTASK 1  Graph-RAG summaries")
    corpus = set(abstracts["pmid"].astype(str))
    rows, md = [], ["# Task 1 - Graph-RAG entity summaries vs no-context baseline\n"]
    for name in RAG_ENTITIES:
        ctx = add_conclusions(graph_context(session, name), abstracts)
        ctx_pmids = {str(s["pmid"]) for s in ctx["studies"]}
        rag = llm(RAG_PROMPT.format(name=name, context=json.dumps(ctx, indent=1, default=str)))
        base = llm(BASELINE_PROMPT.format(name=name))
        for method, text, allowed in [("graph_rag", rag, ctx_pmids), ("no_context", base, corpus)]:
            cites = cited_pmids(text)
            ok = [c for c in cites if c in allowed]
            rows.append({"entity": name, "method": method, "n_citations": len(cites),
                         "verifiable_citations": len(ok),
                         "grounding_rate": round(len(ok) / len(cites), 2) if cites else None,
                         "words": len(text.split())})
        md += [f"## {name}\n", "**Graph context used:** " +
               f"{len(ctx['studies'])} studies, {len(ctx['outcomes'])} outcomes/conditions, "
               f"{len(ctx['related'])} co-occurring concepts, {len(ctx['wikidata'])} Wikidata facts\n",
               "### With knowledge-graph context (Graph-RAG)\n", rag.strip() + "\n",
               "### Without context (baseline)\n", base.strip() + "\n"]
        print(f"  {name}: done")
    (OUT / "task1_rag_summaries.md").write_text("\n".join(md), encoding="utf-8")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "task1_grounding.csv", index=False)
    return res


# =============================================================================
# TASK 2 - finding classification vs BART zero-shot, against manual labels
# =============================================================================
CLS_PROMPT = """Classify the main finding of each study conclusion into exactly one label:
- positive: the intervention/exposure produced a benefit or improvement
- no_effect: no significant difference / no effect / similar results between conditions
- inconclusive: evidence limited, mixed or conflicting, OR the text only describes/reviews without an outcome claim
- negative: harm, increased risk, or a worse outcome

Return JSON: a list of objects {{"i": <index>, "label": <label>, "reason": <max 12 words>}}.

CONCLUSIONS:
{items}"""


def macro_f1(y_true, y_pred):
    f1s = []
    for c in LABELS:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return round(sum(f1s) / len(f1s), 3)


def task2():
    print("\nTASK 2  Finding classification")
    gold = pd.read_csv(OUT / "gold_findings.csv", dtype={"pmid": str})
    preds = {}
    for start in range(0, len(gold), 10):
        chunk = gold.iloc[start:start + 10]
        items = "\n".join(f"[{i}] {c}" for i, c in zip(chunk.index, chunk["conclusion"]))
        for obj in parse_json(llm(CLS_PROMPT.format(items=items), as_json=True)):
            preds[int(obj["i"])] = (obj["label"].strip().lower(), obj.get("reason", ""))
        print(f"  classified {min(start + 10, len(gold))}/{len(gold)}")
    gold["llm_label"] = [preds.get(i, ("missing", ""))[0] for i in gold.index]
    gold["llm_reason"] = [preds.get(i, ("", ""))[1] for i in gold.index]
    gold.to_csv(OUT / "task2_classification.csv", index=False)
    metrics = pd.DataFrame([
        {"method": "BART-large-MNLI zero-shot (Part 1.1)",
         "accuracy": round((gold.gold == gold.bart_label).mean(), 3),
         "macro_f1": macro_f1(gold.gold, gold.bart_label)},
        {"method": f"LLM {MODEL_NAME}",
         "accuracy": round((gold.gold == gold.llm_label).mean(), 3),
         "macro_f1": macro_f1(gold.gold, gold.llm_label)},
    ])
    metrics.to_csv(OUT / "task2_metrics.csv", index=False)
    confusion = pd.crosstab(gold.gold, gold.llm_label, rownames=["gold"], colnames=["llm"])
    confusion.to_csv(OUT / "task2_confusion.csv")
    return metrics, confusion


# =============================================================================
# TASK 3 - knowledge-graph completion: typed relations for co-occurring pairs
# =============================================================================
PAIR_QUERY = """
MATCH (a:Entity)-[c:CO_OCCURS_WITH]-(b:Entity)
WHERE a.entity_type IN ['TRAINING_TYPE', 'EXERCISE', 'NUTRIENT']
  AND b.entity_type IN ['OUTCOME', 'CONDITION']
  AND c.weight >= 4 AND c.pmi > 0.5
  AND NOT (a)-[:RISK_FACTOR|SYMPTOM|SUBCLASS_OF|HAS_PART|PART_OF|INSTANCE_OF|HAS_CAUSE|PROPOSED]-(b)
WITH a, b, c ORDER BY c.weight * c.pmi DESC LIMIT 12
MATCH (a)<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(b)
WITH a, b, c, collect(s.pmid)[0..4] AS pmids
RETURN a.id AS a_id, a.name AS a, b.id AS b_id, b.name AS b, c.weight AS shared, c.pmi AS pmi, pmids
"""

KG_PROMPT = """You extend a fitness & health knowledge graph. For each entity pair, read the study
conclusions and decide the relation from SUBJECT to OBJECT, choosing one of:
IMPROVES, REDUCES, NO_EFFECT_ON, WORSENS, ASSOCIATED_WITH, NONE.
(REDUCES = lowers something, e.g. lowers blood pressure; IMPROVES = makes an outcome better.)
Base the answer only on the conclusions. Return JSON: a list of objects
{{"pair": <index>, "relation": <relation>, "confidence": <0-1>, "evidence_pmids": [..], "rationale": <max 20 words>}}.

PAIRS:
{items}"""


def task3(session, abstracts):
    print("\nTASK 3  Knowledge-graph completion")
    concl = abstracts.set_index("pmid")["conclusion"].to_dict()
    pairs = session.run(PAIR_QUERY).data()
    out = []
    for start in range(0, len(pairs), 4):
        chunk = pairs[start:start + 4]
        items = []
        for i, p in enumerate(chunk, start):
            ev = "\n".join(f"   PMID {pm}: {str(concl.get(str(pm), ''))[:350]}" for pm in p["pmids"])
            items.append(f"[{i}] SUBJECT={p['a']}  OBJECT={p['b']}\n{ev}")
        for obj in parse_json(llm(KG_PROMPT.format(items="\n\n".join(items)), as_json=True)):
            p = pairs[int(obj["pair"])]
            rel = str(obj["relation"]).upper()
            out.append({"subject": p["a"], "relation": rel if rel in RELATIONS else "NONE", "object": p["b"],
                        "confidence": float(obj.get("confidence", 0)), "shared_studies": p["shared"],
                        "pmi": p["pmi"], "evidence_pmids": ";".join(map(str, obj.get("evidence_pmids", []))),
                        "rationale": obj.get("rationale", ""), "a_id": p["a_id"], "b_id": p["b_id"]})
        print(f"  pairs {min(start + 4, len(pairs))}/{len(pairs)}")
    res = pd.DataFrame(out)
    keep = res[(res.relation != "NONE") & (res.confidence >= 0.6)]
    session.run("""UNWIND $rows AS r
                   MATCH (a:Entity {id: r.a_id}), (b:Entity {id: r.b_id})
                   MERGE (a)-[p:PROPOSED {relation: r.relation}]->(b)
                   SET p.confidence = r.confidence, p.evidence = r.evidence_pmids, p.rationale = r.rationale,
                       p.model = $model, p.status = 'pending_review', p.created = datetime()""",
                rows=keep.to_dict("records"), model=MODEL_NAME).consume()
    res.drop(columns=["a_id", "b_id"]).to_csv(OUT / "task3_proposed_relations.csv", index=False)
    return res, len(keep)


# =============================================================================
def main():
    abstracts = pd.read_csv("curated/abstracts_clean.csv", dtype={"pmid": str})
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, getpass.getpass("Neo4j password: ")))
    print(f"LLM: {MODEL_NAME or 'newest Gemini flash (auto)'}")
    only3 = "--only3" in sys.argv          # re-run just Task 3, reuse saved Task 1/2 results
    with driver.session() as s:
        if only3:
            t1 = pd.read_csv(OUT / "task1_grounding.csv")
            t2 = pd.read_csv(OUT / "task2_metrics.csv")
            confusion = pd.read_csv(OUT / "task2_confusion.csv", index_col=0)
        else:
            t1 = task1(s, abstracts)
            t2, confusion = task2()
        t3, n_written = task3(s, abstracts)
    driver.close()

    summary = t1.groupby("method")[["n_citations", "verifiable_citations"]].sum()
    summary["grounding_rate"] = (summary.verifiable_citations / summary.n_citations).round(2)
    report = ["GENAI RESULTS  (model: " + MODEL_NAME + ")", "=" * 60,
              "\nTask 1 - citation grounding (all 5 entities)", summary.to_string(),
              "\nPer entity:", t1.to_string(index=False),
              "\nTask 2 - finding classification on 40 manually labelled conclusions", t2.to_string(index=False),
              "\nLLM confusion matrix:", confusion.to_string(),
              f"\nTask 3 - KG completion: {len(t3)} pairs assessed, {n_written} PROPOSED edges written to Neo4j",
              t3[["subject", "relation", "object", "confidence", "shared_studies", "evidence_pmids"]].to_string(index=False)]
    text = "\n".join(report)
    (OUT / "genai_results.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nAll outputs saved in '{OUT}/'")


if __name__ == "__main__":
    main()
