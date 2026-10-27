"""
Fitness & Health Knowledge Lake - prepare the three heterogeneous sources for BigQuery

Creates bigquery/*.jsonl|csv to upload to the BigQuery dataset `knowledge_lake`:
  foods.csv            STRUCTURED       flat table (USDA nutrition)
  exercises.jsonl      SEMI-STRUCTURED  nested JSON with arrays (muscles, instructions)
  abstracts.jsonl      UNSTRUCTURED     free text + extracted entities + 384-d text embedding
  entities.jsonl       KNOWLEDGE GRAPH  entity table with Neo4j PageRank + Louvain community
  demo_queries.jsonl   embedded natural-language questions used for semantic search

Embeddings: sentence-transformers all-MiniLM-L6-v2 (already downloaded by KeyBERT in curate.py)
PageRank/community: read from Neo4j (written by neo4j_queries.cypher 2c/2d)

Run:  python prepare_bigquery.py          (Neo4j must be running; asks for the password)
"""

import getpass
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

OUT = Path("bigquery")
OUT.mkdir(exist_ok=True)
CUR, DATA = Path("curated"), Path("data")

DEMO_QUERIES = {
    "q1": "best exercises to build leg muscle strength in older adults",
    "q2": "can high intensity interval training lower blood pressure",
    "q3": "does creatine improve sprint and jump performance in athletes",
    "q4": "how to reduce body fat in people with type 2 diabetes",
}


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(rows)} rows)")


def graph_scores():
    """PageRank + community per entity from Neo4j (falls back to empty if Neo4j is off)."""
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver("neo4j://127.0.0.1:7687",
                                   auth=("neo4j", getpass.getpass("Neo4j password: ")))
        with drv.session() as s:
            rows = s.run("MATCH (e:Entity) RETURN e.id AS id, e.pagerank AS pagerank, "
                         "e.community AS community").data()
        drv.close()
        return {r["id"]: r for r in rows}
    except Exception as ex:
        print("  Neo4j not reachable, PageRank left empty:", type(ex).__name__)
        return {}


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- STRUCTURED
    foods = pd.read_csv(CUR / "foods_clean.csv")
    foods.to_csv(OUT / "foods.csv", index=False)
    print(f"  {OUT / 'foods.csv'}  ({len(foods)} rows)")

    # ---- SEMI-STRUCTURED: keep the nested JSON shape (arrays) on purpose
    ex = json.load(open(DATA / "exercises.json", encoding="utf-8"))
    write_jsonl([{"id": e["id"], "name": e["name"], "category": e.get("category"),
                  "level": e.get("level"), "equipment": e.get("equipment") or "none",
                  "mechanic": e.get("mechanic"), "force": e.get("force"),
                  "primary_muscles": e.get("primaryMuscles", []),
                  "secondary_muscles": e.get("secondaryMuscles", []),
                  "instructions": e.get("instructions", [])} for e in ex], OUT / "exercises.jsonl")

    # ---- KNOWLEDGE GRAPH entities
    ents = pd.read_csv(CUR / "entities.csv")
    links = pd.read_csv(CUR / "entity_wikidata_links.csv")[["entity_id", "qid", "wd_description"]]
    ents = ents.merge(links, on="entity_id", how="left")
    gs = graph_scores()
    rows = []
    for r in ents.itertuples():
        g = gs.get(r.entity_id, {})
        rows.append({"entity_id": r.entity_id, "name": r.entity_name, "type": r.entity_type,
                     "n_studies": int(r.n_abstracts),
                     "pagerank": g.get("pagerank"), "community": g.get("community"),
                     "qid": r.qid if isinstance(r.qid, str) else None,
                     "wd_description": r.wd_description if isinstance(r.wd_description, str) else None})
    write_jsonl(rows, OUT / "entities.jsonl")

    # ---- UNSTRUCTURED abstracts + entities + embeddings
    a = pd.read_csv(CUR / "abstracts_clean.csv", dtype={"pmid": str})
    ae = pd.read_csv(CUR / "abstract_entities.csv", dtype={"pmid": str})
    ae = ae.merge(ents[["entity_id", "entity_name"]], on="entity_id")
    ent_by_pmid = ae.groupby("pmid")["entity_name"].apply(lambda s: sorted(set(s))).to_dict()
    texts = (a["title"].fillna("") + ". " + a["abstract"].fillna("")).tolist()
    print("  embedding abstracts...")
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    write_jsonl([{"pmid": r.pmid, "title": r.title, "year": int(r.year) if pd.notna(r.year) else None,
                  "journal": r.journal, "abstract": r.abstract, "conclusion": r.conclusion,
                  "finding": r.finding,
                  "keywords": [k for k in str(r.keywords).split("; ") if k and k != "nan"],
                  "entities": ent_by_pmid.get(r.pmid, []),
                  "embedding": [round(float(x), 6) for x in emb[i]]}
                 for i, r in enumerate(a.itertuples())], OUT / "abstracts.jsonl")

    # ---- demo questions (embedded with the same model)
    q_emb = model.encode(list(DEMO_QUERIES.values()), normalize_embeddings=True)
    write_jsonl([{"query_id": k, "query_text": v, "embedding": [round(float(x), 6) for x in q_emb[i]]}
                 for i, (k, v) in enumerate(DEMO_QUERIES.items())], OUT / "demo_queries.jsonl")
    print("\nDone. Upload the 5 files in bigquery/ to the dataset 'knowledge_lake'.")


if __name__ == "__main__":
    main()
