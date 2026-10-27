"""
Fitness & Health Knowledge Lake - load the Knowledge Lake graph into Neo4j

Reads the gold tables exported from Databricks (neo4j/nodes.csv, neo4j/edges.csv)
and loads them with the official Python driver (batched UNWIND + MERGE).

Graph model
  (:Study)-[:MENTIONS]->(:Entity)              literature -> extracted entities
  (:Entity)-[:CO_OCCURS_WITH {pmi}]->(:Entity) statistical association
  (:Entity)-[:RISK_FACTOR|SYMPTOM|SUBCLASS_OF|...]->(:Entity|:Concept)   Wikidata
  (:ExerciseVariant)-[:VARIANT_OF]->(:Entity:Exercise)
  (:ExerciseVariant)-[:TARGETS {role}]->(:Entity:Muscle)
  (:Food)-[:CONTAINS {grams}]->(:Entity:Nutrient)
Every Entity also gets a second label from its type (:Exercise, :Muscle, :Condition ...).

Run:
    pip install neo4j pandas
    python load_neo4j.py                    # asks for the password
"""

import getpass
import json
import math
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
DATA = Path("neo4j")
BATCH = 500

TYPE_LABEL = {"EXERCISE": "Exercise", "TRAINING_TYPE": "TrainingType", "MUSCLE": "Muscle",
              "NUTRIENT": "Nutrient", "CONDITION": "Condition", "OUTCOME": "Outcome",
              "POPULATION": "Population", "BIOMARKER": "Biomarker", "BODY_STRUCTURE": "BodyStructure"}


def clean(v):
    """Neo4j properties cannot be null/NaN; drop them."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v in ("NaN", "nan", ""):
        return None
    return v


def node_rows(nodes: pd.DataFrame):
    for r in nodes.itertuples(index=False):
        props = {k: clean(v) for k, v in json.loads(r.props).items()}
        props = {k: v for k, v in props.items() if v is not None}
        props["id"], props["name"] = r.node_id, r.name
        if r.label == "Study" and "keywords" in props:
            props["keywords"] = [k.strip() for k in str(props["keywords"]).split(";") if k.strip()]
        yield r.label, TYPE_LABEL.get(props.get("entity_type")), props


def batched(rows, n=BATCH):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def main():
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")
    pwd = getpass.getpass(f"Neo4j password for {USER}@{URI}: ")
    driver = GraphDatabase.driver(URI, auth=(USER, pwd))
    driver.verify_connectivity()

    with driver.session() as s:
        print("Clearing database...")
        s.run("MATCH (n) DETACH DELETE n").consume()      # small graph: one transaction is fine

        for label in nodes["label"].unique():
            s.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

        # ---- nodes, grouped by (label, type label)
        groups = {}
        for label, tlabel, props in node_rows(nodes):
            groups.setdefault((label, tlabel), []).append(props)
        for (label, tlabel), rows in groups.items():
            labels = f"{label}:{tlabel}" if tlabel else label
            for b in batched(rows):
                s.run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row, n:{labels}",
                      rows=b).consume()
            print(f"  nodes :{labels:24} {len(rows)}")

        # ---- edges, grouped by relationship type
        edges["weight"] = edges["weight"].astype(float)
        label_of = dict(zip(nodes["node_id"], nodes["label"]))   # lets MATCH use the id index
        edges["src_label"] = edges["src"].map(label_of)
        edges["dst_label"] = edges["dst"].map(label_of)
        for (rel, sl, dl), g in edges.groupby(["rel", "src_label", "dst_label"]):
            rows = []
            for r in g.itertuples(index=False):
                props = {"weight": r.weight, "source": r.source}
                if isinstance(r.source, str) and r.source.startswith("pmi="):
                    props["pmi"] = float(r.source[4:])
                    props["source"] = "cooccurrence"
                if rel == "TARGETS":
                    props["role"] = r.source.split(":")[-1]
                rows.append({"src": r.src, "dst": r.dst, "props": props})
            for b in batched(rows):
                s.run(f"""UNWIND $rows AS row
                          MATCH (a:{sl} {{id: row.src}}) MATCH (b:{dl} {{id: row.dst}})
                          MERGE (a)-[r:{rel}]->(b) SET r += row.props""", rows=b).consume()
            print(f"  edges ({sl})-[:{rel}]->({dl}): {len(rows)}")

        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        m = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\nLoaded {n} nodes and {m} relationships.")
    driver.close()


if __name__ == "__main__":
    main()
