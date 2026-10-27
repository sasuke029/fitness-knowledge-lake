"""
Fitness & Health Knowledge Lake - Wikidata entity linking + enrichment (used in Part 1.2c and Part 2.2)

Links each curated entity (squat, creatine, obesity, ...) to a Wikidata item and
pulls external facts about it (description, class, MeSH ID, causes, symptoms,
treatments, part-of ...). These become new nodes/edges in the Neo4j graph.

Two steps
  1. Entity linking   : wbsearchentities API -> candidates -> pick best by
                        type-aware scoring of the description (+ search rank)
  2. Enrichment       : one SPARQL query for all linked QIDs -> selected properties

Run locally:   python wikidata_enrich.py        (reads/writes the curated/ folder)
The Databricks notebook contains the same functions.
"""

import re
import time

import pandas as pd
import requests

WD_API = "https://www.wikidata.org/w/api.php"
WD_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "FitnessKnowledgeLake/1.0 (portfolio project; https://github.com/sasuke029)"}

# Better search terms for ambiguous / plural / abbreviated entity names
SEARCH_ALIAS = {
    "squat": "squat", "hair loss": "hair loss", "heart rate": "heart rate",
    "triglyceride": "triglyceride", "lactate": "lactic acid", "body weight": "human body weight",
    "muscle protein synthesis": "protein biosynthesis", "endurance performance": "endurance training", "deadlift": "deadlift", "lunge": "lunge",
    "plank": "plank", "row": "bent-over row", "sprint": "sprinting",
    "running": "running", "cycling": "cycling", "walking": "walking", "swimming": "swimming",
    "push-up": "push-up", "pull-up": "pull-up", "leg press": "leg press", "hip thrust": "hip thrust",
    "biceps curl": "biceps curl", "leg curl": "leg curl", "leg extension": "leg extension",
    "countermovement jump": "vertical jump", "drop jump": "depth jump",
    "HIIT": "high-intensity interval training", "resistance training": "strength training",
    "aerobic exercise": "aerobic exercise", "plyometric training": "plyometrics",
    "stretching": "stretching", "isometric exercise": "isometric exercise",
    "blood flow restriction training": "blood flow restriction training",
    "sprint interval training": "interval training", "combined training": "circuit training",
    "glutes": "gluteus maximus", "calves": "triceps surae muscle", "shoulders": "deltoid muscle",
    "chest": "pectoralis major", "lats": "latissimus dorsi", "abdominals": "rectus abdominis",
    "lower back": "erector spinae", "biceps": "biceps brachii", "triceps": "triceps brachii",
    "quadriceps": "quadriceps femoris", "hamstrings": "hamstring", "forearms": "forearm",
    "traps": "trapezius", "adductors": "adductor muscles of the hip", "abductors": "hip abductors",
    "protein": "protein in nutrition", "carbohydrate": "carbohydrate", "omega-3": "omega-3 fatty acid",
    "HMB": "beta-Hydroxy beta-methylbutyric acid", "nitrate": "dietary nitrate",
    "essential amino acids": "essential amino acid", "fiber": "dietary fiber",
    "dietary fat": "fat", "whey protein": "whey protein",
    "type 2 diabetes": "type 2 diabetes", "cancer": "cancer", "knee injury": "anterior cruciate ligament injury",
    "COPD": "chronic obstructive pulmonary disease", "frail": "frailty",
    "muscle hypertrophy": "muscle hypertrophy", "muscle strength": "muscle strength", "power": "muscle power",
    "fat loss": "weight loss", "lean mass": "lean body mass", "BMI": "body mass index",
    "cardiorespiratory fitness": "VO2 max", "glycemic control": "glycated hemoglobin",
    "flexibility": "flexibility", "jump performance": "vertical jump",
    "sprint performance": "sprinting",
    "physical function": "physical fitness", "injury risk": "sports injury",
    "bone density": "bone density", "muscle damage": "delayed onset muscle soreness",
    "inflammation": "inflammation", "cholesterol": "cholesterol",
    "older adults": "old age", "athletes": "athlete", "young adults": "young adult",
    "children and adolescents": "adolescence", "women": "woman", "men": "man",
    "trained individuals": "athlete", "untrained individuals": "sedentary lifestyle", "patients": "patient",
    "skeletal muscle": "skeletal muscle", "mtor": "mTOR", "ampk": "AMP-activated protein kinase",
    "physical exercise": "physical exercise", "strengthening": "strength training",
}

# Words that should appear in a correct candidate's description, by our entity type
TYPE_HINTS = {
    "EXERCISE":       ["exercise", "training", "movement", "sport", "jump", "locomotion", "gait",
                       "physical activity", "running", "athletic", "activity"],
    "TRAINING_TYPE":  ["exercise", "training", "physical", "sport", "fitness", "method"],
    "MUSCLE":         ["muscle", "anatomical", "anatomy", "body"],
    "BODY_STRUCTURE": ["muscle", "joint", "anatomical", "anatomy", "organ", "tissue", "tendon",
                       "limb", "bone", "body"],
    "NUTRIENT":       ["chemical", "compound", "acid", "nutrient", "supplement", "protein", "food",
                       "macronutrient", "vitamin", "alkaloid", "molecule", "organic", "dietary"],
    "CONDITION":      ["disease", "disorder", "condition", "syndrome", "medical", "symptom", "injury",
                       "health", "pathology", "pain", "cancer"],
    "OUTCOME":        ["muscle", "measure", "physiolog", "fitness", "health", "body", "index",
                       "blood", "capacity", "strength", "property", "metabolic", "loss", "mass"],
    "POPULATION":     ["person", "human", "people", "age", "stage", "adult", "individual", "athlete",
                       "female", "male", "lifestyle"],
    "BIOMARKER":      ["hormone", "protein", "enzyme", "chemical", "measure", "physiolog", "blood",
                       "kinase", "compound", "rate", "molecule", "body"],
}
BAD_HINTS = ["film", "album", "song", "band", "novel", "television", "episode", "video game",
             "magazine", "journal", "scientific article", "family name", "given name", "company",
             "village", "city", "river", "software", "painting", "wikimedia", "disambiguation",
             "fictional", "spacecraft", "clinical trial", "thesis", "book", "gene in the species Mus",
             "taxon", "species of", "genus of", "musical", "sculpture", "surname"]

# Wikidata properties pulled during enrichment
PROPS = {
    "P31": "instance_of", "P279": "subclass_of", "P361": "part_of", "P527": "has_part",
    "P828": "has_cause", "P1542": "has_effect", "P780": "symptom", "P2176": "drug_or_therapy",
    "P2175": "treats_condition", "P1995": "health_specialty", "P2293": "genetic_association",
    "P2789": "connects_with", "P3094": "develops_from", "P5642": "risk_factor",
}


def _get(url, params, timeout=30, tries=6):
    """GET with retry + exponential backoff (Wikidata rate-limits bursts with HTTP 429)."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 2 ** k))
                time.sleep(min(wait, 60))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** k)
    raise RuntimeError(f"Wikidata request failed after {tries} tries: {params}")


def wd_search(term: str, limit: int = 7) -> list[dict]:
    """Candidates from (1) label/alias search and (2) full-text search, merged."""
    j = _get(WD_API, {"action": "wbsearchentities", "search": term, "language": "en", "uselang": "en",
                      "type": "item", "limit": limit, "format": "json"})
    cands = j.get("search", [])
    seen = {c["id"] for c in cands}
    # full-text search finds items whose label differs from our wording
    j = _get(WD_API, {"action": "query", "list": "search", "srsearch": term, "srlimit": 5,
                      "srnamespace": 0, "format": "json"})
    extra = [h["title"] for h in j.get("query", {}).get("search", []) if h["title"] not in seen]
    if extra:
        j = _get(WD_API, {"action": "wbgetentities", "ids": "|".join(extra), "props": "labels|descriptions",
                          "languages": "en", "format": "json"})
        for q in extra:
            ent = j.get("entities", {}).get(q, {})
            cands.append({"id": q, "label": ent.get("labels", {}).get("en", {}).get("value", ""),
                          "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
                          "match": {"text": ""}, "fulltext": True})
    return cands


def score_candidate(cand: dict, rank: int, name: str, etype: str) -> float:
    desc = (cand.get("description") or "").lower()
    label = (cand.get("label") or "").lower()
    s = 1.0 - 0.1 * rank                                            # search rank prior
    s += 1.0 * sum(h in desc for h in TYPE_HINTS.get(etype, [])) ** 0.5  # type-consistent words
    s -= 3.0 * any(b in desc for b in BAD_HINTS)                    # clearly wrong kinds of item
    matched = (cand.get("match", {}).get("text") or "").lower()
    targets = {name.lower(), SEARCH_ALIAS.get(name, name).lower()}
    if label in targets or matched in targets:
        s += 1.2                                                    # exact label / alias match
    if cand.get("fulltext"):
        s -= 0.3                                                    # full-text hits are less certain
    if not desc:
        s -= 0.5
    return round(s, 3)


def link_entities(entities: pd.DataFrame, min_score: float = 1.0) -> pd.DataFrame:
    """Entity linking: returns entity_id, qid, wd_label, wd_description, link_score, linked."""
    rows = []
    for i, e in enumerate(entities.itertuples()):
        term = SEARCH_ALIAS.get(e.entity_name, e.entity_name)
        try:
            cands = wd_search(term)
        except RuntimeError as ex:
            print(f"  search failed for {term}: {ex}")
            cands = []
        best, best_s = None, -9
        for rank, c in enumerate(cands):
            s = score_candidate(c, rank, e.entity_name, e.entity_type)
            if s > best_s:
                best, best_s = c, s
        ok = best is not None and best_s >= min_score
        rows.append({"entity_id": e.entity_id, "entity_name": e.entity_name, "entity_type": e.entity_type,
                     "search_term": term,
                     "qid": best["id"] if ok else None,
                     "wd_label": best.get("label") if ok else None,
                     "wd_description": best.get("description") if ok else None,
                     "link_score": best_s if best else None, "linked": ok,
                     "n_candidates": len(cands)})
        time.sleep(0.5)
        if (i + 1) % 25 == 0:
            print(f"  linked {i + 1}/{len(entities)}")
    return pd.DataFrame(rows)


def enrich(qids: list[str]) -> pd.DataFrame:
    """One SPARQL query (batched) for selected properties of all linked items."""
    out = []
    for start in range(0, len(qids), 60):
        batch = qids[start:start + 60]
        values = " ".join(f"wd:{q}" for q in batch)
        props = " ".join(f"wdt:{p}" for p in PROPS)
        query = f"""
        SELECT ?item ?prop ?value ?valueLabel ?mesh WHERE {{
          VALUES ?item {{ {values} }}
          VALUES ?prop {{ {props} }}
          ?item ?prop ?value .
          OPTIONAL {{ ?item wdt:P486 ?mesh . }}
          FILTER(isIRI(?value))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""
        res = _get(WD_SPARQL, {"query": query, "format": "json"}, timeout=120)
        for b in res["results"]["bindings"]:
            pid = b["prop"]["value"].rsplit("/", 1)[-1]
            vq = b["value"]["value"].rsplit("/", 1)[-1]
            vlabel = b.get("valueLabel", {}).get("value", vq)
            if re.fullmatch(r"Q\d+", vlabel):      # value has no English label -> skip
                continue
            out.append({"qid": b["item"]["value"].rsplit("/", 1)[-1], "property": pid,
                        "relation": PROPS.get(pid, pid), "value_qid": vq, "value_label": vlabel,
                        "mesh_id": b.get("mesh", {}).get("value")})
        time.sleep(1)
    return pd.DataFrame(out).drop_duplicates(subset=["qid", "property", "value_qid"])


def summarise(links: pd.DataFrame, rels: pd.DataFrame) -> dict:
    by_type = links.groupby("entity_type")["linked"].mean().round(3).to_dict()
    return {"n_entities": len(links), "n_linked": int(links["linked"].sum()),
            "link_rate": round(links["linked"].mean(), 3),
            "avg_link_score": round(links.loc[links["linked"], "link_score"].mean(), 3),
            "n_relations": len(rels), "n_distinct_new_concepts": rels["value_qid"].nunique() if len(rels) else 0,
            "link_rate_by_type": by_type}


if __name__ == "__main__":
    from pathlib import Path
    cur = Path("curated")
    ents = pd.read_csv(cur / "entities.csv")
    print(f"Linking {len(ents)} entities to Wikidata...")
    links = link_entities(ents)
    print("Enriching linked items via SPARQL...")
    rels = enrich(links.loc[links["linked"], "qid"].unique().tolist())
    mesh = rels.dropna(subset=["mesh_id"]).drop_duplicates("qid").set_index("qid")["mesh_id"]
    links["mesh_id"] = links["qid"].map(mesh)
    rels = links[["entity_id", "qid"]].dropna().merge(rels.drop(columns="mesh_id"), on="qid")
    links.to_csv(cur / "entity_wikidata_links.csv", index=False)
    rels.to_csv(cur / "wikidata_relations.csv", index=False)
    s = summarise(links, rels)
    for k, v in s.items():
        print(f"{k:25}: {v}")
    print("\nUnlinked:", ", ".join(links.loc[~links["linked"], "entity_name"]))
    print("\nSample links:")
    print(links.loc[links["linked"], ["entity_name", "qid", "wd_label", "wd_description"]].head(20).to_string())
