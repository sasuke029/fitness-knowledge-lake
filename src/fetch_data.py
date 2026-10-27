"""
Fitness & Health Knowledge Lake
Step 1: Build the raw dataset (3 sources, 3 formats)

  1. Unstructured    -> PubMed research abstracts      (data/abstracts.jsonl + .csv)
  2. Semi-structured -> free-exercise-db exercise JSON  (data/exercises.json)
  3. Structured      -> USDA FoodData Central nutrition (data/foods.csv)

Run:   pip install requests pandas
       python fetch_data.py
Takes about 2-3 minutes.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests

OUT = Path("data")
OUT.mkdir(exist_ok=True)

# NCBI asks API users to identify themselves. Put your own email here.
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")  # set in .env / your shell
NCBI_TOOL = "fitness_knowledge_lake"

# USDA: DEMO_KEY works for a small number of calls per hour.
# For a dedicated key (free, instant): https://fdc.nal.usda.gov/api-key-signup
USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")

# ---------------------------------------------------------------------------
# 1. PubMed abstracts (unstructured text)
# ---------------------------------------------------------------------------
PUBMED_TOPICS = [
    "resistance training hypertrophy",
    "high-intensity interval training",
    "protein supplementation muscle",
    "squat exercise",
    "aerobic exercise obesity",
    "exercise type 2 diabetes",
    "creatine supplementation",
    "stretching flexibility",
    "plyometric training",
    "exercise blood pressure",
]
PER_TOPIC = 40  # 10 topics x 40 = ~400 abstracts

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(topic: str, n: int) -> list[str]:
    # Each word must appear somewhere in title/abstract (not as one exact phrase)
    words = " AND ".join(f"{w}[Title/Abstract]" for w in topic.split())
    term = f"({words}) AND hasabstract AND english[lang] AND 2015:2026[dp]"
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={"db": "pubmed", "term": term, "retmax": n, "sort": "relevance",
                "retmode": "json", "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def text_of(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def pubmed_fetch(pmids: list[str]) -> list[dict]:
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
                "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for art in root.findall(".//PubmedArticle"):
        pmid = text_of(art.find(".//MedlineCitation/PMID"))
        a = art.find(".//MedlineCitation/Article")
        title = text_of(a.find("ArticleTitle"))
        parts = []
        for ab in a.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            txt = text_of(ab)
            parts.append(f"{label}: {txt}" if label else txt)
        year = text_of(a.find("Journal/JournalIssue/PubDate/Year")) or \
               text_of(a.find("Journal/JournalIssue/PubDate/MedlineDate"))[:4]
        journal = text_of(a.find("Journal/Title"))
        mesh = [text_of(m) for m in art.findall(".//MeshHeading/DescriptorName")]
        if parts:
            rows.append({"pmid": pmid, "title": title, "abstract": " ".join(parts),
                         "year": year, "journal": journal, "mesh_terms": mesh})
    return rows


def build_pubmed():
    by_pmid = {}
    for topic in PUBMED_TOPICS:
        ids = pubmed_search(topic, PER_TOPIC)
        time.sleep(0.4)  # stay under NCBI's 3 requests/second limit
        new = [i for i in ids if i not in by_pmid]
        for start in range(0, len(new), 100):
            for row in pubmed_fetch(new[start:start + 100]):
                row["topics"] = [topic]
                by_pmid[row["pmid"]] = row
            time.sleep(0.4)
        for i in ids:  # papers found under more than one topic
            if i in by_pmid and topic not in by_pmid[i]["topics"]:
                by_pmid[i]["topics"].append(topic)
        print(f"  PubMed '{topic}': {len(ids)} ids, total so far {len(by_pmid)}")

    rows = list(by_pmid.values())
    with open(OUT / "abstracts.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    df["mesh_terms"] = df["mesh_terms"].apply("; ".join)
    df["topics"] = df["topics"].apply("; ".join)
    df.to_csv(OUT / "abstracts.csv", index=False)
    print(f"Saved {len(rows)} abstracts")


# ---------------------------------------------------------------------------
# 2. Exercise database (semi-structured JSON)
# ---------------------------------------------------------------------------
def build_exercises():
    url = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    for e in data:          # images are not needed for the Knowledge Lake
        e.pop("images", None)
    with open(OUT / "exercises.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Saved {len(data)} exercises")


# ---------------------------------------------------------------------------
# 3. USDA nutrition table (structured CSV)
# ---------------------------------------------------------------------------
# (search query, words the USDA description MUST contain, words it must NOT contain)
FOODS = [
    ("chicken breast meat only raw", ["chicken", "breast"], ["skin", "fried", "breaded"]),
    ("egg whole raw fresh", ["egg", "whole", "raw"], ["dried", "frozen"]),
    ("salmon atlantic farmed raw", ["salmon", "atlantic"], ["smoked", "cooked"]),
    ("tuna light canned in water", ["tuna", "canned", "water"], []),
    ("beef ground 90% lean raw", ["beef", "ground", "90%"], ["cooked", "patty"]),
    ("yogurt greek plain nonfat", ["yogurt", "greek", "nonfat"], ["fruit", "flavored"]),
    ("milk whole 3.25% milkfat", ["milk", "whole"], ["cheese", "dry", "chocolate"]),
    ("cheese cottage lowfat 2%", ["cottage", "lowfat"], ["vegetables", "fruit"]),
    ("tofu raw firm", ["tofu", "firm"], ["fried", "dried"]),
    ("lentils mature seeds cooked boiled without salt", ["lentils", "cooked"], ["salt,", "with salt"]),
    ("chickpeas mature seeds cooked boiled without salt", ["chickpeas", "cooked"], ["with salt"]),
    ("beans black mature seeds cooked boiled without salt", ["beans", "black", "cooked"], ["with salt"]),
    ("oats rolled dry", ["oats"], ["cooked", "instant"]),
    ("rice brown long-grain cooked", ["rice", "brown", "cooked"], ["fried"]),
    ("rice white long-grain regular cooked", ["rice", "white", "cooked"], ["fried", "glutinous"]),
    ("quinoa cooked", ["quinoa", "cooked"], []),
    ("sweet potato raw unprepared", ["sweet potato", "raw"], ["leaves", "canned"]),
    ("potatoes flesh and skin raw", ["potato", "raw"], ["sweet", "chips", "fried"]),
    ("bananas raw", ["banana", "raw"], ["dehydrated", "chips"]),
    ("apples raw with skin", ["apples", "raw", "with skin"], ["dried"]),
    ("blueberries raw", ["blueberries", "raw"], ["frozen", "canned"]),
    ("broccoli raw", ["broccoli", "raw"], ["chinese", "frozen"]),
    ("spinach raw", ["spinach", "raw"], ["frozen", "canned"]),
    ("avocados raw all commercial varieties", ["avocado", "raw"], []),
    ("nuts almonds", ["almonds"], ["oil", "butter", "paste", "chocolate"]),
    ("peanut butter smooth style without salt", ["peanut butter"], ["reduced", "chunk"]),
    ("oil olive salad or cooking", ["oil", "olive"], []),
    ("bread whole-wheat commercially prepared", ["bread", "whole-wheat"], ["toasted"]),
    ("beverages whey protein powder isolate", ["whey", "protein"], []),
    ("cheese cheddar", ["cheese", "cheddar"], ["reduced", "low", "sauce"]),
]

NUTRIENTS = {  # USDA nutrient id -> column name (values are per 100 g)
    1008: "energy_kcal", 2047: "energy_kcal", 2048: "energy_kcal",  # 2047/2048 = Foundation foods
    1003: "protein_g", 1004: "fat_g", 1005: "carbs_g",
    1079: "fiber_g", 2000: "sugars_g", 1063: "sugars_g", 1093: "sodium_mg",
}


def pick_food(foods, must, must_not):
    for f in foods:
        d = f["description"].lower()
        if all(m.lower() in d for m in must) and not any(x.lower() in d for x in must_not):
            return f
    return None


def build_foods():
    rows = []
    for q, must, must_not in FOODS:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"api_key": USDA_KEY, "query": q, "pageSize": 25,
                    "dataType": "SR Legacy,Foundation"},
            timeout=30,
        )
        if r.status_code == 429:
            print("  USDA rate limit hit - get a free key (see USDA_KEY above) and re-run")
            break
        r.raise_for_status()
        food = pick_food(r.json().get("foods", []), must, must_not)
        if food is None:
            print(f"  no good USDA match for '{q}'")
            continue
        row = {"fdc_id": food["fdcId"], "query": q,
               "description": food["description"],
               "category": food.get("foodCategory", ""),
               "data_type": food.get("dataType", "")}
        for n in food.get("foodNutrients", []):
            col = NUTRIENTS.get(n.get("nutrientId"))
            if col and col not in row:
                row[col] = n.get("value")
        rows.append(row)
        print(f"  {q!r:55} -> {food['description']}")
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(OUT / "foods.csv", index=False)
    print(f"Saved {len(rows)} foods")


if __name__ == "__main__":
    import sys
    # Run everything:            python fetch_data.py
    # Or only some parts:        python fetch_data.py foods pubmed
    parts = sys.argv[1:] or ["exercises", "foods", "pubmed"]
    if "exercises" in parts:
        print("Exercises...")
        build_exercises()
    if "foods" in parts:
        print("Foods...")
        build_foods()
    if "pubmed" in parts:
        print("PubMed abstracts...")
        build_pubmed()
    print("\nDone. Files are in the 'data' folder.")
