"""
Fitness & Health Knowledge Lake
Step 2 (Part 1.1c): Data Curation - Ingest -> Extract -> Store

Runs the same logic as the Snowflake pipeline design (snowflake_pipeline.sql)
locally in Python, so the curated tables are real and feed every later step.

Extraction uses three generations of techniques (matches Part 1.1a):
  1. Rule-based      : dictionary (gazetteer) matching with synonyms
  2. ML / embeddings : KeyBERT keywords (sentence embeddings)
  3. Transformers    : biomedical NER model + zero-shot finding classification

Run (first time downloads ~2 GB of models, then cached):
    pip install pandas transformers torch keybert
    python curate.py            # full run, ~10-15 min on CPU
    python curate.py --fast     # dictionary only, seconds (for testing)
    python curate.py --refine   # re-use existing curated/ output: re-classify findings
                                # (4 labels) + clean transformer entities, ~5 min

Inputs : data/abstracts.jsonl, data/exercises.json, data/foods.csv
Outputs: curated/*.csv  +  curated/extraction_summary.txt
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUT = Path("curated")
OUT.mkdir(exist_ok=True)
FAST = "--fast" in sys.argv
REFINE = "--refine" in sys.argv

# Zero-shot labels for the finding of each conclusion. Four classes, because many
# reviews end with "evidence is limited" - that is inconclusive, not harmful.
FINDING_LABELS = {
    "the results show a clear benefit or improvement": "positive",
    "the results show no significant difference or no effect": "no_effect",
    "the evidence is limited, mixed or inconclusive": "inconclusive",
    "the results show harm or a worse outcome": "negative",
}

NER_MODEL = "d4data/biomedical-ner-all"          # ~260 MB, DistilBERT trained on clinical case reports
ZS_MODEL = "facebook/bart-large-mnli"             # ~1.6 GB, zero-shot classifier
KW_MODEL = "all-MiniLM-L6-v2"                     # ~90 MB, sentence embeddings for KeyBERT

# ===========================================================================
# 1. RULE-BASED: domain dictionary (canonical name -> synonyms / variants)
#    Matching is case-insensitive with word boundaries.
# ===========================================================================
VOCAB = {
    "TRAINING_TYPE": {
        "resistance training": ["resistance training", "resistance exercise", "strength training",
                                "weight training", "weightlifting", "resistance-trained"],
        "HIIT": ["high-intensity interval training", "high intensity interval training", "HIIT",
                 "high-intensity interval exercise", "interval training"],
        "aerobic exercise": ["aerobic exercise", "aerobic training", "endurance training",
                             "endurance exercise", "cardiovascular exercise", "moderate-intensity continuous training", "MICT"],
        "plyometric training": ["plyometric training", "plyometrics", "plyometric exercise", "jump training"],
        "stretching": ["stretching", "static stretching", "dynamic stretching", "PNF stretching",
                       "proprioceptive neuromuscular facilitation"],
        "sprint interval training": ["sprint interval training", "SIT", "repeated sprint training"],
        "yoga": ["yoga"],
        "pilates": ["pilates"],
        "isometric exercise": ["isometric exercise", "isometric training", "isometric handgrip"],
        "blood flow restriction training": ["blood flow restriction", "BFR training", "BFR"],
        "combined training": ["combined training", "concurrent training", "combined aerobic and resistance"],
    },
    "EXERCISE": {
        "squat": ["squat", "squats", "back squat", "front squat", "squatting"],
        "deadlift": ["deadlift", "deadlifts", "romanian deadlift"],
        "bench press": ["bench press"],
        "leg press": ["leg press"],
        "lunge": ["lunge", "lunges"],
        "hip thrust": ["hip thrust", "hip thrusts"],
        "push-up": ["push-up", "push-ups", "push up"],
        "pull-up": ["pull-up", "pull-ups", "chin-up"],
        "leg extension": ["leg extension", "knee extension exercise"],
        "leg curl": ["leg curl", "hamstring curl", "nordic hamstring", "nordic curl"],
        "biceps curl": ["biceps curl", "bicep curl", "arm curl"],
        "row": ["bent-over row", "seated row", "barbell row"],
        "plank": ["plank"],
        "countermovement jump": ["countermovement jump", "CMJ"],
        "drop jump": ["drop jump", "depth jump"],
        "sprint": ["sprint", "sprints", "sprinting"],
        "running": ["running", "treadmill running", "jogging"],
        "cycling": ["cycling", "cycle ergometer", "stationary bike"],
        "walking": ["walking", "brisk walking"],
        "swimming": ["swimming"],
        "rowing": ["rowing ergometer"],
    },
    "MUSCLE": {
        "quadriceps": ["quadriceps", "quadricep", "quads", "vastus lateralis", "vastus medialis", "vastus",
                       "rectus femoris", "knee extensors"],
        "hamstrings": ["hamstrings", "hamstring", "biceps femoris", "knee flexors"],
        "glutes": ["glutes", "gluteus maximus", "gluteus medius", "gluteal"],
        "calves": ["calves", "calf", "gastrocnemius", "soleus", "triceps surae"],
        "chest": ["pectoralis major", "pectoralis", "chest muscles"],
        "shoulders": ["deltoid", "deltoids", "shoulder muscles"],
        "biceps": ["biceps brachii", "elbow flexors"],
        "triceps": ["triceps brachii", "elbow extensors"],
        "lats": ["latissimus dorsi", "lats"],
        "abdominals": ["abdominals", "rectus abdominis", "abdominal muscles", "core muscles", "obliques"],
        "lower back": ["erector spinae", "lower back muscles", "lumbar extensors"],
        "adductors": ["adductors", "hip adductors"],
        "abductors": ["abductors", "hip abductors"],
        "forearms": ["forearm", "forearms"],
        "traps": ["trapezius"],
    },
    "NUTRIENT": {
        "protein": ["protein supplementation", "protein intake", "dietary protein", "protein supplement",
                    "protein ingestion", "protein"],
        "whey protein": ["whey protein", "whey"],
        "casein": ["casein"],
        "creatine": ["creatine monohydrate", "creatine supplementation", "creatine"],
        "leucine": ["leucine"],
        "essential amino acids": ["essential amino acids", "EAA", "amino acids"],
        "carbohydrate": ["carbohydrate", "carbohydrates"],
        "caffeine": ["caffeine"],
        "beta-alanine": ["beta-alanine", "β-alanine"],
        "vitamin D": ["vitamin D"],
        "omega-3": ["omega-3", "fish oil", "n-3 fatty acids"],
        "sodium bicarbonate": ["sodium bicarbonate"],
        "nitrate": ["nitrate", "beetroot juice"],
        "HMB": ["HMB", "β-hydroxy-β-methylbutyrate", "beta-hydroxy-beta-methylbutyrate"],
        "dietary fat": ["dietary fat"],
        "fiber": ["dietary fiber", "dietary fibre"],
    },
    "CONDITION": {
        "obesity": ["obesity", "obese"],
        "overweight": ["overweight"],
        "type 2 diabetes": ["type 2 diabetes", "type 2 diabetes mellitus", "T2DM", "T2D", "diabetes"],
        "prediabetes": ["prediabetes", "impaired glucose tolerance"],
        "hypertension": ["hypertension", "hypertensive", "high blood pressure"],
        "sarcopenia": ["sarcopenia", "sarcopenic"],
        "metabolic syndrome": ["metabolic syndrome"],
        "cardiovascular disease": ["cardiovascular disease", "coronary artery disease", "heart failure"],
        "osteoarthritis": ["osteoarthritis", "knee osteoarthritis"],
        "osteoporosis": ["osteoporosis"],
        "chronic kidney disease": ["chronic kidney disease", "CKD"],
        "cancer": ["cancer"],
        "depression": ["depression", "depressive symptoms"],
        "insulin resistance": ["insulin resistance"],
        "low back pain": ["low back pain"],
        "knee injury": ["ACL injury", "anterior cruciate ligament", "patellofemoral pain"],
        "COPD": ["COPD", "chronic obstructive pulmonary disease"],
    },
    "OUTCOME": {
        "muscle hypertrophy": ["hypertrophy", "muscle hypertrophy", "muscle growth", "muscle thickness",
                               "cross-sectional area", "muscle size", "muscle mass"],
        "muscle strength": ["muscle strength", "muscular strength", "maximal strength", "strength gains",
                            "one-repetition maximum", "1RM", "1-RM", "handgrip strength"],
        "power": ["muscle power", "muscular power", "power output", "peak power"],
        "jump performance": ["jump height", "jump performance", "vertical jump"],
        "sprint performance": ["sprint performance", "sprint time", "sprint speed"],
        "fat loss": ["fat loss", "fat mass", "body fat", "adiposity", "visceral fat", "weight loss"],
        "lean mass": ["lean mass", "lean body mass", "fat-free mass"],
        "body composition": ["body composition"],
        "BMI": ["body mass index", "BMI"],
        "cardiorespiratory fitness": ["VO2max", "VO2 max", "VO2peak", "VO 2max", "maximal oxygen uptake",
                                      "cardiorespiratory fitness", "aerobic capacity", "peak oxygen uptake"],
        "blood pressure": ["blood pressure", "systolic blood pressure", "diastolic blood pressure", "SBP", "DBP"],
        "glycemic control": ["glycemic control", "glycaemic control", "HbA1c", "glycated hemoglobin",
                             "fasting glucose", "blood glucose"],
        "insulin sensitivity": ["insulin sensitivity", "HOMA-IR"],
        "flexibility": ["flexibility", "range of motion", "ROM", "sit-and-reach"],
        "muscle protein synthesis": ["muscle protein synthesis", "MPS"],
        "endurance performance": ["endurance performance", "time to exhaustion", "time trial"],
        "cholesterol": ["cholesterol", "LDL", "HDL", "triglycerides", "lipid profile"],
        "inflammation": ["inflammation", "C-reactive protein", "CRP", "inflammatory markers"],
        "quality of life": ["quality of life"],
        "physical function": ["physical function", "physical performance", "gait speed", "functional capacity"],
        "injury risk": ["injury risk", "injury prevention", "injury incidence"],
        "bone density": ["bone mineral density", "BMD", "bone density"],
        "muscle damage": ["muscle damage", "muscle soreness", "DOMS", "creatine kinase"],
        "fatigue": ["fatigue"],
    },
    "POPULATION": {
        "older adults": ["older adults", "elderly", "older people", "aged 65", "older women", "older men"],
        "athletes": ["athletes", "athlete", "players", "soccer players", "football players", "basketball players"],
        "young adults": ["young adults", "young men", "young women", "college students", "university students"],
        "children and adolescents": ["children", "adolescents", "youth"],
        "women": ["women", "females", "postmenopausal"],
        "men": ["men", "males"],
        "trained individuals": ["trained individuals", "resistance-trained men", "trained men", "well-trained"],
        "untrained individuals": ["untrained", "sedentary"],
        "patients": ["patients"],
    },
}

# Acronyms that must match in UPPER CASE only (otherwise "sit", "rom", "bfr" match ordinary words)
CASE_SENSITIVE = {"SIT", "ROM", "BFR", "MPS", "EAA", "HMB", "CKD", "CRP", "BMD", "SBP", "DBP",
                  "LDL", "HDL", "CMJ", "MICT", "HIIT", "T2D", "T2DM", "BMI", "COPD", "DOMS", "1RM", "1-RM"}


def build_patterns():
    """Compile one regex per (type, canonical) pair. Longer synonyms are tried first."""
    pats = []
    for etype, entries in VOCAB.items():
        for canon, syns in entries.items():
            ci = sorted([s for s in syns if s not in CASE_SENSITIVE], key=len, reverse=True)
            cs = sorted([s for s in syns if s in CASE_SENSITIVE], key=len, reverse=True)
            if ci:
                pats.append((etype, canon, re.compile(
                    r"(?<![\w-])(" + "|".join(map(re.escape, ci)) + r")(?![\w-])", re.I)))
            if cs:
                pats.append((etype, canon, re.compile(
                    r"(?<![\w-])(" + "|".join(map(re.escape, cs)) + r")(?![\w-])")))
    return pats


PATTERNS = build_patterns()

# Generic words the NER model often returns that add no knowledge
NER_STOP = {"exercise", "training", "muscle", "muscles", "participants", "subjects", "patients",
            "intervention", "control", "group", "groups", "body", "study", "studies", "trial",
            "effect", "effects", "weeks", "week", "session", "sessions", "adults", "health",
            "performance", "activity", "physical activity", "program", "programme", "load",
            "supplementation", "supplement", "placebo", "baseline", "men", "women", "people"}

NER_STOP |= {  # direction / statistics / study-design words the model tags as entities
    "increased", "increase", "increases", "decreased", "decrease", "reduced", "reduction", "improved",
    "improves", "improvement", "improvements", "greater", "higher", "high", "lower", "low", "positive",
    "superior", "meta", "analysis", "analyses", "random", "trials", "controlled trials", "search",
    "review", "data", "outcomes", "measures", "strength", "mass", "weight", "resistance", "type", "over",
    "under", "back", "lower", "upper", "blood", "oxygen", "fitness", "dietary", "anti", "medications",
    "physical", "skeletal", "hyper", "stiff", "adverse", "meals", "sitting", "interventions", "therapy",
    "risk", "level", "levels", "function", "change", "changes", "response", "responses",
    "torque", "change of", "damage", "failure", "loss", "stretch", "tension", "acceleration",
    "activation", "chronic", "contraction", "disease", "endurance", "symptoms", "dysfunction",
    "aging", "discomfort", "vast", "overhead", "patellar", "hand", "elevated", "normal", "moderate",
    "heavy resistance", "jump", "lean", "metabolic", "motion", "pressure", "renal", "sleep", "type 1",
    "type 2", "impairment"}

NER_LABEL_MAP = {  # d4data/biomedical-ner-all label -> our entity type
    "Disease_disorder": "CONDITION", "Sign_symptom": "CONDITION",
    "Biological_structure": "BODY_STRUCTURE", "Medication": "NUTRIENT",
    "Therapeutic_procedure": "TRAINING_TYPE", "Activity": "EXERCISE",
    "Diagnostic_procedure": "BIOMARKER", "Lab_value": "BIOMARKER",
}
# Biomarker terms worth keeping (everything else the model tags as a "procedure" is noise)
BIOMARKER_KEEP = {"insulin", "glucose", "heart rate", "lactate", "testosterone", "cortisol",
                  "body weight", "waist circumference", "triglyceride", "triglycerides", "hemoglobin",
                  "growth hormone", "igf-1", "mtor", "ampk", "interleukin-6", "il-6", "tnf-α",
                  "creatinine", "myostatin", "vo2", "heart rate variability", "arterial stiffness"}


def refine_mentions(m: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Clean transformer NER mentions (dictionary mentions are kept as they are).
    1. map an NER term containing a known synonym onto that dictionary entity
       e.g. 'diabetes mellitus' -> 'type 2 diabetes'
    2. drop sub-word fragments ('hyper', 'sarcope'): term must occur as a whole word
    3. drop direction/statistics words and generic single words (NER_STOP)
    4. keep BIOMARKER terms only if whitelisted
    """
    syn2canon = {}
    for etype, entries in VOCAB.items():
        for canon, syns in entries.items():
            for s in syns + [canon]:
                syn2canon[s.lower()] = (etype, canon)
    syn_pat = re.compile(r"(?<![\w-])(" + "|".join(
        map(re.escape, sorted(syn2canon, key=len, reverse=True))) + r")(?![\w-])")
    corpus = corpus.lower()
    whole_word_cache = {}

    def whole_word(term):
        if term not in whole_word_cache:
            whole_word_cache[term] = re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", corpus) is not None
        return whole_word_cache[term]

    rows = []
    for r in m.itertuples(index=False):
        if r.method != "transformer_ner":
            rows.append(r._asdict())
            continue
        name = r.entity_name.lower().strip()
        etype_in = {"MEASURE": "BIOMARKER"}.get(r.entity_type, r.entity_type)  # labels of older runs
        if etype_in == "OUTCOME":   # NER 'Lab_value' hits are numbers/adjectives; outcomes come from the dictionary
            continue
        r = r._replace(entity_type=etype_in)
        hit = syn_pat.search(name)
        if hit:                                          # rule 1
            etype, canon = syn2canon[hit.group(1)]
            rows.append(r._asdict() | {"entity_type": etype, "entity_name": canon})
            continue
        if name in NER_STOP or not whole_word(name):     # rules 2-3
            continue
        if any(w in NER_STOP for w in name.split()) and len(name.split()) == 1:
            continue
        if name in BIOMARKER_KEEP:                       # e.g. 'insulin' tagged as Medication
            r = r._replace(entity_type="BIOMARKER")
        if r.entity_type == "BIOMARKER" and name not in BIOMARKER_KEEP:   # rule 4
            continue
        rows.append(r._asdict() | {"entity_name": name})
    out = pd.DataFrame(rows)
    # merge simple plurals onto the singular form ('skeletal muscles' -> 'skeletal muscle')
    names = set(out["entity_name"])
    out["entity_name"] = [n[:-1] if n.endswith("s") and n[:-1] in names else n for n in out["entity_name"]]
    return out.groupby(["pmid", "entity_type", "entity_name", "method"], as_index=False)["mention_count"].sum()


# ===========================================================================
# INGEST
# ===========================================================================
def ingest():
    abstracts = pd.DataFrame([json.loads(l) for l in open(DATA / "abstracts.jsonl", encoding="utf-8")])
    exercises = json.load(open(DATA / "exercises.json", encoding="utf-8"))
    foods = pd.read_csv(DATA / "foods.csv")
    print(f"Ingested {len(abstracts)} abstracts, {len(exercises)} exercises, {len(foods)} foods")
    return abstracts, exercises, foods


# ===========================================================================
# CLEAN / TRANSFORM the structured + semi-structured sources
# ===========================================================================
def curate_foods(foods: pd.DataFrame) -> pd.DataFrame:
    f = foods.copy()
    # Short readable name: first 1-2 comma-separated parts of the USDA description
    f["food_name"] = f["description"].str.split(",").str[:2].str.join(",").str.strip()
    # Fibre is blank for animal foods in USDA = they contain none -> 0
    animal = f["category"].str.contains("Poultry|Beef|Fish|Dairy|Egg|Finfish", case=False, na=False)
    f.loc[animal & f["fiber_g"].isna(), "fiber_g"] = 0.0
    # Derived features (feature engineering for search + dashboard)
    f["protein_per_100kcal"] = (f["protein_g"] / f["energy_kcal"] * 100).round(2)
    f["high_protein"] = f["protein_g"] >= 15
    cols = ["fdc_id", "food_name", "description", "category", "energy_kcal", "protein_g", "carbs_g",
            "fat_g", "fiber_g", "sugars_g", "sodium_mg", "protein_per_100kcal", "high_protein"]
    return f[cols]


def curate_exercises(exercises: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten nested JSON (like Snowflake LATERAL FLATTEN) into two relational tables."""
    ex_rows, mus_rows = [], []
    for e in exercises:
        ex_rows.append({k: e.get(k) for k in ["id", "name", "category", "level", "equipment", "mechanic", "force"]}
                       | {"instructions": " ".join(e.get("instructions", []))})
        for m in e.get("primaryMuscles", []):
            mus_rows.append({"exercise_id": e["id"], "muscle": m, "role": "primary"})
        for m in e.get("secondaryMuscles", []):
            mus_rows.append({"exercise_id": e["id"], "muscle": m, "role": "secondary"})
    ex = pd.DataFrame(ex_rows)
    ex["equipment"] = ex["equipment"].fillna("none")
    return ex, pd.DataFrame(mus_rows)


# ===========================================================================
# EXTRACT from unstructured abstracts
# ===========================================================================
def conclusion_of(text: str) -> str:
    """Take the labelled CONCLUSION(S) section; otherwise the last two sentences."""
    m = re.search(r"CONCLUSIONS?(?: AND [A-Z ]+)?:\s*(.+?)(?:\s+[A-Z][A-Z ]{3,}:|$)", text)
    if m:
        return m.group(1).strip()
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[-2:])


def dictionary_entities(text: str):
    """Rule-based extraction. Returns list of (type, canonical, matched_text, start, end)."""
    found = []
    for etype, canon, pat in PATTERNS:
        for m in pat.finditer(text):
            found.append((etype, canon, m.group(0), m.start(), m.end()))
    # Remove matches that sit inside a longer match (e.g. 'protein' inside 'whey protein')
    found.sort(key=lambda x: (x[3], -(x[4] - x[3])))
    kept, last_end = [], -1
    for f in found:
        if f[3] >= last_end:
            kept.append(f)
            last_end = f[4]
    return kept


def run_extraction(abstracts: pd.DataFrame):
    texts = (abstracts["title"].fillna("") + ". " + abstracts["abstract"].fillna("")).tolist()
    conclusions = [conclusion_of(t) for t in abstracts["abstract"]]

    # ---- 1. Rule-based dictionary NER ----
    t0 = time.time()
    mentions = []  # (pmid, type, canonical, method)
    for pmid, text in zip(abstracts["pmid"], texts):
        for etype, canon, *_ in dictionary_entities(text):
            mentions.append((pmid, etype, canon, "dictionary"))
    print(f"  dictionary: {len(mentions)} mentions in {time.time() - t0:.1f}s")

    keywords = [[] for _ in texts]
    finding = [("not_run", None)] * len(texts)

    if not FAST:
        from transformers import pipeline
        from keybert import KeyBERT

        # ---- 2. Transformer biomedical NER (finds what the dictionary missed) ----
        t0 = time.time()
        # "first" merges sub-word pieces into whole words (avoids fragments like 'sarcope')
        ner = pipeline("token-classification", model=NER_MODEL, aggregation_strategy="first")
        known = {s.lower() for entries in VOCAB.values() for canon, syns in entries.items()
                 for s in syns + [canon]}
        n_ner = 0
        for i, (pmid, text) in enumerate(zip(abstracts["pmid"], texts)):
            for ent in ner(text[:2000]):  # model max length ~512 tokens
                label = ent["entity_group"]
                word = re.sub(r"\s+", " ", ent["word"].replace(" ##", "")).strip(" .,;:()").lower()
                if (label in NER_LABEL_MAP and ent["score"] >= 0.70 and len(word) > 3
                        and word not in NER_STOP and word not in known and not word.isdigit()
                        and "#" not in word):
                    mentions.append((pmid, NER_LABEL_MAP[label], word, "transformer_ner"))
                    n_ner += 1
            if (i + 1) % 50 == 0:
                print(f"    NER {i + 1}/{len(texts)}")
        print(f"  transformer NER: {n_ner} extra mentions in {time.time() - t0:.0f}s")

        # ---- 3. KeyBERT keyword extraction (embedding similarity) ----
        t0 = time.time()
        kw_model = KeyBERT(model=KW_MODEL)
        for i, text in enumerate(texts):
            kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english",
                                            use_mmr=True, diversity=0.5, top_n=5)
            keywords[i] = [k for k, _ in kws]
        print(f"  KeyBERT: keywords for {len(texts)} abstracts in {time.time() - t0:.0f}s")

        # ---- 4. Zero-shot finding direction ("sentiment" for scientific text) ----
        finding = classify_findings(conclusions)

    m = pd.DataFrame(mentions, columns=["pmid", "entity_type", "entity_name", "method"])
    m = m.groupby(["pmid", "entity_type", "entity_name", "method"], as_index=False).size() \
         .rename(columns={"size": "mention_count"})
    return m, conclusions, keywords, finding


def classify_findings(conclusions):
    from transformers import pipeline
    t0 = time.time()
    zs = pipeline("zero-shot-classification", model=ZS_MODEL)
    finding = []
    for i, c in enumerate(conclusions):
        r = zs(c[:1500], candidate_labels=list(FINDING_LABELS), hypothesis_template="{}.")
        finding.append((FINDING_LABELS[r["labels"][0]], round(r["scores"][0], 3)))
        if (i + 1) % 50 == 0:
            print(f"    zero-shot {i + 1}/{len(conclusions)}")
    print(f"  zero-shot: findings classified in {time.time() - t0:.0f}s")
    return finding


# ===========================================================================
# STORE: curated tables
# ===========================================================================
def store(abstracts, m, conclusions, keywords, finding, ex, ex_mus, foods):
    # m: one row per (pmid, entity_type, entity_name, method) with mention_count
    corpus = " ".join(abstracts["title"].fillna("") + ". " + abstracts["abstract"].fillna(""))
    before = (m["method"] == "transformer_ner").sum()
    m = refine_mentions(m, corpus)
    print(f"  refine: transformer mention rows {before} -> {(m['method'] == 'transformer_ner').sum()}")

    # entities table: one row per unique (type, name)
    ent = (m.groupby(["entity_type", "entity_name"])
             .agg(n_abstracts=("pmid", "nunique"), n_mentions=("mention_count", "sum"),
                  methods=("method", lambda s: ";".join(sorted(set(s)))))
             .reset_index())
    # keep transformer-only entities only if seen in >= 2 abstracts (reduces noise)
    ent = ent[(ent["methods"] != "transformer_ner") | (ent["n_abstracts"] >= 2)]
    ent = ent.sort_values(["entity_type", "n_abstracts"], ascending=[True, False]).reset_index(drop=True)
    ent.insert(0, "entity_id", [f"E{i:04d}" for i in range(1, len(ent) + 1)])

    # abstract_entities: bridge table (which abstract mentions which entity, how often)
    ae = (m.merge(ent[["entity_id", "entity_type", "entity_name"]], on=["entity_type", "entity_name"])
            .groupby(["pmid", "entity_id", "method"], as_index=False)["mention_count"].sum())

    # abstracts_clean
    a = abstracts[["pmid", "title", "year", "journal"]].copy()
    a["topics"] = abstracts["topics"].apply("; ".join)
    a["mesh_terms"] = abstracts["mesh_terms"].apply("; ".join)
    a["abstract"] = abstracts["abstract"]
    a["conclusion"] = conclusions
    a["keywords"] = ["; ".join(k) for k in keywords]
    a["finding"] = [f[0] for f in finding]
    a["finding_confidence"] = [f[1] for f in finding]
    a["n_entities"] = a["pmid"].map(ae.groupby("pmid")["entity_id"].nunique()).fillna(0).astype(int)

    a.to_csv(OUT / "abstracts_clean.csv", index=False)
    ent.to_csv(OUT / "entities.csv", index=False)
    ae.to_csv(OUT / "abstract_entities.csv", index=False)
    ex.to_csv(OUT / "exercises.csv", index=False)
    ex_mus.to_csv(OUT / "exercise_muscles.csv", index=False)
    foods.to_csv(OUT / "foods_clean.csv", index=False)

    # ---- summary for the report ----
    lines = ["EXTRACTION SUMMARY", "=" * 60,
             f"Abstracts processed        : {len(a)}",
             f"Unique entities            : {len(ent)}",
             f"Abstract-entity links      : {len(ae)}",
             f"Avg entities per abstract  : {a['n_entities'].mean():.1f}",
             f"Abstracts with 0 entities  : {(a['n_entities'] == 0).sum()}",
             f"Exercises / muscle links   : {len(ex)} / {len(ex_mus)}",
             f"Foods                      : {len(foods)}", "",
             "Entities by type:", ent.groupby("entity_type").size().to_string(), "",
             "Mentions by method:", ae.groupby("method")["mention_count"].sum().to_string(), "",
             "Finding direction:", a["finding"].value_counts().to_string(), "",
             "Top 5 entities per type (by number of abstracts):"]
    for t, g in ent.groupby("entity_type"):
        top = ", ".join(f"{r.entity_name} ({r.n_abstracts})" for r in g.head(5).itertuples())
        lines.append(f"  {t:15} {top}")
    summary = "\n".join(lines)
    (OUT / "extraction_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def load_previous(abstracts):
    """--refine: rebuild mentions + keywords from an earlier full run in curated/."""
    ent = pd.read_csv(OUT / "entities.csv")
    ae = pd.read_csv(OUT / "abstract_entities.csv", dtype={"pmid": str})
    prev = pd.read_csv(OUT / "abstracts_clean.csv", dtype={"pmid": str}).set_index("pmid")
    m = ae.merge(ent[["entity_id", "entity_type", "entity_name"]], on="entity_id")
    m = m[["pmid", "entity_type", "entity_name", "method", "mention_count"]]
    pmids = abstracts["pmid"].astype(str)
    keywords = [str(prev.at[p, "keywords"]).split("; ") if p in prev.index and pd.notna(prev.at[p, "keywords"])
                else [] for p in pmids]
    return m, keywords


if __name__ == "__main__":
    abstracts, exercises, foods = ingest()
    abstracts["pmid"] = abstracts["pmid"].astype(str)
    foods_c = curate_foods(foods)
    ex, ex_mus = curate_exercises(exercises)
    if REFINE:
        print("Refining previous run (re-classify findings, clean entities)...")
        m, keywords = load_previous(abstracts)
        conclusions = [conclusion_of(t) for t in abstracts["abstract"]]
        finding = classify_findings(conclusions)
    else:
        print("Extracting from abstracts" + (" (FAST: dictionary only)" if FAST else "") + "...")
        m, conclusions, keywords, finding = run_extraction(abstracts)
    store(abstracts, m, conclusions, keywords, finding, ex, ex_mus, foods_c)
    print(f"\nSaved curated tables to '{OUT}/'")
