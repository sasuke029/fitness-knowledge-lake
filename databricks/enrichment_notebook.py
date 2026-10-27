# Databricks notebook source
# MAGIC %md
# MAGIC # Fitness & Health Knowledge Lake - Data Enrichment workflow in Databricks
# MAGIC **Fitness & Health Knowledge Lake** — Delta Lake (Bronze → Silver → Gold) + external knowledge (Wikidata) + MLflow tracking
# MAGIC
# MAGIC | Layer | What happens | Tables |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | Curated output of the Snowflake stage (Part 1.1) landed as-is | `bronze_*` |
# MAGIC | **Silver** | Entities enriched with Wikidata (QID, description, MeSH ID), manual review of wrong links via Delta `MERGE`, relation filtering | `silver_entities_enriched`, `silver_wikidata_relations` |
# MAGIC | **Gold** | Graph-ready node and edge tables for Neo4j (Part 1.3) incl. statistical co-occurrence enrichment | `gold_nodes`, `gold_edges` |
# MAGIC | **MLflow** | Parameters, link-quality metrics and graph statistics of every enrichment run | experiment run |
# MAGIC
# MAGIC **Before running:** Catalog → `workspace` → `default` → create volume `knowledge_lake` (cell 1 does this) → *Upload to this volume* → upload the 8 CSV files from the local `curated/` folder.

# COMMAND ----------

# Cell 1 - configuration (works in Databricks; also runs locally with plain PySpark for testing)
import json, math, os, re
from pyspark.sql import functions as F, Window

try:
    spark  # noqa: F821  (pre-defined in Databricks)
    IS_DBX = True
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").getOrCreate()
    IS_DBX = False

CATALOG, SCHEMA, VOLUME = "workspace", "default", "knowledge_lake"
if IS_DBX:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
    SRC = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
else:

    SRC = os.environ.get("KL_SRC", "curated")


def table(name):
    return f"{CATALOG}.{SCHEMA}.{name}" if IS_DBX else name


def save(df, name):
    """Write a Delta table (Databricks) / parquet table (local test) and return it."""
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .format("delta" if IS_DBX else "parquet").saveAsTable(table(name)))
    return spark.table(table(name))


print("Databricks:", IS_DBX, "| source:", SRC)

# COMMAND ----------

# MAGIC %md ## 1. Bronze – land the curated files as Delta tables

# COMMAND ----------

def read_csv(fname):
    return (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
                 .option("inferSchema", True).csv(f"{SRC}/{fname}"))

BRONZE = {
    "bronze_entities": "entities.csv",
    "bronze_abstract_entities": "abstract_entities.csv",
    "bronze_abstracts": "abstracts_clean.csv",
    "bronze_exercises": "exercises.csv",
    "bronze_exercise_muscles": "exercise_muscles.csv",
    "bronze_foods": "foods_clean.csv",
    "bronze_wikidata_links": "entity_wikidata_links.csv",
    "bronze_wikidata_relations": "wikidata_relations.csv",
}
bronze = {}
for tname, fname in BRONZE.items():
    df = read_csv(fname).withColumn("_ingested_at", F.current_timestamp()).withColumn("_source_file", F.lit(fname))
    bronze[tname] = save(df, tname)
    print(f"{tname:28} {bronze[tname].count():>6} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. External knowledge: Wikidata entity linking
# MAGIC Linking was run with `wikidata_enrich.py` (label/alias search + full-text search → type-aware candidate scoring → SPARQL for selected properties).
# MAGIC The cell below re-runs the **same API call live** for a few entities to show the enrichment step inside Databricks.
# MAGIC If outbound internet is restricted on the workspace, the pre-fetched links (identical code) are used.

# COMMAND ----------

import requests

HEADERS = {"User-Agent": "FitnessKnowledgeLake/1.0 (portfolio project; https://github.com/sasuke029)"}
try:
    live = []
    for term in ["squat", "creatine", "sarcopenia"]:
        r = requests.get("https://www.wikidata.org/w/api.php", headers=HEADERS, timeout=15,
                         params={"action": "wbsearchentities", "search": term, "language": "en",
                                 "format": "json", "limit": 1})
        hit = r.json()["search"][0]
        live.append((term, hit["id"], hit.get("label"), hit.get("description")))
    display_df = spark.createDataFrame(live, "term string, qid string, label string, description string")
    display_df.show(truncate=80)
    print("Live Wikidata access: OK")
except Exception as ex:
    print("Live Wikidata access not available from this workspace -> using pre-fetched links.", type(ex).__name__)

# COMMAND ----------

# MAGIC %md ## 3. Silver – enriched entities + human review of low-quality links (Delta MERGE)

# COMMAND ----------

ents = bronze["bronze_entities"].select("entity_id", "entity_type", "entity_name", "n_abstracts", "n_mentions", "methods")
links = bronze["bronze_wikidata_links"].select(
    "entity_id", "qid", "wd_label", "wd_description", "link_score",
    F.col("linked").cast("boolean").alias("linked"), "mesh_id")

silver_ents = save(
    ents.join(links, "entity_id", "left")
        .withColumn("link_status", F.when(F.col("linked"), "auto_linked").otherwise("unlinked")),
    "silver_entities_enriched")

# Manual review: links the scorer got wrong (found by inspecting all 129 links)
overrides = spark.createDataFrame([
    ("knee", "pro gamer called 'Knee' instead of the knee joint"),
    ("ampk", "rat protein subunit instead of the AMPK enzyme class"),
    ("triglyceride", "one specific triglyceride isomer instead of the lipid class"),
], "entity_name string, reason string")
overrides.createOrReplaceTempView("link_overrides")

if IS_DBX:
    spark.sql(f"""
        MERGE INTO {table('silver_entities_enriched')} t
        USING link_overrides o ON t.entity_name = o.entity_name
        WHEN MATCHED THEN UPDATE SET t.qid = NULL, t.wd_label = NULL, t.wd_description = NULL,
                                     t.mesh_id = NULL, t.linked = false, t.link_status = 'rejected_in_review'
    """)
else:  # local test: same effect without Delta
    o = [r.entity_name for r in overrides.collect()]
    rej = F.col("entity_name").isin(o)
    df = spark.table(table("silver_entities_enriched")).toPandas()
    df.loc[df.entity_name.isin(o), ["qid", "wd_label", "wd_description", "mesh_id"]] = None
    df.loc[df.entity_name.isin(o), "linked"] = False
    df.loc[df.entity_name.isin(o), "link_status"] = "rejected_in_review"
    save(spark.createDataFrame(df), "silver_entities_enriched_tmp")
    save(spark.table(table("silver_entities_enriched_tmp")), "silver_entities_enriched")

silver_ents = spark.table(table("silver_entities_enriched"))
silver_ents.groupBy("entity_type").agg(
    F.count("*").alias("entities"),
    F.round(F.avg(F.col("linked").cast("int")), 2).alias("link_rate")).orderBy("entity_type").show()

# COMMAND ----------

# Delta Lake time travel: the table before and after the review MERGE (Databricks only)
if IS_DBX:
    spark.sql(f"DESCRIBE HISTORY {table('silver_entities_enriched')}") \
         .select("version", "timestamp", "operation", "operationMetrics").show(truncate=60)
    v0 = spark.sql(f"SELECT entity_name, wd_label, link_status FROM {table('silver_entities_enriched')} VERSION AS OF 0 "
                   "WHERE entity_name IN ('knee','ampk','triglyceride')")
    v0.show(truncate=60)

# COMMAND ----------

# Wikidata relations: keep knowledge useful for the graph, cap noisy relation types
KEEP = ["subclass_of", "instance_of", "part_of", "has_part", "has_cause", "has_effect", "symptom",
        "risk_factor", "drug_or_therapy", "treats_condition", "health_specialty", "connects_with"]
CAP = {"drug_or_therapy": 12, "subclass_of": 8, "instance_of": 5, "part_of": 6, "has_part": 6}

valid = silver_ents.filter("linked").select("entity_id", "qid")
rels = (bronze["bronze_wikidata_relations"]
        .join(valid, ["entity_id", "qid"])                       # drops relations of rejected links
        .filter(F.col("relation").isin(KEEP))                   # drops genetic_association (509 gene rows)
        .filter(~F.col("value_label").rlike(r"(?i)process|^regulation of|binding$|biosynthetic|catabolic")))
w = Window.partitionBy("entity_id", "relation").orderBy("value_label")
cap_expr = F.coalesce(*[F.when(F.col("relation") == k, F.lit(v)) for k, v in CAP.items()], F.lit(1000))
rels = rels.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= cap_expr).drop("rn")
silver_rels = save(rels, "silver_wikidata_relations")
silver_rels.groupBy("relation").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gold – graph-ready nodes and edges for Neo4j
# MAGIC Node labels: `Study`, `Entity` (typed: EXERCISE, MUSCLE, NUTRIENT, CONDITION, …), `Concept` (from Wikidata), `ExerciseVariant` (exercise DB), `Food`

# COMMAND ----------

abstracts = bronze["bronze_abstracts"]
ae = bronze["bronze_abstract_entities"]

# --- extra entity nodes needed by the structured sources (muscles / nutrients not seen in abstracts)
db_muscles = [r.muscle for r in bronze["bronze_exercise_muscles"].select("muscle").distinct().collect()]
food_nutrients = {"protein_g": "protein", "carbs_g": "carbohydrate", "fat_g": "dietary fat", "fiber_g": "fiber"}
existing = {(r.entity_type, r.entity_name) for r in silver_ents.select("entity_type", "entity_name").collect()}
extra = [("MUSCLE", m) for m in db_muscles if ("MUSCLE", m) not in existing] + \
        [("NUTRIENT", n) for n in food_nutrients.values() if ("NUTRIENT", n) not in existing]
extra_df = spark.createDataFrame([(f"X{i:03d}", t, n) for i, (t, n) in enumerate(extra, 1)],
                                 "entity_id string, entity_type string, entity_name string")
entity_nodes = silver_ents.select("entity_id", "entity_type", "entity_name", "n_abstracts", "qid",
                                  "wd_label", "wd_description", "mesh_id", "link_status") \
                          .unionByName(extra_df, allowMissingColumns=True) \
                          .fillna({"n_abstracts": 0, "link_status": "structured_source"})
eid = {(r.entity_type, r.entity_name): r.entity_id for r in entity_nodes.collect()}

# Concepts = Wikidata values that are NOT already one of our entities
ent_qids = silver_ents.filter("linked").select(F.col("qid").alias("value_qid"), F.col("entity_id").alias("dst_entity"))
rel_resolved = silver_rels.join(ent_qids, "value_qid", "left")
concept_nodes = (rel_resolved.filter("dst_entity IS NULL")
                 .select(F.concat(F.lit("C_"), "value_qid").alias("node_id"), "value_qid", "value_label")
                 .dropDuplicates(["node_id"]))

nodes = (
    abstracts.select(F.concat(F.lit("S_"), F.col("pmid").cast("string")).alias("node_id"), F.lit("Study").alias("label"),
                     F.col("title").alias("name"),
                     F.to_json(F.struct("pmid", "year", "journal", "finding", "finding_confidence", "keywords", "topics")).alias("props"))
    .unionByName(entity_nodes.select(F.col("entity_id").alias("node_id"), F.lit("Entity").alias("label"),
                     F.col("entity_name").alias("name"),
                     F.to_json(F.struct("entity_type", "n_abstracts", "qid", "wd_label", "wd_description", "mesh_id", "link_status")).alias("props")))
    .unionByName(concept_nodes.select("node_id", F.lit("Concept").alias("label"), F.col("value_label").alias("name"),
                     F.to_json(F.struct(F.col("value_qid").alias("qid"))).alias("props")))
    .unionByName(bronze["bronze_exercises"].select(F.concat(F.lit("X_"), "id").alias("node_id"), F.lit("ExerciseVariant").alias("label"),
                     "name", F.to_json(F.struct("category", "level", "equipment", "mechanic", "force")).alias("props")))
    .unionByName(bronze["bronze_foods"].select(F.concat(F.lit("F_"), F.col("fdc_id").cast("string")).alias("node_id"), F.lit("Food").alias("label"),
                     F.col("food_name").alias("name"),
                     F.to_json(F.struct("category", "energy_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g",
                                        "protein_per_100kcal", "high_protein")).alias("props")))
)
gold_nodes = save(nodes, "gold_nodes")

# COMMAND ----------

# --- Edges
# (Study)-[:MENTIONS]->(Entity)
e_mentions = ae.groupBy("pmid", "entity_id").agg(F.sum("mention_count").alias("weight"),
                                                 F.concat_ws(";", F.collect_set("method")).alias("method")) \
    .select(F.concat(F.lit("S_"), F.col("pmid").cast("string")).alias("src"), F.lit("MENTIONS").alias("rel"),
            F.col("entity_id").alias("dst"), F.col("weight").cast("double"), F.col("method").alias("source"))

# (Entity)-[:CO_OCCURS_WITH]-(Entity): statistical enrichment with pointwise mutual information
N = abstracts.count()
pairs = (ae.select("pmid", "entity_id").distinct().alias("a")
         .join(ae.select("pmid", "entity_id").distinct().alias("b"), "pmid")
         .filter(F.col("a.entity_id") < F.col("b.entity_id"))
         .groupBy(F.col("a.entity_id").alias("e1"), F.col("b.entity_id").alias("e2")).agg(F.count("*").alias("n12")))
df_ = ae.select("pmid", "entity_id").distinct().groupBy("entity_id").agg(F.count("*").alias("n"))
pairs = (pairs.join(df_.withColumnRenamed("entity_id", "e1").withColumnRenamed("n", "n1"), "e1")
              .join(df_.withColumnRenamed("entity_id", "e2").withColumnRenamed("n", "n2"), "e2")
              .withColumn("pmi", F.round(F.log2(F.col("n12") * N / (F.col("n1") * F.col("n2"))), 3))
              .filter((F.col("n12") >= 3) & (F.col("pmi") > 0)))
e_cooc = pairs.select(F.col("e1").alias("src"), F.lit("CO_OCCURS_WITH").alias("rel"), F.col("e2").alias("dst"),
                      F.col("n12").cast("double").alias("weight"), F.concat(F.lit("pmi="), F.col("pmi")).alias("source"))

# (Entity)-[wikidata relation]->(Entity | Concept)
e_wd = rel_resolved.select(F.col("entity_id").alias("src"), F.upper("relation").alias("rel"),
                           F.coalesce("dst_entity", F.concat(F.lit("C_"), "value_qid")).alias("dst"),
                           F.lit(1.0).alias("weight"), F.lit("wikidata").alias("source"))

# (ExerciseVariant)-[:TARGETS {primary|secondary}]->(Entity MUSCLE)
mus_map = spark.createDataFrame([(m, eid[("MUSCLE", m)]) for m in db_muscles], "muscle string, dst string")
e_targets = (bronze["bronze_exercise_muscles"].join(mus_map, "muscle")
             .select(F.concat(F.lit("X_"), "exercise_id").alias("src"), F.lit("TARGETS").alias("rel"), "dst",
                     F.when(F.col("role") == "primary", 1.0).otherwise(0.5).alias("weight"),
                     F.concat(F.lit("exercise_db:"), "role").alias("source")))

# (ExerciseVariant)-[:VARIANT_OF]->(Entity EXERCISE)  rule-based name matching
VARIANT_RULES = {"squat": r"\bsquat", "deadlift": r"deadlift", "bench press": r"bench press", "leg press": r"leg press",
                 "lunge": r"\blunge", "hip thrust": r"hip thrust|glute bridge|hip bridge", "push-up": r"push-?up",
                 "pull-up": r"pull-?up|chin-?up", "leg extension": r"leg extension", "leg curl": r"leg curl",
                 "biceps curl": r"\b(bicep|biceps|barbell|dumbbell|hammer|preacher|concentration) curl",
                 "plank": r"\bplank", "row": r"\brow\b", "running": r"\brun\b|running|jog", "cycling": r"bicycl|cycling|bike",
                 "walking": r"\bwalk", "sprint": r"sprint", "countermovement jump": r"vertical jump|countermovement",
                 "drop jump": r"depth jump|drop jump"}
rules = [(eid[("EXERCISE", k)], v) for k, v in VARIANT_RULES.items() if ("EXERCISE", k) in eid]
ex = bronze["bronze_exercises"]
e_variant = None
for dst, pattern in rules:
    part = ex.filter(F.lower("name").rlike(pattern)).select(
        F.concat(F.lit("X_"), "id").alias("src"), F.lit("VARIANT_OF").alias("rel"), F.lit(dst).alias("dst"),
        F.lit(1.0).alias("weight"), F.lit("rule:name_match").alias("source"))
    e_variant = part if e_variant is None else e_variant.unionByName(part)

# (Food)-[:CONTAINS {g per 100 g}]->(Entity NUTRIENT)
e_contains = None
for col, nutrient in food_nutrients.items():
    part = bronze["bronze_foods"].filter(F.col(col) > 0).select(
        F.concat(F.lit("F_"), F.col("fdc_id").cast("string")).alias("src"), F.lit("CONTAINS").alias("rel"),
        F.lit(eid[("NUTRIENT", nutrient)]).alias("dst"), F.col(col).cast("double").alias("weight"),
        F.lit(f"usda:{col}").alias("source"))
    e_contains = part if e_contains is None else e_contains.unionByName(part)

edges = e_mentions.unionByName(e_cooc).unionByName(e_wd).unionByName(e_targets).unionByName(e_variant).unionByName(e_contains)
gold_edges = save(edges.dropDuplicates(["src", "rel", "dst"]), "gold_edges")
gold_edges.groupBy("rel").count().orderBy(F.desc("count")).show(30)

# COMMAND ----------

# MAGIC %md ## 5. MLflow – track the enrichment run (parameters, quality metrics, graph statistics)

# COMMAND ----------

ent_stats = silver_ents.agg(F.count("*").alias("n"), F.sum(F.col("linked").cast("int")).alias("linked"),
                            F.avg(F.when(F.col("linked"), F.col("link_score"))).alias("score")).first()
by_type = {r.entity_type: round(r.rate, 3) for r in
           silver_ents.groupBy("entity_type").agg(F.avg(F.col("linked").cast("int")).alias("rate")).collect()}
edge_counts = {r.rel: r["count"] for r in gold_edges.groupBy("rel").count().collect()}
node_counts = {r.label: r["count"] for r in gold_nodes.groupBy("label").count().collect()}

metrics = {"link_rate": ent_stats.linked / ent_stats.n, "avg_link_score": float(ent_stats.score),
           "n_wikidata_relations": silver_rels.count(), "n_nodes": gold_nodes.count(), "n_edges": gold_edges.count(),
           **{f"link_rate_{k.lower()}": v for k, v in by_type.items()},
           **{f"edges_{k.lower()}": v for k, v in edge_counts.items()},
           **{f"nodes_{k.lower()}": v for k, v in node_counts.items()}}
params = {"linker": "wbsearchentities+fulltext, type-aware scoring", "min_link_score": 1.0,
          "manual_overrides": overrides.count(), "relations_kept": ",".join(KEEP),
          "cooccurrence_min_studies": 3, "cooccurrence_measure": "PMI>0"}

try:
    import mlflow
    with mlflow.start_run(run_name="wikidata_enrichment_v1") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"overrides": [r.asDict() for r in overrides.collect()]}, "manual_overrides.json")
        mlflow.log_dict({"unlinked": [r.entity_name for r in silver_ents.filter("NOT linked").collect()]}, "unlinked_entities.json")
        mlflow.set_tags({"layer": "silver->gold", "source": "Wikidata", "project": "Fitness & Health Knowledge Lake"})
        print("MLflow run:", run.info.run_id)
except ImportError:
    print("mlflow not installed locally - metrics:")
for k, v in sorted(metrics.items()):
    print(f"  {k:32} {v:.3f}" if isinstance(v, float) else f"  {k:32} {v}")

# COMMAND ----------

# MAGIC %md ## 6. Export gold tables for Neo4j (Part 1.3)

# COMMAND ----------

out = f"{SRC}/neo4j" if IS_DBX else os.environ.get("KL_OUT", "neo4j")
import os
os.makedirs(out, exist_ok=True)
gold_nodes.toPandas().to_csv(f"{out}/nodes.csv", index=False)
gold_edges.toPandas().to_csv(f"{out}/edges.csv", index=False)
print("Exported to", out)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: what Wikidata adds to the graph (Part 2.2)
# MAGIC Relations between our own entities that **no abstract states explicitly** but Wikidata does:

# COMMAND ----------

names = entity_nodes.select(F.col("entity_id").alias("id"), F.col("entity_name").alias("name"))
(gold_edges.filter("source = 'wikidata' AND dst NOT LIKE 'C_%'")
    .join(names.withColumnRenamed("id", "src").withColumnRenamed("name", "from"), "src")
    .join(names.withColumnRenamed("id", "dst").withColumnRenamed("name", "to"), "dst")
    .select("from", "rel", "to").orderBy("rel").show(50, truncate=False))
