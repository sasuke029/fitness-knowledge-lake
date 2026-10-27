-- ============================================================================
-- Fitness & Health Knowledge Lake - Data Curation pipeline in Snowflake
-- Fitness & Health Knowledge Lake:  INGEST  ->  EXTRACT  ->  STORE
--
-- Design implemented in Snowflake SQL. The same logic was executed locally in
-- Python (curate.py) to produce the curated tables used downstream.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 0. Setup: warehouse, database, layered schemas (raw -> curated)
-- ---------------------------------------------------------------------------
CREATE WAREHOUSE IF NOT EXISTS KL_WH WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
CREATE DATABASE IF NOT EXISTS FITNESS_KL;
CREATE SCHEMA IF NOT EXISTS FITNESS_KL.RAW;       -- landing zone, data as received
CREATE SCHEMA IF NOT EXISTS FITNESS_KL.CURATED;   -- cleaned, extracted, query-ready
USE SCHEMA FITNESS_KL.RAW;

-- ---------------------------------------------------------------------------
-- 1. INGEST: one stage, three file formats (structured / semi / unstructured)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE STAGE KL_STAGE;  -- PUT file://data/* @KL_STAGE;  (or an S3 external stage)

CREATE OR REPLACE FILE FORMAT FF_CSV  TYPE = CSV  SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"';
CREATE OR REPLACE FILE FORMAT FF_JSON TYPE = JSON STRIP_OUTER_ARRAY = TRUE;   -- exercises.json is one array

-- Structured: USDA foods
CREATE OR REPLACE TABLE RAW_FOODS (
    fdc_id INT, query STRING, description STRING, category STRING, data_type STRING,
    protein_g FLOAT, fiber_g FLOAT, sodium_mg FLOAT, sugars_g FLOAT, fat_g FLOAT,
    carbs_g FLOAT, energy_kcal FLOAT);
COPY INTO RAW_FOODS FROM @KL_STAGE/foods.csv FILE_FORMAT = FF_CSV;

-- Semi-structured: exercises kept as native JSON in a VARIANT column (schema-on-read)
CREATE OR REPLACE TABLE RAW_EXERCISES (doc VARIANT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP());
COPY INTO RAW_EXERCISES (doc) FROM @KL_STAGE/exercises.json FILE_FORMAT = FF_JSON;

-- Unstructured: PubMed abstracts (JSON Lines, one article per line)
CREATE OR REPLACE TABLE RAW_ABSTRACTS (doc VARIANT, loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP());
COPY INTO RAW_ABSTRACTS (doc) FROM @KL_STAGE/abstracts.jsonl FILE_FORMAT = (TYPE = JSON);

-- ---------------------------------------------------------------------------
-- 2. TRANSFORM structured + semi-structured sources
-- ---------------------------------------------------------------------------
USE SCHEMA FITNESS_KL.CURATED;

-- Foods: clean missing fibre (animal foods contain none) + derived features
CREATE OR REPLACE TABLE FOODS_CLEAN AS
SELECT fdc_id,
       SPLIT_PART(description, ',', 1) || ',' || SPLIT_PART(description, ',', 2) AS food_name,
       description, category, energy_kcal, protein_g, carbs_g, fat_g,
       IFF(fiber_g IS NULL AND category ILIKE ANY ('%poultry%','%beef%','%fish%','%dairy%','%egg%'),
           0, fiber_g)                                    AS fiber_g,
       sugars_g, sodium_mg,
       ROUND(protein_g / NULLIF(energy_kcal, 0) * 100, 2) AS protein_per_100kcal,
       protein_g >= 15                                    AS high_protein
FROM RAW.RAW_FOODS;

-- Exercises: VARIANT -> relational columns
CREATE OR REPLACE TABLE EXERCISES AS
SELECT doc:id::STRING        AS id,
       doc:name::STRING      AS name,
       doc:category::STRING  AS category,
       doc:level::STRING     AS level,
       COALESCE(doc:equipment::STRING, 'none') AS equipment,
       doc:mechanic::STRING  AS mechanic,
       doc:force::STRING     AS force,
       ARRAY_TO_STRING(doc:instructions, ' ') AS instructions
FROM RAW.RAW_EXERCISES;

-- Exercise -> muscle bridge table: LATERAL FLATTEN un-nests the muscle arrays
CREATE OR REPLACE TABLE EXERCISE_MUSCLES AS
SELECT doc:id::STRING AS exercise_id, m.value::STRING AS muscle, 'primary' AS role
FROM RAW.RAW_EXERCISES, LATERAL FLATTEN(input => doc:primaryMuscles) m
UNION ALL
SELECT doc:id::STRING, m.value::STRING, 'secondary'
FROM RAW.RAW_EXERCISES, LATERAL FLATTEN(input => doc:secondaryMuscles) m;

-- ---------------------------------------------------------------------------
-- 3. EXTRACT from unstructured abstracts
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE ABSTRACTS AS
SELECT doc:pmid::STRING     AS pmid,
       doc:title::STRING    AS title,
       doc:year::INT        AS year,
       doc:journal::STRING  AS journal,
       doc:abstract::STRING AS abstract,
       -- labelled CONCLUSION(S) section, else whole abstract
       COALESCE(REGEXP_SUBSTR(doc:abstract::STRING, 'CONCLUSIONS?:\\s*(.*)', 1, 1, 'e', 1),
                doc:abstract::STRING) AS conclusion
FROM RAW.RAW_ABSTRACTS;

-- 3a. RULE-BASED: dictionary (gazetteer) of canonical entities + synonyms
CREATE OR REPLACE TABLE ENTITY_DICTIONARY (entity_type STRING, entity_name STRING, synonym STRING);
INSERT INTO ENTITY_DICTIONARY VALUES
  ('TRAINING_TYPE','HIIT','high-intensity interval training'), ('TRAINING_TYPE','HIIT','HIIT'),
  ('EXERCISE','squat','squat'),            ('EXERCISE','squat','back squat'),
  ('MUSCLE','quadriceps','vastus lateralis'), ('MUSCLE','quadriceps','quadriceps'),
  ('NUTRIENT','creatine','creatine'),      ('CONDITION','type 2 diabetes','T2DM'),
  ('OUTCOME','muscle hypertrophy','cross-sectional area'), ('OUTCOME','muscle hypertrophy','hypertrophy');
  -- ... 331 synonyms for 113 canonical entities (full list in curate.py VOCAB)

CREATE OR REPLACE TABLE MENTIONS_RULE AS
SELECT a.pmid, d.entity_type, d.entity_name, 'dictionary' AS method,
       REGEXP_COUNT(a.abstract, '\\b' || d.synonym || '\\b', 1, 'i') AS mention_count
FROM ABSTRACTS a
JOIN ENTITY_DICTIONARY d
  ON REGEXP_LIKE(a.title || ' ' || a.abstract, '.*\\b' || d.synonym || '\\b.*', 'is');

-- 3b. LLM extraction with Cortex AI_EXTRACT (catches entities the dictionary misses)
CREATE OR REPLACE TABLE MENTIONS_LLM AS
WITH llm AS (
  SELECT pmid,
         AI_EXTRACT(
           text => title || '. ' || abstract,
           responseFormat => {
             'schema': {'type': 'object', 'properties': {
               'EXERCISE':   {'description': 'Exercises or training methods studied', 'type': 'array'},
               'MUSCLE':     {'description': 'Muscles or body structures mentioned',   'type': 'array'},
               'NUTRIENT':   {'description': 'Nutrients or supplements',               'type': 'array'},
               'CONDITION':  {'description': 'Diseases or health conditions',          'type': 'array'},
               'OUTCOME':    {'description': 'Measured outcomes (e.g. strength, blood pressure)', 'type': 'array'},
               'POPULATION': {'description': 'Study population (e.g. older adults, athletes)',    'type': 'array'}}}}
         ):response AS j
  FROM ABSTRACTS)
SELECT llm.pmid, t.key AS entity_type, LOWER(e.value::STRING) AS entity_name,
       'cortex_ai_extract' AS method, 1 AS mention_count
FROM llm, LATERAL FLATTEN(input => llm.j) t, LATERAL FLATTEN(input => t.value) e;

-- 3c. Finding direction, sentiment and keywords per abstract (Cortex AI functions)
CREATE OR REPLACE TABLE ABSTRACT_INSIGHTS AS
SELECT pmid,
       AI_CLASSIFY(conclusion, ['positive effect', 'no significant effect',
                                'inconclusive or mixed evidence', 'harmful effect']):labels[0]::STRING AS finding,
       AI_SENTIMENT(conclusion)                                                         AS sentiment,
       AI_COMPLETE('mistral-large2',
           'List the 5 most important keyphrases, comma-separated, no other text: ' || abstract) AS keywords
FROM ABSTRACTS;

-- ---------------------------------------------------------------------------
-- 4. STORE: curated entity + bridge tables (input to Databricks and Neo4j)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE ENTITIES AS
SELECT 'E' || LPAD(ROW_NUMBER() OVER (ORDER BY entity_type, COUNT(DISTINCT pmid) DESC), 4, '0') AS entity_id,
       entity_type, entity_name,
       COUNT(DISTINCT pmid) AS n_abstracts,
       SUM(mention_count)   AS n_mentions,
       LISTAGG(DISTINCT method, ';') AS methods
FROM (SELECT * FROM MENTIONS_RULE UNION ALL SELECT * FROM MENTIONS_LLM)
GROUP BY entity_type, entity_name;

CREATE OR REPLACE TABLE ABSTRACT_ENTITIES AS
SELECT m.pmid, e.entity_id, m.method, SUM(m.mention_count) AS mention_count
FROM (SELECT * FROM MENTIONS_RULE UNION ALL SELECT * FROM MENTIONS_LLM) m
JOIN ENTITIES e USING (entity_type, entity_name)
GROUP BY m.pmid, e.entity_id, m.method;

-- ---------------------------------------------------------------------------
-- 5. AUTOMATE: process only NEW abstracts as they arrive (Stream + Task)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE STREAM RAW.ABSTRACTS_STREAM ON TABLE RAW.RAW_ABSTRACTS;

CREATE OR REPLACE TASK CURATE_NEW_ABSTRACTS
  WAREHOUSE = KL_WH
  SCHEDULE  = '60 MINUTE'
  WHEN SYSTEM$STREAM_HAS_DATA('RAW.ABSTRACTS_STREAM')
AS
  INSERT INTO ABSTRACTS (pmid, title, year, journal, abstract, conclusion)
  SELECT doc:pmid, doc:title, doc:year, doc:journal, doc:abstract,
         COALESCE(REGEXP_SUBSTR(doc:abstract::STRING, 'CONCLUSIONS?:\\s*(.*)', 1, 1, 'e', 1), doc:abstract)
  FROM RAW.ABSTRACTS_STREAM;
  -- (downstream extraction steps 3a-4 chained as child tasks with AFTER CURATE_NEW_ABSTRACTS)

ALTER TASK CURATE_NEW_ABSTRACTS RESUME;

-- ---------------------------------------------------------------------------
-- 6. Example query on curated data: most-studied outcomes for HIIT
-- ---------------------------------------------------------------------------
SELECT o.entity_name AS outcome, COUNT(DISTINCT ae1.pmid) AS n_studies
FROM ABSTRACT_ENTITIES ae1
JOIN ENTITIES t ON ae1.entity_id = t.entity_id AND t.entity_name = 'HIIT'
JOIN ABSTRACT_ENTITIES ae2 ON ae1.pmid = ae2.pmid
JOIN ENTITIES o ON ae2.entity_id = o.entity_id AND o.entity_type = 'OUTCOME'
GROUP BY o.entity_name ORDER BY n_studies DESC LIMIT 10;
