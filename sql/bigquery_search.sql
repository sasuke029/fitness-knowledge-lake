-- =============================================================================
-- Fitness & Health Knowledge Lake - Federated search over heterogeneous data in BigQuery
-- Fitness & Health Knowledge Lake
--
--   knowledge_lake.foods          STRUCTURED       (CSV)    flat columns
--   knowledge_lake.exercises      SEMI-STRUCTURED  (JSONL)  nested arrays
--   knowledge_lake.abstracts      UNSTRUCTURED     (JSONL)  free text + embedding
--   knowledge_lake.entities       KNOWLEDGE GRAPH  (JSONL)  Neo4j PageRank / community
--   knowledge_lake.demo_queries   embedded questions for semantic search
--
-- Run each block separately in the BigQuery console.  [SS] = screenshot for the report
-- =============================================================================


-- ----------------------------------------------------------------------------
-- 0. Look at the three data shapes
-- ----------------------------------------------------------------------------
-- [SS] 0a. Semi-structured: query INSIDE the nested arrays with UNNEST
SELECT muscle, COUNT(*) AS exercises,
       COUNTIF(level = 'beginner') AS beginner
FROM knowledge_lake.exercises, UNNEST(primary_muscles) AS muscle
GROUP BY muscle ORDER BY exercises DESC;

-- 0b. Full-text index on the unstructured text (BigQuery search index; skip if the sandbox refuses it,
--     SEARCH() still works without an index)
CREATE SEARCH INDEX IF NOT EXISTS abstracts_text_idx
ON knowledge_lake.abstracts (title, abstract, conclusion);


-- ----------------------------------------------------------------------------
-- 1. [SS] FEDERATED KEYWORD SEARCH: one query, three data types, one ranked result
--    SEARCH() works on plain columns, on whole rows (incl. nested arrays) and on text.
-- ----------------------------------------------------------------------------
DECLARE q STRING DEFAULT 'squat quadriceps';

WITH
terms AS (SELECT SPLIT(LOWER(q), ' ') AS t),

text_hits AS (          -- unstructured: research abstracts
  SELECT 'abstract (unstructured)' AS source, CAST(a.pmid AS STRING) AS item_id, a.title AS title,
         CONCAT('finding: ', a.finding, ' | entities: ', ARRAY_TO_STRING(a.entities, ', ')) AS detail,
         (SELECT SUM(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(LOWER(a.title || ' ' || a.abstract), x)))
          FROM terms, UNNEST(t) AS x) AS score
  FROM knowledge_lake.abstracts a
  WHERE SEARCH((a.title, a.abstract), q)
),

json_hits AS (          -- semi-structured: search the WHOLE nested record (name, arrays, instructions)
  SELECT 'exercise (semi-structured)' AS source, e.id, e.name,
         CONCAT(e.level, ' | ', e.equipment, ' | primary: ', ARRAY_TO_STRING(e.primary_muscles, ', ')),
         (SELECT SUM(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(LOWER(TO_JSON_STRING(e)), x)))
          FROM terms, UNNEST(t) AS x)
  FROM knowledge_lake.exercises e
  -- whole-row SEARCH(e, q) did not reach the nested arrays, so match the serialised JSON record
  WHERE (SELECT LOGICAL_AND(REGEXP_CONTAINS(LOWER(TO_JSON_STRING(e)), x)) FROM terms, UNNEST(t) AS x)
),

table_hits AS (         -- structured: nutrients linked to leg-training/hypertrophy -> protein-rich foods
  SELECT 'food (structured)', CAST(f.fdc_id AS STRING), f.food_name,
         FORMAT('%.1f g protein / 100 g, %.0f kcal', f.protein_g, f.energy_kcal),
         f.protein_g / 10
  FROM knowledge_lake.foods f
  WHERE f.high_protein
)

SELECT source, item_id, SUBSTR(title, 1, 45) AS title, ROUND(score, 1) AS score FROM (
  SELECT * FROM text_hits UNION ALL SELECT * FROM json_hits UNION ALL SELECT * FROM table_hits
)
WHERE TRUE
QUALIFY ROW_NUMBER() OVER (PARTITION BY source ORDER BY score DESC) <= 5
ORDER BY source, score DESC;


-- ----------------------------------------------------------------------------
-- 2. [SS] SEMANTIC SEARCH with VECTOR_SEARCH (meaning, not keywords)
--    q1 = "best exercises to build leg muscle strength in older adults"
-- ----------------------------------------------------------------------------
SELECT ROUND(1 - distance, 3) AS similarity,
       base.title, base.finding, ARRAY_TO_STRING(base.entities, ', ') AS entities
FROM VECTOR_SEARCH(
       TABLE knowledge_lake.abstracts, 'embedding',
       (SELECT embedding FROM knowledge_lake.demo_queries WHERE query_id = 'q1'), 'embedding',
       top_k => 8, distance_type => 'COSINE')
ORDER BY similarity DESC;


-- ----------------------------------------------------------------------------
-- 3. [SS] HYBRID, KNOWLEDGE-GRAPH-BOOSTED FEDERATED SEARCH ("answer a question")
--    question -> semantic top studies (unstructured)
--             -> entities in those studies, weighted by Neo4j PageRank (knowledge graph)
--             -> matching exercises via nested muscle arrays (semi-structured)
--             -> protein-rich foods if protein is relevant (structured)
-- ----------------------------------------------------------------------------
DECLARE qid STRING DEFAULT 'q1';      -- try q2, q3, q4

WITH
sem AS (
  SELECT base.pmid, base.title, base.finding, base.entities, 1 - distance AS sim
  FROM VECTOR_SEARCH(TABLE knowledge_lake.abstracts, 'embedding',
         (SELECT embedding FROM knowledge_lake.demo_queries WHERE query_id = qid), 'embedding',
         top_k => 15, distance_type => 'COSINE')
),
pr AS (SELECT MAX(pagerank) AS max_pr FROM knowledge_lake.entities),
kg AS (            -- concepts behind the question, ranked by evidence x graph centrality
  SELECT e.name, e.type, e.community,
         SUM(s.sim) * (0.5 + 0.5 * IFNULL(e.pagerank, 0) / pr.max_pr) AS relevance
  FROM sem s, UNNEST(s.entities) AS ent
  JOIN knowledge_lake.entities e ON e.name = ent
  CROSS JOIN pr
  GROUP BY e.name, e.type, e.community, e.pagerank, pr.max_pr
),
studies AS (       -- studies re-ranked: similarity boosted by the centrality of their entities
  SELECT 'study' AS result_type, s.title AS result,
         CONCAT('finding: ', s.finding) AS evidence,
         s.sim * (1 + IFNULL(AVG(k.relevance), 0) / 10) AS score
  FROM sem s LEFT JOIN UNNEST(s.entities) AS ent
  LEFT JOIN kg k ON k.name = ent
  GROUP BY s.pmid, s.title, s.finding, s.sim
),
exercises AS (     -- semi-structured: exercises whose PRIMARY muscles are relevant MUSCLE concepts
  SELECT 'exercise' AS result_type, x.name AS result,
         CONCAT(x.level, ', ', x.equipment, ' | trains: ', STRING_AGG(m, ', ')) AS evidence,
         SUM(k.relevance) AS score
  FROM knowledge_lake.exercises x, UNNEST(x.primary_muscles) AS m
  JOIN kg k ON k.name = m AND k.type = 'MUSCLE'
  WHERE x.category = 'strength'
  GROUP BY x.name, x.level, x.equipment
),
foods AS (         -- structured: only if a protein-type nutrient is among the relevant concepts
  SELECT 'food' AS result_type, f.food_name AS result,
         FORMAT('%.1f g protein / 100 g (%.1f g per 100 kcal)', f.protein_g, f.protein_per_100kcal) AS evidence,
         f.protein_per_100kcal AS score
  FROM knowledge_lake.foods f
  WHERE f.high_protein
    AND EXISTS (SELECT 1 FROM kg WHERE type = 'NUTRIENT' AND name IN ('protein', 'whey protein', 'leucine', 'essential amino acids'))
),
concepts AS (
  SELECT 'concept' AS result_type, name AS result, CONCAT(type, ' | community ', CAST(community AS STRING)) AS evidence,
         relevance AS score FROM kg
)
SELECT result_type, SUBSTR(result, 1, 45) AS result, SUBSTR(evidence, 1, 55) AS evidence, ROUND(score, 3) AS score
FROM (
  SELECT * FROM concepts UNION ALL SELECT * FROM studies
  UNION ALL SELECT * FROM exercises UNION ALL SELECT * FROM foods
)
WHERE TRUE
QUALIFY ROW_NUMBER() OVER (PARTITION BY result_type ORDER BY score DESC) <= 5
ORDER BY CASE result_type WHEN 'concept' THEN 1 WHEN 'study' THEN 2 WHEN 'exercise' THEN 3 ELSE 4 END,
         score DESC;


-- ----------------------------------------------------------------------------
-- 4. [SS] EVALUATION: keyword vs semantic vs hybrid search (relevant results in top 10)
--    A study is relevant to a question only if it mentions BOTH concept groups,
--    e.g. q2 needs (HIIT) AND (blood pressure OR hypertension).
--    Hybrid = Reciprocal Rank Fusion (RRF) of semantic + expert-keyword rankings.
-- ----------------------------------------------------------------------------
WITH
grp AS (
  SELECT * FROM UNNEST([
    STRUCT('q1' AS query_id, ['muscle strength'] AS g1, ['older adults', 'sarcopenia'] AS g2),
    STRUCT('q2', ['HIIT', 'sprint interval training'], ['blood pressure', 'hypertension']),
    STRUCT('q3', ['creatine'], ['sprint', 'sprint performance', 'jump performance', 'countermovement jump', 'power', 'athletes']),
    STRUCT('q4', ['type 2 diabetes'], ['fat loss', 'obesity', 'body composition', 'BMI'])])
),
labels AS (        -- ground truth from the extracted entities
  SELECT g.query_id, a.pmid,
         EXISTS(SELECT 1 FROM UNNEST(a.entities) e WHERE e IN UNNEST(g.g1))
     AND EXISTS(SELECT 1 FROM UNNEST(a.entities) e WHERE e IN UNNEST(g.g2)) AS relevant
  FROM knowledge_lake.abstracts a CROSS JOIN grp g
),
semantic AS (
  SELECT query.query_id, base.pmid, ROW_NUMBER() OVER (PARTITION BY query.query_id ORDER BY distance) AS rk
  FROM VECTOR_SEARCH(TABLE knowledge_lake.abstracts, 'embedding',
         (SELECT query_id, embedding FROM knowledge_lake.demo_queries), 'embedding',
         top_k => 50, distance_type => 'COSINE')
),
kw_hits AS (       -- SEARCH() needs constant strings, so one branch per question
  SELECT 'keyword (natural question)' AS method, 'q1' AS query_id, pmid, title, abstract, 'best exercises to build leg muscle strength in older adults' AS kw
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'best exercises to build leg muscle strength in older adults')
  UNION ALL SELECT 'keyword (natural question)', 'q2', pmid, title, abstract, 'can high intensity interval training lower blood pressure'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'can high intensity interval training lower blood pressure')
  UNION ALL SELECT 'keyword (natural question)', 'q3', pmid, title, abstract, 'does creatine improve sprint and jump performance in athletes'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'does creatine improve sprint and jump performance in athletes')
  UNION ALL SELECT 'keyword (natural question)', 'q4', pmid, title, abstract, 'how to reduce body fat in people with type 2 diabetes'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'how to reduce body fat in people with type 2 diabetes')
  UNION ALL SELECT 'keyword (expert terms)', 'q1', pmid, title, abstract, 'strength older'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'strength older')
  UNION ALL SELECT 'keyword (expert terms)', 'q2', pmid, title, abstract, 'interval blood pressure'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'interval blood pressure')
  UNION ALL SELECT 'keyword (expert terms)', 'q3', pmid, title, abstract, 'creatine sprint jump'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'creatine sprint jump')
  UNION ALL SELECT 'keyword (expert terms)', 'q4', pmid, title, abstract, 'diabetes fat'
    FROM knowledge_lake.abstracts WHERE SEARCH((title, abstract), 'diabetes fat')
),
keyword AS (       -- rank keyword hits by term frequency
  SELECT method, query_id, pmid,
         ROW_NUMBER() OVER (PARTITION BY method, query_id ORDER BY
           (SELECT SUM(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(LOWER(title || ' ' || abstract), w)))
            FROM UNNEST(SPLIT(LOWER(kw), ' ')) AS w) DESC) AS rk
  FROM kw_hits
),
hybrid AS (        -- Reciprocal Rank Fusion: score = sum 1 / (60 + rank)
  SELECT query_id, pmid, ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY SUM(1 / (60 + rk)) DESC) AS rk
  FROM (SELECT query_id, pmid, rk FROM semantic
        UNION ALL SELECT query_id, pmid, rk FROM keyword WHERE method = 'keyword (expert terms)')
  GROUP BY query_id, pmid
),
top10 AS (
  SELECT method, query_id, pmid FROM keyword WHERE rk <= 10
  UNION ALL SELECT 'semantic (VECTOR_SEARCH)', query_id, pmid FROM semantic WHERE rk <= 10
  UNION ALL SELECT 'hybrid (RRF)', query_id, pmid FROM hybrid WHERE rk <= 10
),
grid AS (
  SELECT m AS method, g.query_id
  FROM UNNEST(['keyword (natural question)', 'keyword (expert terms)', 'semantic (VECTOR_SEARCH)', 'hybrid (RRF)']) AS m
  CROSS JOIN grp g
),
scored AS (
  SELECT gr.method, gr.query_id,
         COUNT(t.pmid) AS returned,
         COUNTIF(l.relevant) AS relevant_in_top10,
         (SELECT COUNTIF(relevant) FROM labels x WHERE x.query_id = gr.query_id) AS relevant_in_corpus
  FROM grid gr
  LEFT JOIN top10 t ON t.method = gr.method AND t.query_id = gr.query_id
  LEFT JOIN labels l ON l.query_id = t.query_id AND l.pmid = t.pmid
  GROUP BY gr.method, gr.query_id
)
SELECT * FROM (
SELECT method, query_id, returned, relevant_in_top10, relevant_in_corpus,
       ROUND(SAFE_DIVIDE(relevant_in_top10, 10), 2) AS precision_at_10,
       ROUND(SAFE_DIVIDE(relevant_in_top10, LEAST(10, relevant_in_corpus)), 2) AS recall_at_10
FROM scored
UNION ALL
SELECT method, 'ALL', SUM(returned), SUM(relevant_in_top10), SUM(relevant_in_corpus),
       ROUND(SUM(relevant_in_top10) / 40, 2),
       ROUND(SUM(relevant_in_top10) / SUM(LEAST(10, relevant_in_corpus)), 2)
FROM scored GROUP BY method
)
ORDER BY query_id = 'ALL', query_id, method;


-- ----------------------------------------------------------------------------
-- 5. (Optional) TRUE FEDERATION: query a Google Sheet live, without loading it
--    1) Upload bigquery/foods.csv to Google Sheets
--    2) BigQuery: knowledge_lake -> Create table -> Source: Drive -> paste the sheet URL
--       -> format Google Sheets -> table name foods_sheet_ext -> Table type: External -> auto-detect schema
-- ----------------------------------------------------------------------------
-- SELECT food_name, protein_g FROM knowledge_lake.foods_sheet_ext WHERE protein_g > 20 ORDER BY protein_g DESC;
