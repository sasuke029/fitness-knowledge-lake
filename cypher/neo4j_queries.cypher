// =============================================================================
// Fitness & Health Knowledge Lake - Neo4j graph prototype - Fitness & Health Knowledge Lake
// Run each block separately in Neo4j Browser / Query (after python load_neo4j.py)
// Screenshot tag  [SS]  = worth a screenshot for the report
// =============================================================================


// ---------------------------------------------------------------- 1. THE GRAPH
// [SS] 1a. Schema of the knowledge graph (node labels + relationship types)
CALL db.schema.visualization();

// 1b. Size of the graph by node label
MATCH (n) UNWIND labels(n) AS label
RETURN label, count(*) AS nodes ORDER BY nodes DESC;

// 1c. Relationship counts
MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS n ORDER BY n DESC;

// [SS] 1d. Visual: the "squat" neighbourhood (exercise variants, muscles, studies, Wikidata)
MATCH (sq:Exercise {name: 'squat'})
OPTIONAL MATCH (v:ExerciseVariant)-[:VARIANT_OF]->(sq)
WITH sq, collect(v)[0..8] AS variants
UNWIND variants AS v
MATCH (v)-[t:TARGETS {role: 'primary'}]->(m:Muscle)
OPTIONAL MATCH (sq)-[w:INSTANCE_OF|SUBCLASS_OF]->(c)
RETURN sq, v, t, m, w, c;


// ------------------------------------------------- 2. GRAPH DATA SCIENCE (GDS)
// 2a. Project an in-memory graph: entities linked by co-occurrence (weighted)
//     + knowledge from Wikidata between our own entities
CALL gds.graph.drop('entityGraph', false);
CALL gds.graph.project(
  'entityGraph',
  'Entity',
  {
    CO_OCCURS_WITH: {orientation: 'UNDIRECTED', properties: 'weight'},
    RISK_FACTOR:    {orientation: 'UNDIRECTED', properties: 'weight'},
    SYMPTOM:        {orientation: 'UNDIRECTED', properties: 'weight'},
    SUBCLASS_OF:    {orientation: 'UNDIRECTED', properties: 'weight'},
    HAS_PART:       {orientation: 'UNDIRECTED', properties: 'weight'},
    PART_OF:        {orientation: 'UNDIRECTED', properties: 'weight'},
    INSTANCE_OF:    {orientation: 'UNDIRECTED', properties: 'weight'},
    HAS_CAUSE:      {orientation: 'UNDIRECTED', properties: 'weight'}
  }
) YIELD graphName, nodeCount, relationshipCount;

// [SS] 2b. PageRank: the most central concepts in fitness & health research
CALL gds.pageRank.stream('entityGraph', {relationshipWeightProperty: 'weight', dampingFactor: 0.85})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS e, score
RETURN e.name AS entity, e.entity_type AS type, round(score, 3) AS pagerank, e.n_abstracts AS studies
ORDER BY pagerank DESC LIMIT 15;

// 2c. write PageRank back as a node property (used by search, dashboard and GenAI)
CALL gds.pageRank.write('entityGraph', {relationshipWeightProperty: 'weight', writeProperty: 'pagerank'})
YIELD nodePropertiesWritten, ranIterations;

// [SS] 2d. Louvain community detection: research themes that cluster together
CALL gds.louvain.write('entityGraph', {relationshipWeightProperty: 'weight', writeProperty: 'community'})
YIELD communityCount, modularity;

MATCH (e:Entity) WHERE e.community IS NOT NULL
WITH e.community AS community, e ORDER BY e.pagerank DESC
WITH community, collect(e.name) AS members, count(*) AS size
WHERE size >= 3
RETURN community, size, members[0..10] AS top_members ORDER BY size DESC;

// 2e. Betweenness centrality: "bridge" concepts that connect different themes
CALL gds.betweenness.stream('entityGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS entity, round(score, 1) AS betweenness
ORDER BY betweenness DESC LIMIT 10;


// ------------------------------------------------------ 3. KNOWLEDGE QUERIES
// [SS] 3a. Evidence table: which training types improve which outcomes, and how consistently?
MATCH (t:TrainingType)<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(o:Outcome)
WITH t.name AS training, o.name AS outcome, count(s) AS studies,
     sum(CASE s.finding WHEN 'positive' THEN 1 ELSE 0 END) AS positive
WHERE studies >= 5
RETURN training, outcome, studies, positive,
       round(100.0 * positive / studies) AS pct_positive
ORDER BY studies DESC LIMIT 15;

// [SS] 3b. Multi-hop: exercises studied for sarcopenia, with the muscles they train
MATCH (c:Condition {name: 'sarcopenia'})<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(x:Entity)
WHERE x:Exercise OR x:TrainingType
WITH x, count(DISTINCT s) AS studies
OPTIONAL MATCH (v:ExerciseVariant)-[:VARIANT_OF]->(x)
OPTIONAL MATCH (v)-[:TARGETS {role: 'primary'}]->(m:Muscle)
RETURN x.name AS exercise_or_training, studies, collect(DISTINCT m.name)[0..5] AS muscles_trained
ORDER BY studies DESC;

// 3c. Cross-source query: high-protein foods + nutrients researched for muscle hypertrophy
MATCH (h:Outcome {name: 'muscle hypertrophy'})<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(n:Nutrient)
WITH n, count(DISTINCT s) AS studies
OPTIONAL MATCH (f:Food)-[c:CONTAINS]->(n) WHERE c.weight >= 15
RETURN n.name AS nutrient, studies,
       collect(f.name + ' (' + toString(c.weight) + ' g)')[0..5] AS rich_foods
ORDER BY studies DESC;

// [SS] 3d. Knowledge added by Wikidata: obesity is a RISK_FACTOR of type 2 diabetes,
//      so studies on exercise for obesity become relevant evidence for diabetes prevention
MATCH (d:Condition {name: 'type 2 diabetes'})-[:RISK_FACTOR]->(o:Condition)
MATCH (o)<-[:MENTIONS]-(s:Study)-[:MENTIONS]->(t:TrainingType)
WHERE s.finding = 'positive'
RETURN d.name AS condition, o.name AS via_risk_factor, t.name AS training,
       count(s) AS positive_studies ORDER BY positive_studies DESC;

// 3e. Shortest path: how is creatine connected to type 2 diabetes?
MATCH p = shortestPath((a:Entity {name: 'creatine'})-[*..6]-(b:Entity {name: 'type 2 diabetes'}))
RETURN p;

// 3f. Exercise recommender: beginner, body-weight exercises for the glutes
MATCH (v:ExerciseVariant {level: 'beginner', equipment: 'body only'})-[:TARGETS {role: 'primary'}]->(m:Muscle {name: 'glutes'})
RETURN v.name AS exercise, v.mechanic AS mechanic LIMIT 10;
