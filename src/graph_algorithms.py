"""
Fitness & Health Knowledge Lake - graph algorithms on the Neo4j Knowledge Lake graph
Backup for when the Neo4j Graph Data Science (GDS) plugin is not available.

Reads the entity graph from Neo4j (same projection as neo4j_queries.cypher block 2a),
runs PageRank, Louvain community detection and betweenness centrality with NetworkX,
and writes the scores back to Neo4j as node properties: pagerank, community, betweenness.
After this, the remaining Cypher blocks (2d community table, 3a-3f) work unchanged.

Run:  pip install neo4j networkx scipy
      python graph_algorithms.py
"""

import getpass

import networkx as nx
from neo4j import GraphDatabase

URI, USER = "neo4j://127.0.0.1:7687", "neo4j"
REL_TYPES = ["CO_OCCURS_WITH", "RISK_FACTOR", "SYMPTOM", "SUBCLASS_OF",
             "HAS_PART", "PART_OF", "INSTANCE_OF", "HAS_CAUSE"]


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, getpass.getpass("Neo4j password: ")))
    with driver.session() as s:
        nodes = s.run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, "
                      "e.entity_type AS type, e.n_abstracts AS studies").data()
        edges = s.run("""MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) IN $types
                         RETURN a.id AS a, b.id AS b, coalesce(r.weight, 1.0) AS w""", types=REL_TYPES).data()

        # Undirected weighted projection (same as GDS orientation: 'UNDIRECTED')
        G = nx.Graph()
        for n in nodes:
            G.add_node(n["id"], **n)
        for e in edges:
            w = G[e["a"]][e["b"]]["weight"] + e["w"] if G.has_edge(e["a"], e["b"]) else e["w"]
            G.add_edge(e["a"], e["b"], weight=w)
        print(f"Projected graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} relationships")

        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        comms = nx.community.louvain_communities(G, weight="weight", seed=42)
        community = {n: i for i, c in enumerate(comms) for n in c}
        modularity = nx.community.modularity(G, comms, weight="weight")
        btw = nx.betweenness_centrality(G)          # unweighted, like the GDS default

        rows = [{"id": n, "pagerank": round(pr[n], 5), "community": community[n],
                 "betweenness": round(btw[n], 5)} for n in G.nodes]
        s.run("""UNWIND $rows AS r MATCH (e:Entity {id: r.id})
                 SET e.pagerank = r.pagerank, e.community = r.community, e.betweenness = r.betweenness""",
              rows=rows).consume()

    name = {n["id"]: (n["name"], n["type"], n["studies"]) for n in nodes}
    print("\nTop 15 entities by PageRank")
    print(f"{'entity':28} {'type':15} {'pagerank':>9} {'studies':>8}")
    for n, score in sorted(pr.items(), key=lambda x: -x[1])[:15]:
        nm, t, st = name[n]
        print(f"{nm:28} {t:15} {score:9.4f} {st or 0:8}")

    print(f"\nLouvain: {len(comms)} communities, modularity = {modularity:.3f}")
    for i, c in sorted(enumerate(comms), key=lambda x: -len(x[1])):
        if len(c) >= 3:
            members = sorted(c, key=lambda n: -pr[n])
            print(f"  community {i:2} ({len(c):2}): " + ", ".join(name[m][0] for m in members[:10]))

    print("\nTop 10 bridge concepts (betweenness)")
    for n, score in sorted(btw.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name[n][0]:28} {score:.3f}")
    print("\nScores written to Neo4j as e.pagerank, e.community, e.betweenness")
    driver.close()


if __name__ == "__main__":
    main()
