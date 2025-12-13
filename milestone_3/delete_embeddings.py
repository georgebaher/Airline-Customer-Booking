from run_queries import run_query

# Delete all embeddings
run_query("""
MATCH (j:Journey) 
REMOVE 
    j.embedding, j.embedding_description, 
    j.simple_embedding, j.simple_embedding_description,
    j.embedding_minilm_rich, j.description_rich,
    j.embedding_minilm_simple, j.description_simple,
    j.embedding_mpnet_rich,
    j.embedding_mpnet_simple
""")