from configPy import EnvManager
from llama_index.embeddings.ollama import OllamaEmbedding

ollamenv = EnvManager.ollama()

base = f"http://{ollamenv.BASE_OLLAMA}:{ollamenv.PORT_OLLAMA}"

embed_ollama = OllamaEmbedding(ollamenv.EMBEDDING_MODEL, base)

embeddings = embed_ollama.get_text_embedding_batch(
    ["This is a passage!", "This is another passage"], show_progress=True
)
print(f"Got vectors of length {len(embeddings[0])}")
print(embeddings[0][:10])
