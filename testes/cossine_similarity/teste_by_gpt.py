from configPy import EnvManager, Config
from transcription_processing.transcricao.findPattern import TranscriptionParser
from llama_index.embeddings.ollama import OllamaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer

# ----------- Diretórios -----------
output_dir = Config.get_dir_output()
ata_dir = output_dir["ata_formatada_llm"]
audio_dir = output_dir["audio"]

audio_files = list(audio_dir.iter_files())
ata_files = [f for f in ata_dir.iter_files() if f.name in [a.name for a in audio_files]]

# Escolher par
audio_file = audio_files[1]
ata_file = ata_files[0]

with open(audio_file, "r", encoding="utf-8") as f:
    audio_text = f.read()

with open(ata_file, "r", encoding="utf-8") as f:
    ata_text = f.read()

# ----------- Parser de transcrição -----------
parser = TranscriptionParser(audio_text)
speech_texts = [f.text for f in parser.speeches]

# ----------- Tokenizer para pesos semânticos -----------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def count_tokens(text: str) -> int:
    return len(tokenizer.tokenize(text))


speech_tokens = np.array([count_tokens(t) for t in speech_texts])
total_tokens = speech_tokens.sum()

# ----------- Embeddings Ollama -----------
ollamenv = EnvManager.ollama()
embedder = OllamaEmbedding(
    ollamenv.EMBEDDING_MODEL, f"http://{ollamenv.BASE_OLLAMA}:{ollamenv.PORT_OLLAMA}"
)

# Embeds das falas
audio_embeds = embedder.get_text_embedding_batch(speech_texts)
audio_embeds = np.array(audio_embeds)

# ----------- Embedding da transcrição (média ponderada) -----------
weights = speech_tokens / total_tokens
transcription_vec = np.sum(audio_embeds * weights[:, None], axis=0).reshape(1, -1)

# ----------- Embedding da ATA por seções -----------
ata_sections = [sec for sec in ata_text.split("\n\n") if sec.strip()]

print("============== SESSÕES DAS ATAS ================")
print(ata_sections)
print("================================================")

ata_embeds = embedder.get_text_embedding_batch(ata_sections)
ata_embeds = np.array(ata_embeds)

# Média dos embeddings da ata
ata_vec = ata_embeds.mean(axis=0).reshape(1, -1)

# ----------- Similaridade final -----------
similarity = cosine_similarity(transcription_vec, ata_vec)[0][0]

# Normalizar opcional para intervalo [0,1]
similarity_norm = (similarity + 1) / 2

print("===================================")
print(f"Similaridade semântica (cos): {similarity:.4f}")
print(f"Similaridade normalizada:     {similarity_norm:.4f}")
print("===================================")
