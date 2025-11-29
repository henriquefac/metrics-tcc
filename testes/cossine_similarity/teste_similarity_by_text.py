from configPy import EnvManager, Config
from transcription_processing.transcricao.findPattern import TranscriptionParser
from llama_index.embeddings.ollama import OllamaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# diretório dos audios e atas
output_dir = Config.get_dir_output()
ata_dir = output_dir["ata"]
audio_dir = output_dir["audio"]

# arquivos
audio_files = list(audio_dir.iter_files())
ata_files = [
    file
    for file in list(ata_dir.iter_files())
    if file.name in list(map(lambda x: x.name, audio_files))
]

# escolhe o par
teste_audio = audio_files[0]
teste_ata = ata_files[0]

with open(teste_audio, "r", encoding="utf-8") as audio:
    audio_text = audio.read()

with open(teste_ata, "r", encoding="utf-8") as ata:
    ata_text = ata.read()

# reconstruir transcrição apenas com o texto das falas

new_text_audio = ""
parser = TranscriptionParser(audio_text)

for speech in parser.speeches:
    new_text_audio += speech.text
    new_text_audio += "\n\n"


# configuração do ollama
ollamenv = EnvManager.ollama()
base = f"http://{ollamenv.BASE_OLLAMA}:{ollamenv.PORT_OLLAMA}"

embedder = OllamaEmbedding(ollamenv.EMBEDDING_MODEL, base)

embeds = embedder.get_text_embedding_batch([new_text_audio, ata_text])

audio_vec = np.array(embeds[0]).reshape(1, -1)
ata_vec = np.array(embeds[1]).reshape(1, -1)

# similaridade do cosseno
sim = cosine_similarity(audio_vec, ata_vec)[0][0]

print("==================================")
print(f"Similaridade semântica (coseno): {sim:.4f}")
print("==================================")
