from configPy import EnvManager, Config
from transcription_processing.transcricao.findPattern import TranscriptionParser
from llama_index.embeddings.ollama import OllamaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
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

print("=====================")
print(f"Ata: {teste_ata.name}")
print(f"Audio: {teste_audio.name}")
print("===================")

with open(teste_audio, "r", encoding="utf-8") as audio:
    audio_text = audio.read()

with open(teste_ata, "r", encoding="utf-8") as ata:
    ata_text = ata.read()

# reconstruir transcrição apenas com o texto das falas


# configuração do ollama
ollamenv = EnvManager.ollama()
base = f"http://{ollamenv.BASE_OLLAMA}:{ollamenv.PORT_OLLAMA}"

embedder = OllamaEmbedding(ollamenv.EMBEDDING_MODEL, base)

text_audio_embed: List[Dict] = []
parser = TranscriptionParser(audio_text)

# texto de cada fala
speech_text = list(map(lambda x: x.text, parser.speeches))

# tamanho de cada fala
speech_length = np.array([len(text) for text in speech_text])

# total de caracteres
total_char = np.sum(speech_length)

# embed de cada fala
embeds_for_audio = [
    np.array(embed).reshape(1, -1)
    for embed in embedder.get_text_embedding_batch(speech_text)
]

# embed da ata
embed = embedder.get_text_embedding(ata_text)
ata_vec = np.array(embed).reshape(1, -1)

# lista de similaridades calculadas
# multiplicada pelo total de carateres
# da fala
cosim = []

# calcular similaridade de cosseno ponderado pelo tamnho de cada fala
for embed, length in zip(embeds_for_audio, speech_length):
    cosim.append(cosine_similarity(embed, ata_vec)[0][0] * length)

sim = np.sum(cosim) / total_char

print("==================================")
print(f"Similaridade semântica (coseno): {sim:.4f}")
print("==================================")
