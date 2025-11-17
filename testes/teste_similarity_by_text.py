from configPy import EnvManager, Config
from transcription_processing.transcricao.findPattern import TranscriptionParser
from llama_index.embeddings.ollama import OllamaEmbedding

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

# configuração do ollama
ollamenv = EnvManager.ollama()
base = f"http://{ollamenv.BASE_OLLAMA}:{ollamenv.PORT_OLLAMA}"

embedder = OllamaEmbedding(ollamenv.EMBEDDING_MODEL, base)

transcription = TranscriptionParser(audio_text)
