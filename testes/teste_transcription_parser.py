from configPy import Config
from transcription_processing.transcricao.findPattern import TranscriptionParser

# diretório dos audios e atas
output_dir = Config.get_dir_output()
ata_dir = output_dir["ata"]
audio_dir = output_dir["audio"]

# arquivos
audio_files = list(audio_dir.iter_files())
# escolhe o par
teste_audio = audio_files[0]

with open(teste_audio, "r", encoding="utf-8") as audio:
    audio_text = audio.read()


transcription = TranscriptionParser(audio_text)
for speech in transcription.speeches:
    print(speech)
