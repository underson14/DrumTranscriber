from DrumTranscriber import DrumTranscriber
import librosa
import pandas as pd

# Carregar amostras de áudio
samples, sr = librosa.load("/content/drive/MyDrive/dt/teste1.wav")

# Instanciar o objeto DrumTranscriber
transcriber = DrumTranscriber()

# Obter as previsões das probabilidades das classes
predictions = transcriber.predict(samples, sr)

# Exportar as previsões para um arquivo CSV
output_path = "/content/drive/MyDrive/dt/output.csv"
predictions.to_csv(output_path, index=False)