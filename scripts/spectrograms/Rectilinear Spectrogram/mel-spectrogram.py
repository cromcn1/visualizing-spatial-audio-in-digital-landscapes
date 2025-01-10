#script to convert mono audio files to spectrograms.#
#this code is based on the article "Mel Spectrograms with Python and Librosa | Audio Feature Extraction" by Cloud & Data Science.#
#the article is available at https://clouddatascience.medium.com/mel-spectrograms-with-python-and-librosa-audio-feature-extraction-4ab18c14797c#

#import dependencies#
import librosa
import librosa.display
import IPython.display as ip
import matplotlib.pyplot as plt
import numpy as np

#set audio file path. this can be any path on your computer.#
audio_path = 'file path here'
ip.Audio(audio_path)

#load audio file#
y, sr = librosa.load(audio_path)

#plot spectrogram using librosa#
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

#draw spectrogram. in this script, the Y-axis is scaled logarithmically to represent frequencies in a manner similar to how we perceive them.#
plt.figure(figsize=(10,4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('A')
plt.show()