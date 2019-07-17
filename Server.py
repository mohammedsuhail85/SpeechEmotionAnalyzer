import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
import librosa

file = open('lbsave.txt', 'rb')
lb = pickle.load(file)


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

# data, sampling_rate = librosa.load('output10.wav')
data, sampling_rate = librosa.load('hUY8DiQgUUg-0-30_2019-06-22_224559.797771.wav')


# get_ipython().run_line_magic('pylab', 'inline')
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)

featurelive = mfccs

livedf2 = featurelive

livedf2 = pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

# livedf2

twodim = np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

# livepreds

livepreds1 = livepreds.argmax(axis=1)


liveabc = livepreds1.astype(int).flatten()

livepredictions = (lb.inverse_transform(liveabc))
livepredictions

print(livepredictions)


