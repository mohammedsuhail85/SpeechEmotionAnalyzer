import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

file = open('lbsave.txt', 'rb')
lb = pickle.load(file)


# In[2]:


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")


import librosa
data, sampling_rate = librosa.load('output10.wav')


# In[4]:


# get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


# In[5]:


#livedf= pd.DataFrame(columns=['feature'])

X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)

featurelive = mfccs

livedf2 = featurelive

livedf2 = pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

livedf2

twodim = np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

livepreds

livepreds1 = livepreds.argmax(axis=1)


liveabc = livepreds1.astype(int).flatten()

livepredictions = (lb.inverse_transform(liveabc))
livepredictions

print(livepredictions)


