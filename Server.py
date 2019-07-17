import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
import pickle
import pandas as pd


from flask import Flask, request, flash, redirect, url_for, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename
import os
import datetime


from werkzeug.exceptions import BadRequestKeyError

lb = LabelEncoder()
file = open('lbsave.txt', 'rb')
lb = pickle.load(file)


UPLOAD_FOLDER = '/home/suhail/Desktop/SpeechEmotionAnalyzer/uploaded_files'
ALLOWED_EXTENTIONS = ['wav']

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_model():
    global loaded_model
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    loaded_model._make_predict_function()
    # loaded_model._make_predict_function()
    print("Loaded model from disk")


def get_emotion(audio_path):
    try:
        # from keras.models import model_from_json
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
        # loaded_model._make_predict_function()
        # # loaded_model._make_predict_function()
        # print("Loaded model from disk")

        print(audio_path)

        data, sampling_rate = librosa.load(audio_path)
        duration = str(len(data) / sampling_rate) + "sec"
        print(duration)
        # data, sampling_rate = librosa.load('hUY8DiQgUUg-0-30_2019-06-22_224559.797771.wav')

        # # get_ipython().run_line_magic('pylab', 'inline')
        plt.figure(figsize=(15, 5))
        librosa.display.waveplot(data, sr=sampling_rate)

        # livedf= pd.DataFrame(columns=['feature'])
        X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

        featurelive = mfccs

        livedf2 = featurelive
        livedf2 = pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T

        twodim = np.expand_dims(livedf2, axis=2)

        livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()

        livepredictions = (lb.inverse_transform(liveabc))
        print(livepredictions)

        # K.clear_session()

        return livepredictions, duration

    except Exception as ex:
        ex.with_traceback()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS


@app.route('/audio/test', methods=['GET', 'POST'])
def test_api():
    if request.method == 'GET':
        return jsonify({
            "Message": "Success"
        })


@app.route('/audio/getemotion', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('request created')
        try:
            print(request)
            f = request.files['file']

            filename = f.filename.rsplit('.', 1)[0]
            current_time = str(datetime.datetime.now())

            if f and allowed_file(f.filename):
                filename_new = secure_filename(
                    filename + '_' + current_time + ".wav")
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_new))

                audio_path = ("uploaded_files/" + filename_new)

                captured_emotion, duration = get_emotion(audio_path)
                print(captured_emotion)

                return jsonify(({
                    "Audio name": filename_new,
                    "Captured Emotion": captured_emotion[0],
                    "Duration": duration
                }))
            else:
                return jsonify({'Error': 'Unsupported file format. Supports only .wav format'}), 400
        except BadRequestKeyError:
            return jsonify({'Error': "Missing Audio file, Required : form-data with .wav and "
                                     "key name 'file'"}), 400
        except Exception as e:
            e.with_traceback()
            return jsonify({'Error': "System Error"}), 409


if __name__ == '__main__':
    try:
        load_model()
        app.run(debug=True)

    except Exception as e:
        e.with_traceback()
