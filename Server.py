import librosa
import librosa.display
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from pydub import AudioSegment

from flask import Flask, request, jsonify
from flask_restful import Api
from werkzeug import secure_filename
import os
import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from werkzeug.exceptions import BadRequestKeyError
import warnings
warnings.filterwarnings('ignore')

lb = LabelEncoder()
file = open('lbsave.txt', 'rb')
lb = pickle.load(file)


UPLOAD_FOLDER = '/home/suhail/Desktop/SpeechEmotionAnalyzer/uploaded_files'
# UPLOAD_FOLDER = os.environ["UPLOAD_FOLDER"] if "UPLOAD_FOLDER" in os.environ else "./uploaded_files"
PORT = 5000
ALLOWED_EXTENTIONS = ['wav', 'mp4']

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cred = credentials.Certificate('./firebase_config.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


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


def get_emotion(audio_path, session):
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

        data, sampling_rate = librosa.load(audio_path)
        duration = str(len(data) / sampling_rate) + "sec"
        print(duration)
        # data, sampling_rate = librosa.load('hUY8DiQgUUg-0-30_2019-06-22_224559.797771.wav')

        # # get_ipython().run_line_magic('pylab', 'inline')
        # plt.figure(figsize=(15, 5))
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


@app.route('/audio/<session>/getemotion', methods=['POST'])
def upload_file(session):
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

                result = slice_audio(audio_path, session)

                db.collection('sessions').document(session).update({
                    "audio": {
                        "Voice Emotions": result
                    }
                })
                return jsonify({
                    "Session Id": session,
                    "Status": "Process completed and Firebasse updated",
                    "Emotions": result
                })

            else:
                return jsonify({'Error': 'Unsupported file format. Supports only .wav format'}), 400
        except BadRequestKeyError:
            return jsonify({'Error': "Missing Audio file, Required : form-data with .wav and "
                                     "key name 'file'"}), 400
        except Exception as e:
            e.with_traceback()
            return jsonify({'Error': "System Error"}), 409


def slice_audio(audio_path, session):
    try:
        # sound = AudioSegment.from_file_using_temporary_files("/home/suhail/Desktop/SpeechEmotionAnalyzer/test_vid.wav")
        sound = AudioSegment.from_file(audio_path)

        duration = sound.duration_seconds
        slicing_time = 4
        print(duration)

        count = (duration // slicing_time) + 1
        count = int(count)
        print(count)

        list_audio = []

        for x in range(0, count-1):
            start = x*slicing_time*1000
            end = (x+1)*slicing_time*1000
            segment = sound[start:end]
            audio_file_name = "/home/suhail/Desktop/SpeechEmotionAnalyzer/temp/seg"+str(x)+".wav"
            list_audio.append(audio_file_name)
            segment.export(audio_file_name, format="wav")

        print("saved")

        url_emotion = "http://127.0.0.1:5000/audio/getemotion"

        response_list = []

        for x in list_audio:
            print(x)
            # multipart = {'file': ('sample.wav', open(x, 'rb'), 'audio/x-wav', {'Expires': '0'})}
            # print("Making Request")
            #
            # response = requests.post(url_emotion, files=multipart)
            # if response.status_code == 200:
            #     response_list.append(response.content)
            captured_emotion, duration = get_emotion(x, session)

            emotion = captured_emotion[0].split('_')[1]
            print(emotion)

            result = {
                "Emotion": emotion,
                "Duration": duration
            }
            response_list.append(result)


        print("Process completed")
        # for x in response_list:
        #     print(x)
        return response_list

    except Exception as ex:
        return str(ex), 400


if __name__ == '__main__':
    try:
        print("Emotion Recognizer started on PORT "+str(PORT))
        load_model()
        app.run(port=PORT, debug=True)

    except Exception as e:
        e.with_traceback()
