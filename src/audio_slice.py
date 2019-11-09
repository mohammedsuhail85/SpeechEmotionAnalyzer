from pydub import AudioSegment
from pathlib import Path


TEMP_FOLDER = Path("temp")


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

        print(TEMP_FOLDER.name)
        for x in range(0, count-1):
            start = x*slicing_time*1000
            end = (x+1)*slicing_time*1000
            segment = sound[start:end]
            audio_file_name = str(TEMP_FOLDER.name) + "/seg_" + str(x) + "_" + str(session) + ".wav"
            list_audio.append(audio_file_name)
            segment.export(audio_file_name, format="wav")

        print("saved")

        response_list = []

        for x in list_audio:
            print(x)
            captured_emotion, duration = get_emotion(x, session)

            emotion = captured_emotion[0].split('_')[1]
            print(emotion)

            result = {
                "Emotion": emotion,
                "Duration": duration
            }
            response_list.append(result)

        print("Process completed")
        return response_list

    except Exception as ex:
        return str(ex), 400