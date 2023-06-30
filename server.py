from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
import base64
import librosa
import numpy as np
from keras.models import load_model
from pydub import AudioSegment
from werkzeug.utils import secure_filename


app = Flask(__name__)
CORS(app)

# Define the classes
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'metro','siren', 'street_music']

# Load the model
model_path = 'model.h5'  
model = load_model(model_path)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    # check if the post request has the file part
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['audioFile']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # check if the file is a WAV or MP3 file
    if not file.filename.endswith('.wav') and not file.filename.endswith('.mp3'):
        return jsonify({'error': 'Invalid file type, only WAV or MP3 files are allowed'}), 400


    # save the file to disk
    filename = secure_filename(file.filename)
    file_path = os.path.join("audio_files", filename)
    file.save(file_path)

    # check if the file is MP3 and convert to WAV
    if file.filename.endswith('.mp3'):
        audio = AudioSegment.from_mp3(os.path.abspath(file_path))
        wav_path = os.path.join("audio_files", os.path.splitext(filename)[0] + '.wav')
        audio.export(wav_path, format='wav')
        file_path = wav_path

    # load the audio file
    audio_data, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=30)
    mfccs = np.mean(mfccs.T,axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], 1, 1)

    # Predict the class label
    pred = model.predict(mfccs)
    class_idx = np.argmax(pred[0])
    class_label = classes[class_idx]

    return jsonify({'result': class_label, 'filename': filename})

@app.route('/feedback', methods=['POST'])
def process_feedback():
    feedback = request.json.get('feedback')
    reason = request.json.get('reason')
    audio_file = request.json.get('audio_file')

    # Check if the feedback is valid
    if feedback not in ['like', 'dislike']:
        return jsonify({'error': 'Invalid feedback value'}), 400

    # Store the feedback alongside the audio file name
    feedback_data = {
        'audio_file': audio_file,
        'feedback': feedback,
        'reason': reason
    }

    # Save the feedback data to a CSV file
    csv_file = 'feedback_data.csv'

    fieldnames = ['audio_file', 'feedback', 'reason']  

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(feedback_data)

    return jsonify({'message': 'Feedback received'})

if __name__ == '__main__':
    app.run(debug=True)