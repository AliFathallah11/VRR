import os
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import threading
#from infuxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WriteOptions
import datetime
import librosa 
#bjeh rabi ekhdem v3
# Configuration from environment variables
"""influxdb_url = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
token = os.getenv("INFLUXDB_TOKEN","T6lW1p5i8FWe_eZkbB3FuwKLqA_3I5NsTuYN0G3ecDeuEoAf-0cH7HllioCvi6f4zZzl-PGIJLI3UbyoYitP0w==") 
org = os.getenv("INFLUXDB_ORG", "dbsense")
bucket = os.getenv("INFLUXDB_BUCKET", "sound_bucket")
"""

influxdb_url = "http://192.168.1.153:8086"
token = "NVBCL5V9EyvXkNlzBvT1dXWVnWeHR-N5b10pJ6d_9s8VnFTC6ZI2RGN7sUodbWnoeTNnOlgKtHKfoFQZoxT4Gg=="
org = "dbsense"
bucket = "sound_bucket"

# Create InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

from joblib import load
model_path = "lightgbm_model.pkl"
encoder_path = "label_encoder.pkl"

model = load(model_path)
label_encoder = load(encoder_path)


# Define audio recording parameters
sample_rate = 16000
duration = 2
interval = 1

def preprocess_audio(audio):
    audio = np.mean(audio, axis=1)  # Convert to mono
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    return audio

def extract_features(audio, sr=16000, n_mels=64):
    audio = np.mean(audio, axis=1)
    audio = librosa.util.fix_length(audio, size=sr * duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    features = np.mean(mel_spec, axis=1)
    return features.reshape(1, -1)

def predict_sound(data):
    features = extract_features(data, sr=sample_rate)
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    current_time = datetime.datetime.utcnow()
    point = Point("sound_classification")\
        .tag("class", label)\
        .field("probability", 1.0)\
        .time(current_time)
    write_api.write(bucket=bucket, org=org, record=point)
#    print(f"[{current_time}] Prediction: {label}")

def callback(indata, frames, time, status):
    if status:
        print(status)
    if frames:
        predict_sound(indata)
def start_stream():
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback, blocksize=sample_rate * duration):
        print("Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(int(interval * 1000))
        except KeyboardInterrupt:
            pass
# Start the audio stream in a separate thread to keep the main thread responsive
stream_thread = threading.Thread(target=start_stream)
stream_thread.start()

