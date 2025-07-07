import os
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import threading
#from infuxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WriteOptions
import datetime

# Configuration from environment variables
"""influxdb_url = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
token = os.getenv("INFLUXDB_TOKEN","T6lW1p5i8FWe_eZkbB3FuwKLqA_3I5NsTuYN0G3ecDeuEoAf-0cH7HllioCvi6f4zZzl-PGIJLI3UbyoYitP0w==") 
org = os.getenv("INFLUXDB_ORG", "dbsense")
bucket = os.getenv("INFLUXDB_BUCKET", "sound_bucket")
"""

influxdb_url = "http://influxdb:8086"
token = "T6lW1p5i8FWe_eZkbB3FuwKLqA_3I5NsTuYN0G3ecDeuEoAf-0cH7HllioCvi6f4zZzl-PGIJLI3UbyoYitP0w=="
org = "dbsense"
bucket = "sound_bucket"

# Create InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

# Load the YAMNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the class map
class_map_path = 'models/research/audioset/yamnet/yamnet_class_map.csv'  
class_names = np.genfromtxt(class_map_path, delimiter=',', dtype=str, usecols=2, skip_header=1)

# Define audio recording parameters
sample_rate = 16000
duration = 2
interval = 1

def preprocess_audio(audio):
    audio = np.mean(audio, axis=1)  # Convert to mono
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    return audio

def predict_sound(data):
    waveform = preprocess_audio(data)
    waveform = np.reshape(waveform, [-1])  # Reshape to (None,)
    scores, embeddings, spectrogram = model(waveform)
    top_indices = np.argsort(scores.numpy()[0])[::-1][:3]
    top_classes = class_names[top_indices]
    top_scores = scores.numpy()[0][top_indices]
    
    current_time = datetime.datetime.utcnow()

    # Send predictions to InfluxDB
    for i in range(3):
        point = Point("sound_classification")\
            .tag("class", top_classes[i])\
            .field("probability", float(top_scores[i]))\
            .time(current_time)  
        write_api.write(bucket=bucket, org=org, record=point)

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

