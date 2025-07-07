# Use the official Python base image
FROM python:3.8-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev pulseaudio alsa-utils libhdf5-dev libsndfile1 python3-setuptools python3-wheel python3-pip && \
    apt-get clean
#################################################################################################################
#RUN apt-get update &&
 RUN apt-get install -y libpulse0
##########################################################################
# Set the working directory
WORKDIR /app

# Set environment variable for PulseAudio server
ENV PULSE_SERVER=unix:/run/user/1002/pulse/native

COPY h5py-3.8.0 /app/h5py-3.8.0
RUN pip install /app/h5py-3.8.0
Run pip install sounddevice 
# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install pip packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run the application
CMD ["python", "main.py"]

