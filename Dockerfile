# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /src

# Install git and any other dependencies
RUN apt-get update && apt-get install -y git python3-venv ffmpeg libsm6 libxext6

# Clone the repository and install the library
RUN git clone https://github.com/openvinotoolkit/anomalib.git
WORKDIR /src/anomalib
RUN pip install -e .
RUN anomalib install --option core -v
RUN pip install scikit-learn
RUN pip install openvino
RUN pip install dotenv

WORKDIR /src
# Copy your Python scripts into the container
COPY 00_Train_EfficientAD.py ./00_Train_EfficientAD.py 
COPY 01_Inference_EfficientAD.py ./01_Inference_EfficientAD.py
COPY 02_Test_Writing.py ./02_Test_Writing.py
COPY 99_Check_for_GPU.py ./99_Check_for_GPU.py
COPY start.sh ./start.sh

CMD bash

