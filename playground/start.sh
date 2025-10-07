#!/bin/bash
echo "Checking for GPUs"
python 99_Check_for_GPU.py

echo "Training Efficient AD"
python 00_Train_EfficientAD.py

echo "Inference on Efficient AD"
python 01_Inference_EfficientAD.py
