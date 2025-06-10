#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Install system dependencies for PDF support
apt-get update && apt-get install -y \
    graphviz \
    libgl1-mesa-glx \
    libglib2.0-0

# Launch the application
python main.py 