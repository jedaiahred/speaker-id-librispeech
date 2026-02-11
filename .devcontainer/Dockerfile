# Base image: your NGC TensorFlow container
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set a working directory inside the container
WORKDIR /workspace

# (Optional but recommended) Update pip
RUN python -m pip install --upgrade pip

# Install some common Python dev tools / libraries
# Add/remove whatever you like.
RUN pip install --no-cache-dir \
    jupyterlab \
    ipykernel \
    matplotlib \
    pandas \
    scikit-learn \
    seaborn \
    black \
    isort \
    librosa \ 
    soundfile \
    tqdm \ 
    joblib

# Register a Jupyter kernel for this environment (nice for notebooks)
RUN python -m ipykernel install --name tf-25.02 --display-name "Python (TF 25.02)"

# You can add any other ML libs here later, e.g.:
# RUN pip install --no-cache-dir librosa tqdm wandb
