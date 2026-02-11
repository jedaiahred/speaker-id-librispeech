FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace

RUN python -m pip install --upgrade pip

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

RUN python -m ipykernel install --name tf-25.02 --display-name "Python (TF 25.02)"
