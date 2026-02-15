FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \ 
    make \
    pkg-config  \
    libsndfile1 \
    g++ \
    cmake \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt    

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]