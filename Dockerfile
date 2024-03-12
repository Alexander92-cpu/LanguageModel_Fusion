# Stage 1: Build the dependencies
FROM python:3.10-bookworm AS big_python

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKDIR_BASIC=/app
ENV VIRTUAL_ENV=$WORKDIR_BASIC/venv
ENV PROJECT_NAME=rnnt_lm_fusion

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && apt-get install -y  cmake git git-lfs libsndfile1 ffmpeg \
    build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev \
    libboost-test-dev libboost-filesystem-dev

WORKDIR $WORKDIR_BASIC

RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip build Cython setuptools wheel

RUN git clone https://github.com/kpu/kenlm.git
RUN mkdir kenlm/build && cd kenlm/build && cmake .. && make -j 4

COPY pyproject.toml .
COPY $PROJECT_NAME $WORKDIR_BASIC/$PROJECT_NAME

RUN python -m build
RUN pip install --find-links=dist $PROJECT_NAME[test]

# Stage 2: Get CUDA
# FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as cuda_libraries

# Stage 3: Create the final image
FROM python:3.10-slim-bookworm

ENV WORKDIR_BASIC=/app
ENV VIRTUAL_ENV=$WORKDIR_BASIC/venv
ENV PROJECT_NAME=rnnt_lm_fusion

LABEL org.opencontainers.image.source=https://github.com/Alexander92-cpu/LanguageModel_Fusion
LABEL org.opencontainers.image.description="RNN-T Language Model Fusion"
LABEL org.opencontainers.image.licenses=Apache-2.0

WORKDIR $WORKDIR_BASIC

COPY --from=big_python $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=big_python $WORKDIR_BASIC/kenlm kenlm
# COPY --from=cuda_libraries /usr/local/cuda /usr/local/cuda
# COPY --from=cuda_libraries /usr/local/cuda-12 /usr/local/cuda-12
# COPY --from=cuda_libraries /usr/local/cuda-12.2 /usr/local/cuda-12.2

# ENV PATH="/usr/local/cuda/bin:$VIRTUAL_ENV/bin:$PATH"
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser
