FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV PROJECT=/mos4d
RUN mkdir -p $PROJECT
RUN mkdir -p /mos4d/logs

ENV DATA=/mos4d/data
RUN mkdir -p $DATA

# Install MinkowskiEngine Dependencies
RUN apt-get update || true && apt-get install --no-install-recommends -y \
          git \
          libopenblas-dev \
          && rm -rf /var/lib/apt/lists/*

# Install project related dependencies
WORKDIR $PROJECT
COPY . $PROJECT
RUN python3 -m pip install --editable .
RUN rm -rf $PROJECT

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--cuda_home=/usr/local/cuda-11.7" \
                           --install-option="--blas=openblas"


# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
