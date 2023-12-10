FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

COPY /builder/setup.sh /builder/requirements.txt /app/
RUN /bin/bash setup.sh && rm setup.sh
RUN python3 -mpip install -r requirements.txt && rm requirements.txt

RUN git clone https://github.com/tencent-ailab/IP-Adapter.git /tmp/ip_adapter && \
    cp -r /tmp/ip_adapter/ip_adapter /usr/local/lib/python3.10/dist-packages

COPY /src/ /app/

ARG DEPLOYMENT_NAME=sdxl
ENV DEPLOYMENT=$DEPLOYMENT_NAME

RUN python3 cache_models.py

CMD python3 /app/handler.py