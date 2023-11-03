FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

COPY /builder/setup.sh /builder/requirements.txt /app/
RUN /bin/bash setup.sh && rm setup.sh
RUN python3 -mpip install -r requirements.txt && rm requirements.txt

COPY /src/cache_models.py /src/constants.py /app/
RUN python3 cache_models.py

COPY /src/* /app/

CMD python3 /app/handler.py