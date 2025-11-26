FROM nvcr.io/nvidia/pytorch:22.09-py3 

ENV PATH="/usr/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py utils.py dataset.py inference.py ./

COPY model_weights/best_model.pth .

RUN mkdir -p /opt/algorithm && \ 
    mkdir -p /input && \
    mkdir -p /output

RUN groupadd -r app_user && useradd -r -g app_user app_user

RUN chown -R app_user:app_user /app && \
    chown -R app_user:app_user /opt/algorithm && \
    chown -R app_user:app_user /input && \
    chown -R app_user:app_user /output

USER app_user

ENTRYPOINT ["python", "inference.py"]