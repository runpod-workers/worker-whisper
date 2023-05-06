FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Install System Packages
RUN apt-get update
RUN apt-get install ffmpeg

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]
