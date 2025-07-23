FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

# Copy source code
COPY src .

# Add test_input.json
COPY test_input.json /

# Download only the tiny model to save space (moved after copying code to leverage layer caching)
RUN mkdir -p ./weights && \
    wget -q https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./weights

CMD [ "python", "-u", "/rp_handler.py" ]