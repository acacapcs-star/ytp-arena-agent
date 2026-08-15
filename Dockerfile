FROM python:3.11-slim

# `websockets` is required by pirate_client; everything else is up to you.
RUN pip install --no-cache-dir websockets

# Add any extra Python packages your strategy needs here, for example:
# RUN pip install --no-cache-dir numpy
