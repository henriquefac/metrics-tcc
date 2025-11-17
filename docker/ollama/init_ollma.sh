#!/bin/bash

docker run -d \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama || true

docker start ollama

docker exec ollama ollama pull nomic-embed-text

