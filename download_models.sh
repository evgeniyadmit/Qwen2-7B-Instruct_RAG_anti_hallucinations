#!/bin/bash
mkdir -p models/vosk-en models/vosk-ru models/qwen-2-7b models/minilm-multilingual

echo "Downloading Qwen 2 7B model..."
wget -q -O models/qwen-2-7b/model.tar \
  "huggingface.co/Qwen/Qwen2-7B-Instruct/resolve/main/pytorch_model.bin"
echo "✅ Qwen 2 7B instruct model saved"

echo "Downloading paraphrase embedding model..."
wget -q -O models/minilm-multilingual/model.tar \
  "huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/pytorch_model.bin"
echo "✅ paraphrase multilingual MiniLM L12 V2 saved"

echo "All models downloaded locally"
