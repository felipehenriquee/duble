#!/usr/bin/env bash
# Inicia o tradutor de voz com as bibliotecas CUDA do ambiente virtual.
cd "$(dirname "$0")"
NV=.venv/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NV/cublas/lib:$PWD/$NV/cudnn/lib:$LD_LIBRARY_PATH"
exec .venv/bin/python tradutor.py "$@"
