#!/bin/bash

PREPROCESS=1
DOWNLOAD=1
while getopts ":p" opt; do
  case ${opt} in
    p ) # process option h
      DOWNLOAD=0
      ;;
    d )
      PREPROCESS=0
      ;;
    \? )
      echo "Usage: cmd [-p PREPROCESS_ONLY] [-d DOWNLOAD_ONLY]" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

if [[ $DOWNLOAD -eq 1 ]]; then 
    echo "Downloading facebookresearch/empatheticdialogues to ./data..."
    mkdir -p data
    wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
    tar xf empatheticdialogues.tar.gz
    mv empatheticdialogues data
fi

if [[ $PREPROCESS -eq 1 ]]; then
    echo "Preprocessing empatheticdialogues data:"
    python preprocess_ed.py --data-dir ./data/empatheticdialogues --out-file ./data/empatheticdialogues/ed_merged.json
fi
