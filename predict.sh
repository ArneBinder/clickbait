#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

#MODEL_DIR="$SCRIPT_DIR/model_wimages_best"
#echo "use model from $MODEL_DIR"

cd "$SCRIPT_DIR"
python spacy_deep_learning_keras.py --model-dir model_wimages_best "$@"
