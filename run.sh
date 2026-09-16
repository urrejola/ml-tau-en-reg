#!/bin/bash

#keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
export PYTHONPATH=$(pwd):$(pwd)/enreg/omnijet_alpha:$PYTHONPATH
python3 "$@"
