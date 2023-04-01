#!/bin/bash
set -eu


DOCKER_IMAGE=onnx-export-content-vec:20230401_183130

docker run --gpus all --rm -ti --shm-size=256M \
    -v `pwd`/work:/work \
    $DOCKER_IMAGE --input work/checkpoint_best_legacy_500.pt

