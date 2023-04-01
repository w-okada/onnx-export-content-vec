#!/bin/bash
set -eu


DOCKER_IMAGE=dannadori/onnx-export-content-vec:20230401_184324

docker run --gpus all --rm -ti --shm-size=256M \
    -v `pwd`/work:/work \
    $DOCKER_IMAGE --input work/checkpoint_best_legacy_500.pt

