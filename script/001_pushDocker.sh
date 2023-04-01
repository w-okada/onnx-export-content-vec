#!/bin/bash

data_tag=`date +%Y%m%d_%H%M%S`
docker login 

docker tag onnx-export-content-vec dannadori/onnx-export-content-vec:$data_tag
docker push dannadori/onnx-export-content-vec:$data_tag
