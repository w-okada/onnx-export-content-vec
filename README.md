# ContentVecOnnxConverter

Convert contentvec model from pyTorch model to ONNX.

# !!!! NOTE  !!!!
This repository is archved.

New is [here](https://github.com/w-okada/onnx-export-content-vec-multi-version)

# Usage

## Preparation

Download model from [contentvec repository](https://github.com/auspicious3000/contentvec).
File is "ContentVec_legacy,500".
Put the model into `./work` directory. The name is `checkpoint_best_legacy_500.pt`.

```
$ ls work/
checkpoint_best_legacy_500.pt
```

## Execution

```
bash start_docker.sh
```

## Result

You can find the converted model `checkpoint_best_legacy_500_simple.onnx`

Size is 280M!! Great!!!!!!

```
$ ls -lah work/
280M   checkpoint_best_legacy_500.onnx
1.3G   checkpoint_best_legacy_500.pt
280M   checkpoint_best_legacy_500_simple.onnx
```

## Track Record of Use

Generate ONNX model used in [VC Client](https://github.com/w-okada/voice-changer) which is realtime voice conversion for [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) and [MMVC](https://github.com/isletennos/MMVC_Trainer).

# Acknowledgments

- [contentvec](https://github.com/auspicious3000/contentvec)
- [issue](https://github.com/auspicious3000/contentvec/issues/6)
