# ContentVecOnnxConverter

Convert contentvec model from pyTorch model to ONNX.

# Usage

## Preparation

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

```
$ ls work/
checkpoint_best_legacy_500.onnx  checkpoint_best_legacy_500_simple.onnx
checkpoint_best_legacy_500.pt
```

## Track Record of Use

Generate ONNX model used in [VC Client](https://github.com/w-okada/voice-changer) which is realtime voice conversion for [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) and [MMVC](https://github.com/isletennos/MMVC_Trainer).
