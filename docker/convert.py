import os
import torch
from torch import nn
from fairseq import checkpoint_utils
from transformers import HubertConfig, HubertModel
from onnxsim import simplify
import onnx
import logging
# Ignore fairseq's logger
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)

import argparse


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class HubertSoft(nn.Module):
    def __init__(self):
        super(HubertSoft, self).__init__()
        self.hubert = HubertModelWithFinalProj.from_pretrained(".")

    def forward(self, audio):
        # x = self.hubert(audio)["last_hidden_state"]
        x = self.hubert(audio, output_hidden_states=True)["hidden_states"][9]
        x = self.hubert.final_proj(x)
        return x


mapping = {
    "masked_spec_embed": "mask_emb",
    "encoder.layer_norm.bias": "encoder.layer_norm.bias",
    "encoder.layer_norm.weight": "encoder.layer_norm.weight",
    "encoder.pos_conv_embed.conv.bias": "encoder.pos_conv.0.bias",
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv.0.weight_g",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv.0.weight_v",
    "feature_projection.layer_norm.bias": "layer_norm.bias",
    "feature_projection.layer_norm.weight": "layer_norm.weight",
    "feature_projection.projection.bias": "post_extract_proj.bias",
    "feature_projection.projection.weight": "post_extract_proj.weight",
    "final_proj.bias": "final_proj.bias",
    "final_proj.weight": "final_proj.weight",
}


def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to model(pth)")
    parser.add_argument("--output", type=str, help="path to model(pth)")
    return parser


def convertToHuggingfaceModel(filename):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([filename], suffix="")
    model = models[0]
    model.eval()

    hubert = HubertModelWithFinalProj(HubertConfig())

    # Convert encoder
    for layer in range(12):
        for j in ["q", "k", "v"]:
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.weight"] = f"encoder.layers.{layer}.self_attn.{j}_proj.weight"
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.bias"] = f"encoder.layers.{layer}.self_attn.{j}_proj.bias"

        mapping[f"encoder.layers.{layer}.final_layer_norm.bias"] = f"encoder.layers.{layer}.final_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.final_layer_norm.weight"] = f"encoder.layers.{layer}.final_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.layer_norm.bias"] = f"encoder.layers.{layer}.self_attn_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.layer_norm.weight"] = f"encoder.layers.{layer}.self_attn_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.attention.out_proj.bias"] = f"encoder.layers.{layer}.self_attn.out_proj.bias"
        mapping[f"encoder.layers.{layer}.attention.out_proj.weight"] = f"encoder.layers.{layer}.self_attn.out_proj.weight"

        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"] = f"encoder.layers.{layer}.fc1.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"] = f"encoder.layers.{layer}.fc1.weight"

        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.bias"] = f"encoder.layers.{layer}.fc2.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.weight"] = f"encoder.layers.{layer}.fc2.weight"

    # Convert Conv Layers
    for layer in range(7):
        mapping[f"feature_extractor.conv_layers.{layer}.conv.weight"] = f"feature_extractor.conv_layers.{layer}.0.weight"

        if layer != 0:
            continue

        mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.weight"] = f"feature_extractor.conv_layers.{layer}.2.weight"
        mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.bias"] = f"feature_extractor.conv_layers.{layer}.2.bias"

    hf_keys = set(hubert.state_dict().keys())
    fair_keys = set(model.state_dict().keys())

    hf_keys -= set(mapping.keys())
    fair_keys -= set(mapping.values())

    for i, j in zip(sorted(hf_keys), sorted(fair_keys)):
        print(i, j)

    print(hf_keys, fair_keys)
    print(len(hf_keys), len(fair_keys))

    # try loading the weights
    new_state_dict = {}
    for k, v in mapping.items():
        new_state_dict[k] = model.state_dict()[v]

    x = hubert.load_state_dict(new_state_dict, strict=False)
    print(x)
    hubert.eval()

    with torch.no_grad():
        new_input = torch.randn(1, 16384)

        result1 = hubert(new_input, output_hidden_states=True)["hidden_states"][9]
        result1 = hubert.final_proj(result1)

        result2 = model.extract_features(
            **{
                "source": new_input,
                "padding_mask": torch.zeros(1, 16384, dtype=torch.bool),
                # "features_only": True,
                "output_layer": 9,
            }
        )[0]
        result2 = model.final_proj(result2)
        assert torch.allclose(result1, result2, atol=1e-3)
    print("Sanity check passed")
    # Save huggingface model
    hubert.save_pretrained(".")
    print("Saved model")


def convertToOnnx(outputFile, outputSimpleFile):
    model = HubertSoft()
    audio = torch.randn(1, 101467)
    input_names = ["audio",]
    output_names = ["units", ]
    torch.onnx.export(model,
                      (audio,),
                      outputFile,
                      dynamic_axes={
                          "audio": [1],
                      },
                      do_constant_folding=False,
                      opset_version=16,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names)

    model_onnx2 = onnx.load(outputFile)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, outputSimpleFile)


if __name__ == "__main__":
    parser = setupArgParser()
    args, unknown = parser.parse_known_args()
    print(args.input)
    inputFile = args.input
    outputFile = os.path.splitext(inputFile)[0] + '.onnx'
    outputSimpleFile = os.path.splitext(inputFile)[0] + '_simple.onnx'

    convertToHuggingfaceModel(inputFile)
    convertToOnnx(outputFile, outputSimpleFile)
