import re
import sys
import os
import json
import argparse
import torch
from safetensors import safe_open

def mapping_hf_clip_vit(k):
    """
    Mapping weight key to HF clip model key.
    Support load OpenAI Clip/Google Siglip model.
    """
    hf_clip_k = k

    if re.search("layers[.]([0-9]+)", k):
        hf_clip_k = re.sub("^vision_model[.]encoder[.]", "vision_model.transformer.", hf_clip_k)

        hf_clip_k = re.sub("layer_norm1.weight", "first_norm.alpha", hf_clip_k)
        hf_clip_k = re.sub("layer_norm1.bias", "first_norm.bias",  hf_clip_k)
        hf_clip_k = re.sub("layer_norm2.weight", "last_norm.alpha",  hf_clip_k)
        hf_clip_k = re.sub("layer_norm2.bias", "last_norm.bias", hf_clip_k)

        hf_clip_k = re.sub("self_attn", "self_attention", hf_clip_k)
        hf_clip_k = re.sub("out_proj", "o_proj", hf_clip_k)
        hf_clip_k = re.sub("(attention.*)weight$", "\\1W", hf_clip_k)
        hf_clip_k = re.sub("(attention.*)bias$", "\\1b", hf_clip_k)

        hf_clip_k = re.sub("mlp.fc1", "ffn.up_proj", hf_clip_k)
        hf_clip_k = re.sub("mlp.fc1", "ffn.up_proj", hf_clip_k)
        hf_clip_k = re.sub("mlp.fc2", "ffn.down_proj", hf_clip_k)
        hf_clip_k = re.sub("mlp.fc2", "ffn.down_proj", hf_clip_k)
        hf_clip_k = re.sub("(ffn.*)weight$", "\\1W", hf_clip_k)
        hf_clip_k = re.sub("(ffn.*)bias$", "\\1b", hf_clip_k)

    else:
        hf_clip_k = hf_clip_k.replace("vision_model.embeddings.class_embedding", "vision_model.cls") 
        hf_clip_k = hf_clip_k.replace("vision_model.embeddings.patch_embedding.weight", "vision_model.patch_embedding.weight")
        hf_clip_k = hf_clip_k.replace("vision_model.embeddings.patch_embedding.bias", "vision_model.patch_embedding.bias")
        hf_clip_k = hf_clip_k.replace("vision_model.embeddings.position_embedding.weight", "vision_model.pos_embedding.W")
        hf_clip_k = re.sub("vision_model.pre_laye*rnorm.weight", "vision_model.emb_norm.alpha", hf_clip_k)
        hf_clip_k = re.sub("vision_model.pre_laye*rnorm.bias", "vision_model.emb_norm.bias", hf_clip_k)
        hf_clip_k = re.sub("vision_model.post_laye*rnorm.weight", "vision_model.last_norm.alpha", hf_clip_k)
        hf_clip_k = re.sub("vision_model.post_laye*rnorm.bias", "vision_model.last_norm.bias", hf_clip_k)

    return hf_clip_k


def mapping_hf_llama(k):
    """ 
    Mapping weight key to HF llama model key.
    """
    hf_llama_k = k

    hf_llama_k = hf_llama_k.replace("layers.", "transformer.layers.")
    hf_llama_k = hf_llama_k.replace(".self_attn.q_proj.weight", ".self_attention.q_proj.W")
    hf_llama_k = hf_llama_k.replace(".self_attn.k_proj.weight", ".self_attention.k_proj.W")
    hf_llama_k = hf_llama_k.replace(".self_attn.v_proj.weight", ".self_attention.v_proj.W")
    hf_llama_k = hf_llama_k.replace(".self_attn.o_proj.weight", ".self_attention.o_proj.W")
    hf_llama_k = hf_llama_k.replace(".mlp.down_proj.weight", ".ffn.down_proj.W")
    hf_llama_k = hf_llama_k.replace(".mlp.gate_proj.weight", ".ffn.gate_proj.W")
    hf_llama_k = hf_llama_k.replace(".mlp.up_proj.weight", ".ffn.up_proj.W")
    hf_llama_k = hf_llama_k.replace(".input_layernorm.weight", ".first_norm.alpha")
    hf_llama_k = hf_llama_k.replace(".post_attention_layernorm.weight", ".last_norm.alpha")
    hf_llama_k = hf_llama_k.replace("model.norm.weight", "model.last_norm.alpha")
    hf_llama_k = hf_llama_k.replace("model.embed_tokens.weight", "model.word_embedding.W")
    hf_llama_k = hf_llama_k.replace("model.", "lm_model.")
    
    return hf_llama_k 
    

def mapping_hf(k):
    """ 
    Mapping weight key to HF model key.
    """
    if "vision" in k:
        return mapping_hf_clip_vit(k)
    return mapping_hf_llama(k)


def convert_weights(hf_path, convert_path):
    """
    """
    if not os.path.exists(convert_path):
        os.makedirs(convert_path, exist_ok=True)

    idx = 0
    for f in os.listdir(hf_path):
        if not f.endswith(".safetensors"):
            continue
        tensors = {}
        with safe_open(os.path.join(hf_path, f), framework="pt") as f:
            tensor_names = f.keys()
            for name in tensor_names:
                tensors[name] = f.get_tensor(name)
        for k in tensors:
            print(f"mapping {k} weight to {mapping_hf(k)}")
        weights = {mapping_hf(k):tensors[k] for k in tensors}
        torch.save(weights, os.path.join(convert_path, f"model_weights_{idx}"))
        idx += 1


def main():
    """
    """
    description=(
            f"Convert hf model to mimix model."
    )
    usage = (
    'python --hf_path <hf_path> --convert_path <convert_path>'
    )
    parser = argparse.ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--hf_path",
                        type=str,
                        help="Original hf model path.")

    parser.add_argument('--convert_path',
                        type=str,
                        help="Converted model path.")

    args = parser.parse_args(sys.argv[1:])

    convert_weights(args.hf_path, args.convert_path)


if __name__ == "__main__":

    main()
