import json
import io
import requests
from PIL import Image
import torch
import torch.nn as nn
from transformers import SiglipModel, SiglipImageProcessor
from transformers import CLIPProcessor, CLIPModel
from mimixlm import TransformerEncoder,build_image_processor
from convert_hf_weights import mapping_hf_clip_vit


class VIT(nn.Module):
    """
    Vision Transformer (ViT) for Vision-Language Models (VLM)
    """
    def __init__(self, **kwargs):
        """
        Initialize VIT  Model
        """
        super(VIT, self).__init__()
        self.vision_model = TransformerEncoder(**kwargs["model_config"])
        image_size= kwargs["image_size"]
        if isinstance(image_size, int):
            self.image_size = (3, image_size, image_size)
        elif len(image_size) == 1:
            self.image_size = (3, image_size[0], image_size[0])
        else:
            self.image_size = (3, image_size[0], image_size[1])


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize layer parameters.
        """
        self.vision_model.reset_parameters(initializer_range)


    def forward(self,
                x,
                use_checkpoint=False):
        """
        """
        outputs = self.vision_model(x, use_checkpoint=use_checkpoint)
        return outputs


def get_test_img():
    """
    """
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = None
    cnt = 0
    while cnt < 3:
        try:
            image = Image.open(io.BytesIO(requests.get(url).content))
            break
        except:
            print(f"get image error! retry:{cnt}")
            cnt += 1
            continue
    return image
 
def test_siglip():
    """
    """
    ckpt = "model/siglip2-large-patch16-256"
    processor = SiglipImageProcessor.from_pretrained(ckpt)
    
    model = SiglipModel.from_pretrained(ckpt, device_map="cpu").eval()
   
    config = {
            "input_type": "image",
            "n_layers": 24,
            "d_model": 1024,
            "d_ff": 4096,
            "n_heads": 16,
            "use_glu": False,
            "use_attention_bias": True,
            "layer_norm_type": "layer_norm",
            "ln_eps": 1e-6,
            "use_ffn_bias": True,
            "attn_pos_embedding_type": "none",
            "activation": "gelutanh",
            "norm_after_embedding": False,
            "use_cls_embedding": False,
            "use_patch_emb_bias": True,
            "patch_size": 16,
            "image_size": [                                                                                
                256,                                                                                       
                256                                                                                        
            ],                                                                                             
            "resample": 2,                                                                                 
            "rescale_factor": 0.00392156862745098,                                                         
            "norm_mean": [                                                                                 
                0.5,                                                                                       
                0.5,                                                                                       
                0.5                                                                                        
            ],                                                                                             
            "norm_std": [                                                                                  
                0.5,                                                                                       
                0.5, 
                0.5                                                                                        
            ],                                                                                             
            "attention_backend": "torch_native"                                                            
        }
  
    processor2 = build_image_processor(config)
    vit = VIT(**config).float().eval()
    
    vit_weights = {}
    for k,v in model.named_parameters():
        if "vision" in k:
            vit_weights[mapping_hf_clip_vit(k)] = v
    vit.load_state_dict(vit_weights, False)       

    image = get_test_img()

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    x_vision = inputs.pixel_values
    with torch.no_grad():
        image_embeddings = model.vision_model(**inputs, output_attentions=False, output_hidden_states=True)

    x_vision2 = processor2(image).unsqueeze(0)
    with torch.no_grad():
        res = vit(x=x_vision2)["hidden_states"]
 
    for i,(v1,v2) in enumerate(zip(image_embeddings.hidden_states, res)):
        assert torch.max(torch.abs(v1-v2)).item() < 2e-3


def test_clip():
    """
    """
    model = CLIPModel.from_pretrained("model/clip-vit-large-patch14").cpu().float().eval()
    processor = CLIPProcessor.from_pretrained("model/clip-vit-large-patch14")

    config = {
            "input_type": "image",
            "n_layers": 24,
            "d_model": 1024,
            "d_ff": 4096,
            "n_heads": 16,
            "use_glu": False,
            "use_attention_bias": True,
            "layer_norm_type": "layer_norm",
            "use_ffn_bias": True,
            "attn_pos_embedding_type": "none",
            "activation": "geluquick",
            "norm_after_embedding": True,
            "use_cls_embedding": True,
            "use_patch_emb_bias": False,
            "patch_size": 14,
            "image_size": 224,
            "resample": 3,                                                                                 
            "rescale_factor": 0.00392156862745098,                                                         
            "norm_mean": [                                                                                 
                0.48145466,                                                                                
                0.4578275,                                                                                 
                0.40821073                                                                                 
            ],                                                                                             
            "norm_std": [                                                                                  
                0.26862954,                                                                                
                0.26130258,                                                                                
                0.27577711                                                                                 
            ]                                                                                              
        }                                                                                                  
     
    processor2 = build_image_processor(config)
    vit = VIT(**config).float().eval()

    vit_weights = {}
    for k,v in model.named_parameters():                                                                   
        if "vision" in k:                                                                                  
            vit_weights[mapping_hf_clip_vit(k)] = v                                                        
    vit.load_state_dict(vit_weights)                                                                

    image = get_test_img()

    inputs = processor(images=image, return_tensors="pt").to(model.device)                                 
    x_vision = inputs.pixel_values                                                                         
    with torch.no_grad():                                                                                  
        image_embeddings = model.vision_model(**inputs, output_attentions=False, output_hidden_states=True)

    x_vision2 = processor2(image).unsqueeze(0)                                                             
    with torch.no_grad():                                                                                  
        res = vit(x=x_vision2)["hidden_states"]                                                            
 
    for i,(v1,v2) in enumerate(zip(image_embeddings.hidden_states, res)):                                  
        print(torch.max(torch.abs(v1-v2)).item())
        assert torch.max(torch.abs(v1-v2)).item() < 1e-3 
       

if __name__ == "__main__":
   
    test_clip()
    
    #test_siglip()     
 
