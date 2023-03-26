import os
from collections import OrderedDict
from typing import Tuple, Union
import torch
from torch import nn
import numpy as np

from .mat import MultiAxisTransformer
from .mit import MultiframeIntegrationTransformer
from .prompt import VideoSpecificPrompt
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer
import clip

MODEL_PATH = {
    "ViT-B/32": "/PATH/TO/ViT-B-32.pt",
    "ViT-B/16": "/PATH/TO/ViT-B-16.pt",
    "ViT-L/14": "/PATH/TO/ViT-L-14.pt",
    "ViT-L/14@336px": "/PATH/TO/ViT-L-14-336px.pt"
}


def load_state_dict_time(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        # _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        # _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        # self.prompts_generator = VideoSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        # self.use_cache = use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64

        self.visual = MultiAxisTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
        )


        # self.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask()
        # )
        # self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # self.ln_final = LayerNorm(transformer_width)
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #
        # self.cache_text_features = None
        # self.prompts_visual_ln = LayerNorm(vision_width)
        # self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))

        # self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    # def initialize_parameters(self):
    #     nn.init.normal_(self.token_embedding.weight, std=0.02)
    #     nn.init.normal_(self.positional_embedding, std=0.01)
    #
    #     proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
    #     attn_std = self.transformer.width ** -0.5
    #     fc_std = (2 * self.transformer.width) ** -0.5
    #     for block in self.transformer.resblocks:
    #         nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
    #         nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
    #         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
    #         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    #
    #     if self.text_projection is not None:
    #         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_image(self, image):
        return self.visual(image)

    # def encode_text(self, text):
    #     x = self.token_embedding(text)
    #     eos_indx = text.argmax(dim=-1)
    #     K, N1, C = x.shape
    #
    #     x = x + self.positional_embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x)
    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
    #     x = x.reshape(K, -1)
    #     return x

    def encode_video(self, image):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        cls_features, img_features, cos_loss_list = self.encode_image(image)

        # img_features = self.prompts_visual_ln(img_features)
        # img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1)
        img_features = img_features.view(b,t,-1,cls_features.shape[-1])
        
        video_features = self.mit(cls_features)

        return video_features, img_features, cos_loss_list

    # def cache_text(self, text):
    #     self.eval()
    #     with torch.no_grad():
    #         if self.cache_text_features is None:
    #             self.cache_text_features = self.encode_text(text)
    #     self.train()
    #     return self.cache_text_features

    def forward(self, image, text):
        b = image.shape[0]
        video_features, img_features, cos_loss_list = self.encode_video(image)
        img_features = img_features.mean(dim=1, keepdim=False)

        # if self.use_cache:
        #     text_features = self.cache_text(text)
        # else:
        #     text_features = self.encode_text(text)
        
        # text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        # text_features = text_features + self.prompts_generator(text_features, img_features)
           
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)

        logits = video_features

        return logits, cos_loss_list


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4,):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"load pretrained CLIP: {msg}")

    frozen_list = ["visual.class_embedding", "visual.positional_embedding", "visual.proj", "visual.conv1.weight", "visual.ln_pre.weight", "visual.ln_pre.bias", "visual.ln_post.weight", "visual.ln_post.bias"]
    for name, param in model.named_parameters():
        for i in range(len(frozen_list)):
            if name == frozen_list[i]:
                param.requires_grad = False
        for i in range(vision_layers):
            if "visual.transformer.resblocks.{}.attn".format(i) in name:
                param.requires_grad = False
            elif "visual.transformer.resblocks.{}.ln_1".format(i) in name:
                param.requires_grad = False
            elif "visual.transformer.resblocks.{}.mlp".format(i) in name:
                param.requires_grad = False
            elif "visual.transformer.resblocks.{}.ln_2".format(i) in name:
                param.requires_grad = False

    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1, prompts_layers=2, mit_layers=1,
):
    if model_path is None:
        model_path = MODEL_PATH[name]
    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()