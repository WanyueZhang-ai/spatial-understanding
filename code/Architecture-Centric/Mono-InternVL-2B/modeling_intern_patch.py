# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_intern_patch import InternVisionPatchConfig

logger = logging.get_logger(__name__)


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionPatchConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
            
        import os
        mode = os.getenv("VIT_POSITION_MODE", "vanilla")
        # mode = 'noise'
        batch_size, seq_length, embed_dim = pos_embed.shape
        if mode == "vanilla":
            pass
        elif mode == "random":
            indices = torch.randperm(pos_embed.size(1))
            shuffled_position_embedding = pos_embed[:, indices, :]
            pos_embed = shuffled_position_embedding
        elif mode == "first":
            pos_embed = pos_embed[:, 0:1, :].expand(batch_size, seq_length, embed_dim)
        elif mode == "last":
            pos_embed = pos_embed[:, -2:-1, :].expand(batch_size, seq_length, embed_dim)
        elif mode == "noise":
            mean = pos_embed.mean()
            std = pos_embed.std()
            noise = torch.randn_like(pos_embed) * std + mean
            pos_embed = noise
        elif mode == "random_w":
            x = torch.arange(0, seq_length).reshape(-1, H, W)
            # Step 2: shuffle each row (dim=2)
            row_shuffled = torch.stack([
                xi[:, torch.randperm(W)] for xi in x
            ]).reshape(seq_length)
            pos_embed = pos_embed[:, row_shuffled, :]
        elif mode == 'random_h':
            x = torch.arange(0, seq_length).reshape(-1, H, W)
            # Step 2: shuffle each row (dim=2)
            col_shuffled = torch.stack([
                xi.t()[:, torch.randperm(H)].t() for xi in x
            ]).reshape(seq_length)
            pos_embed = pos_embed[:, col_shuffled, :]
                
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings





class InternVisionPatchModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = InternVisionPatchConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionPatchConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)
    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings
 

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')


        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)[:,1:]
        else:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')


        if not return_dict:
            return (hidden_states, None,None)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )
