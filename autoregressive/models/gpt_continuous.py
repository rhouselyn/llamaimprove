from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoregressive.models.gpt import (
    ModelArgs as BaseModelArgs,
    LabelEmbedder,
    CaptionEmbedder,
    RMSNorm,
    FeedForward,
    KVCache,
    Attention,
    TransformerBlock,
    precompute_freqs_cis_2d,
    find_multiple,
)


@dataclass
class ContinuousModelArgs(BaseModelArgs):
    codebook_dim: int = 128


class ContinuousTransformer(nn.Module):
    def __init__(self, config: ContinuousModelArgs):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.codebook_dim = config.codebook_dim

        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")

        self.tok_embeddings = nn.Linear(config.codebook_dim, config.dim, bias=False)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.codebook_dim, bias=False)

        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size, \
            f"block_size={self.block_size} must be a perfect square for 2D RoPE"
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, config.dim // config.n_head, config.rope_base, self.cls_token_num
        )

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num
        )

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            else:
                token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        output = self.output(h).float()

        if self.training:
            output = output[:, self.cls_token_num - 1:].contiguous()

        loss = None
        if valid is not None:
            loss_all = F.mse_loss(output, targets, reduction='none').mean(dim=-1)
            valid_all = valid[:, None].repeat(1, output.shape[1])
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.mse_loss(output, targets)

        return output, loss

    @torch.no_grad()
    def generate(self, cond_idx, max_new_tokens, temperature=1.0, top_k=None, cfg_scale=1.0):
        self.setup_caches(max_batch_size=cond_idx.shape[0], max_seq_length=max_new_tokens + self.cls_token_num, dtype=next(self.parameters()).dtype)

        if cfg_scale != 1.0:
            cond_null = torch.ones_like(cond_idx) * self.num_classes
            cond_combined = torch.cat([cond_idx, cond_null])
        else:
            cond_combined = cond_idx

        bs = cond_combined.shape[0]
        input_pos = torch.arange(0, self.cls_token_num, device=cond_combined.device)
        _, _ = self(None, cond_combined, input_pos=input_pos)

        generated = []
        for i in range(max_new_tokens):
            if i == 0:
                x_prev = torch.zeros(bs, 1, self.codebook_dim, device=cond_combined.device)
            else:
                x_prev = token.clone().detach().unsqueeze(1)

            input_pos = torch.tensor([self.cls_token_num + i], device=cond_combined.device)
            logits, _ = self(x_prev, None, input_pos=input_pos)
            token = torch.sign(logits[:, -1, :])

            if cfg_scale != 1.0:
                cond_token = token[:bs // 2]
                uncond_token = token[bs // 2:]
                token = uncond_token + cfg_scale * (cond_token - uncond_token)
                token = torch.sign(token)
                generated.append(token[:bs // 2])
            else:
                generated.append(token)

        return torch.stack(generated, dim=1)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


def GPT_B_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))

def GPT_L_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))

def GPT_XL_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))

def GPT_XXL_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))

def GPT_XXXL_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))

def GPT_1B_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))

def GPT_3B_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))

def GPT_7B_cont(**kwargs):
    return ContinuousTransformer(ContinuousModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))


GPT_continuous_models = {
    'GPT-B': GPT_B_cont, 'GPT-L': GPT_L_cont, 'GPT-XL': GPT_XL_cont,
    'GPT-XXL': GPT_XXL_cont, 'GPT-XXXL': GPT_XXXL_cont,
    'GPT-1B': GPT_1B_cont, 'GPT-3B': GPT_3B_cont, 'GPT-7B': GPT_7B_cont,
}
