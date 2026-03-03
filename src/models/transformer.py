import torch
import torch.nn as nn
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    def __init__(
        self, 
        img_size=224,
        patch_size=16,
        stride=10,
        in_chans=1,
        embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
    def forward(self, x):
        B, C, F, T = x.shape
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        n_heads=12,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
        ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, D)

        x = self.projection(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None, 
        drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim, 
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class AST(nn.Module):
    def __init__(
        self,
        n_classes=50,
        input_fdim=128,
        input_tdim=128,
        patch_size=16,
        stride=10,
        in_chans=1,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0,
        attn_drop_rate=0
    ):
        super().__init__()

        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.n_patches = self._calculate_n_patches(input_fdim, input_tdim, patch_size, stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _calculate_n_patches(self, fdim, tdim, patch_size, stride):
        f_patches = (fdim - patch_size) // stride + 1
        t_patches = (tdim - patch_size) // stride + 1
        return f_patches * t_patches

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_layer_weights)
    
    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2).unsqueeze(1)
        
        elif x.dim() == 4 and x.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Expected input shape [B, T, F] or [B, 1, F, T], got  {x.shape}")

        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x) 

        cls_output = x[:, 0]

        logits = self.head(cls_output)

        return logits

    def get_attention_maps(self, x, block_idx=-1):
        if x.dim() == 3:
            x = x.transpose(1, 2).unsqueeze(1)
        
        N = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        target_block = self.blocks[block_idx]
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) + block_idx:
                x = block(x)
            else:
                break
                
        x_norm = target_block.norm1(x)
        B, N, D = x_norm.shape
        qkv = target_block.attn.qkv(x_norm).reshape(B, N, 3, target_block.attn.n_heads, target_block.attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * target_block.attn.scale
        attn = attn.softmax(dim=-1)
        
        return attn
