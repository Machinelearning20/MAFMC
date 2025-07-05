import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F




class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x



class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.dropout_rate=0.1
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        #torch.nn.init.xavier_uniform_(self.base_weight, gain=1.0)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        grid = self.grid.to(x.device)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):

        #print("spline",self.spline_weight.device)
        #print("splinescaler",self.spline_scaler.device)
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
    """
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        self.base_weight.data = self.base_weight.data.to(x.device)
        self.spline_weight.data = self.spline_weight.data.to(x.device)
        self.spline_scaler.data=self.spline_scaler.data.to(x.device)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        #print("weight",self.scaled_spline_weight.device)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output
        #return spline_output
    """
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        self.base_weight.data = self.base_weight.data.to(x.device)
        self.spline_weight.data = self.spline_weight.data.to(x.device)
        self.spline_scaler.data=self.spline_scaler.data.to(x.device)

        # 线性部分（保持不变）
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # B样条部分
        spline_bases = self.b_splines(x)  # 形状 (batch_size, in_features, grid_size + spline_order)
        batch_size, in_features, basis_dim = spline_bases.shape

        # 拼接所有输入节点的基函数为 (batch_size, in_features * basis_dim)
        spline_bases_flat = spline_bases.view(batch_size, -1)

        # 应用Dropout（向量化实现）
        if self.train and self.dropout_rate > 0:
            # 生成掩码：形状为 (in_features)，每个元素表示是否保留该输入的所有基函数
            mask = torch.bernoulli(
            torch.ones(in_features, device=x.device) * (1 - self.dropout_rate)
             ) / (1 - self.dropout_rate)  # 保持期望值不变
            # 扩展掩码到基函数维度：形状变为 (1, in_features * basis_dim)
            mask_expanded = mask.repeat_interleave(basis_dim).view(1, -1)
            spline_bases_flat = spline_bases_flat * mask_expanded

        # 一次性计算所有输入的贡献
        spline_output = F.linear(spline_bases_flat, self.scaled_spline_weight.view(self.out_features, -1))

        return base_output + spline_output
        #return spline_output
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class JacobiKANLayer(nn.Module):
    """
    https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py
    """

    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKANLayer(nn.Module):
    """
    https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
    """

    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.dropout_rate=0.1
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
    
    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        if self.training and self.dropout_rate > 0:
        # 生成掩码：形状为 (1, inputdim, 1)，每个输入节点独立采样
            mask = torch.bernoulli(
            torch.ones(1, self.inputdim, 1, device=x.device) * (1 - self.dropout_rate)
        ) / (1 - self.dropout_rate)  # 保持期望值不变
            x = x * mask  # 广播机制应用掩码
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
    """
    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
    """


class TaylorKANLayer(nn.Module):
    """
    https://github.com/Muyuzhierchengse/TaylorKAN/
    """

    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


class RBFKANLayer(nn.Module):
    """
    https://github.com/Sid2690/RBF-KAN/blob/main/RBF_KAN.py
    """

    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        nn.init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        nn.init.xavier_uniform_(self.weights)

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.gaussian_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output

class Kan_mixer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Kan_mixer, self).__init__()

        self.experts = nn.ModuleList([
                #TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                KANLinear(in_features,out_features),
                #JacobiKANLayer(in_features, out_features, degree=6),
                #RBFKANLayer(in_features, out_features, num_centers=5),
                ChebyKANLayer(in_features, out_features, degree=3),
                nn.Linear(in_features, out_features),
            ])


        self.n_expert = len(self.experts)
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Linear(in_features, self.n_expert)


    def forward(self, x):
        
        B, k,N, L = x.shape
        x = x.reshape(B *k*  N, L)
        score = F.softmax(self.gate(x), dim=-1)  # (BxN, E)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
       
        return torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, k,N, -1).contiguous()

"""
class Kan_mixer(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        super(Kan_mixer, self).__init__()
        self.experts = nn.ModuleList([
            KANLinear(in_features,out_features),
            nn.Linear(in_features, out_features),
            ChebyKANLayer(in_features, out_features, degree=3),
            ChebyKANLayer(in_features, out_features, degree=2),
            TaylorKANLayer(in_features, out_features, order=3, addbias=True)
        ])
        self.n_expert = len(self.experts)
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Linear(in_features, self.n_expert)
        self.k = k  # 要激活的专家数量

    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        
        # 计算专家得分
        score = F.softmax(self.gate(x), dim=-1)  # (BxN, E)
        
        # 只保留前k个得分最高的专家
        if self.k < self.n_expert:
            # 获取前k个得分的阈值
            topk_values, _ = torch.topk(score, self.k, dim=-1)
            min_topk = topk_values[:, -1].unsqueeze(-1)  # 第k大的得分
            
            # 创建掩码，只保留大于等于第k大得分的专家
            mask = (score >= min_topk).float()
            
            # 重新归一化得分
            score = score * mask
            score = score / (score.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 计算专家输出
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
        
        # 加权组合
        return torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).contiguous()
"""

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GPHTBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()

        self.multipiler = configs.GT_pooling_rate[depth]
        self.patch_size = configs.token_len // self.multipiler
        self.down_sample = nn.MaxPool1d(self.multipiler)
        d_model = configs.GT_d_model
        d_ff = configs.GT_d_ff
        self.patch_embedding = PatchEmbedding(d_model, self.patch_size, self.patch_size, 0, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model,
                        configs.n_heads),
                    d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.GT_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.forecast_head = Kan_mixer(d_model, configs.pred_len)

    def forward(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        x_enc = self.down_sample(x_enc)
        enc_out, n_vars = self.patch_embedding.encode_patch(x_enc)
        enc_out = self.patch_embedding.pos_and_dropout(enc_out)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        bs = enc_out.shape[0]
        
        # Decoder
        dec_out = self.forecast_head(enc_out).reshape(bs, n_vars, -1)  # z: [bs x nvars x seq_len]
        # bn = nn.BatchNorm1d(n_vars).to(dec_out.device)
        # dec_out = bn(dec_out) 
        # in_norm = nn.InstanceNorm1d(n_vars).to(dec_out.device)
        # dec_out = in_norm(dec_out)
        return dec_out.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.encoders = nn.ModuleList([
            GPHTBlock(configs, i)
            for i in range(configs.depth)])

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seq_len = x_enc.shape[1]
        dec_out = 0

        for i, enc in enumerate(self.encoders):
            out_enc = enc(x_enc)
            dec_out += out_enc[:, -seq_len:, :]
            ar_roll = torch.zeros((x_enc.shape[0], self.configs.token_len, x_enc.shape[2])).to(x_enc.device)
            ar_roll = torch.cat([ar_roll, out_enc], dim=1)[:, :-self.configs.token_len, :]
            x_enc = x_enc - ar_roll

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
