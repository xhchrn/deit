import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NystromAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 is_mask=0, num_landmarks=32):  #, seq_len=128**2):
        super().__init__()
        self.head_dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.num_landmarks = num_landmarks
        # self.seq_len = seq_len
        self.proj = nn.Linear(dim, dim)
        # self.use_conv = "conv_kernel_size" in config
        # if self.use_conv:
        #     self.conv = nn.Conv2d(
        #         in_channels = self.num_head, out_channels = self.num_head,
        #         kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
        #         bias = False,
        #         groups = self.num_head)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        # K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        Q = Q / math.sqrt(math.sqrt(C // self.num_heads))
        K = K / math.sqrt(math.sqrt(C // self.num_heads))
        seq_len = N
        # if self.num_landmarks == self.seq_len:
        if self.num_landmarks == seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)), dim = -1)
            X = torch.matmul(attn, V)
        else:
            # Q_landmarks = Q.reshape(B, self.num_heads, self.num_landmarks, self.seq_len // self.num_landmarks, C // self.num_heads).mean(dim = -2)
            # K_landmarks = K.reshape(B, self.num_heads, self.num_landmarks, self.seq_len // self.num_landmarks, C // self.num_heads).mean(dim = -2)
            Q_landmarks = Q.reshape(B, self.num_heads, self.num_landmarks, seq_len // self.num_landmarks, C // self.num_heads).mean(dim = -2)
            K_landmarks = K.reshape(B, self.num_heads, self.num_landmarks, seq_len // self.num_landmarks, C // self.num_heads).mean(dim = -2)
            kernel_1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = F.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) , dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))
        X = X.transpose(1, 2).reshape(B, N, C)
        X = self.proj(X)
        # if self.use_conv:
        #     X += self.conv(V * mask[:, None, :, None])
        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

