import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''

#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)

#     def forward(self, q, k, v, mask=None):

#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)

#         attn = self.dropout(F.softmax(attn, dim=-1))
#         output = torch.matmul(attn, v)

#         return output, attn

# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

#         self.dropout = nn.Dropout(dropout)
#         self.batch_norm = nn.BatchNorm1d(d_model)


#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

#         residual = q

#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.

#         q, attn = self.attention(q, k, v, mask=mask)

#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual

#         seq_len = q.size(1)
#         q = self.batch_norm(q.view(-1, self.d_model)).view(-1, seq_len, self.d_model)

#         return q, attn
    
# class Attention(nn.Module):
#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

#     def forward(self, enc_input, slf_attn_mask=None, return_attns=None):
#         enc_output, enc_slf_attn = self.slf_attn(
#             enc_input, enc_input, enc_input, mask=slf_attn_mask)
#         if return_attns:
#             return enc_output, enc_slf_attn
#         return enc_output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class CrossAttention(nn.Module):
    ''' Cross-Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, output_dim=100, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.output_dim = output_dim

        # Query projection from first tensor
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # Key and Value projections from second tensor
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        # Output projection to desired dimension
        self.fc = nn.Linear(n_head * d_v, output_dim, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, tensor1, tensor2, mask=None):
        """
        Cross-attention where:
        - tensor1 provides the query
        - tensor2 provides the keys and values
        
        Args:
            tensor1: [batch_size, num_concepts, d_model]
            tensor2: [batch_size, num_concepts, d_model]
        Returns:
            output: [batch_size, num_concepts, output_dim]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k = tensor1.size(0), tensor1.size(1), tensor2.size(1)

        # Pass through the projections
        # tensor1 -> Query, tensor2 -> Key, Value
        q = self.w_qs(tensor1).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(tensor2).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(tensor2).view(sz_b, len_k, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose and reshape: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        # Project to output dimension
        output = self.dropout(self.fc(q))

        # Apply batch normalization
        seq_len = output.size(1)
        output = self.batch_norm(output.view(-1, self.output_dim)).view(-1, seq_len, self.output_dim)

        return output

class Attention(nn.Module):
    def __init__(self, input_dim=768, output_dim=100, n_head=8, dropout=0.1):
        super().__init__()
        
        # For multi-head attention, typically d_k = d_v = d_model/n_head
        d_k = d_v = input_dim // n_head
        
        self.cross_attention = CrossAttention(
            n_head=n_head,
            d_model=input_dim,  # 768 based on the input
            d_k=d_k,
            d_v=d_v,
            output_dim=output_dim,  # 100 as specified in the output
            dropout=dropout
        )
    
    def forward(self, tensor1, tensor2, mask=None):
        """
        Args:
            tensor1: [batch_size, num_concepts, 768]
            tensor2: [batch_size, num_concepts, 768]
        Returns:
            weights: [batch_size, num_concepts, 100]
        """
        return self.cross_attention(tensor1, tensor2, mask=mask)