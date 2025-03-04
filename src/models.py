import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, init_method: str = "random", bias: bool = False, n_out: Optional[int] = None):
        '''
            Initialize the MultiHeadAttention module.
            Parameters:
                n_embd (int): Embedding dimension for the input.
                n_head (int): Number of attention heads.
                init_method (str): Method to initialize weights; options include:
                    - "random": Random initialization.
                    - "small_id": Initialize as a small identity matrix.
                    - "oppo_small_id": Initialize half heads with a positive small identity matrix, half with a negative small identity matrix.
                bias (bool): Whether to enable bias in the linear layers.
                n_out (int, optional): Output dimension. If None, the default dimension is n_embd.
        '''
        super(MultiHeadAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.bias = bias
        self.n_out = n_out

        # Define the linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(n_embd, n_embd * n_head, bias=self.bias)
        self.k_proj = nn.Linear(n_embd, n_embd * n_head, bias=self.bias)
        self.v_proj = nn.Linear(n_embd, n_embd * n_head, bias=self.bias)
        self.o_proj = nn.Linear(n_embd * n_head, n_embd, bias=self.bias) if self.n_out is None else nn.Linear(n_embd * n_head, n_out, bias=self.bias)

        # Handle different weight initialization strategies
        if init_method == "random":
            # Default random initialization; no additional action needed
            pass
        elif init_method == "small_id":
            # Initialize projections with a small identity matrix
            self.q_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            self.k_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            self.v_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            if self.n_out is None:
                self.o_proj.weight.data = torch.eye(self.n_embd).repeat(1, self.n_head) * 1e-6
        elif init_method == "small_id_qk":
            # Initialize only the query and key projections as small identity matrices
            self.q_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-4
            self.k_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-4
        elif init_method == "oppo_small_id":
            # Initialize half heads with positive small identity and half with negative
            assert self.n_head % 2 == 0, "The number of heads must be divisible by 2 for 'oppo_small_id' initialization."
            positive_id = torch.eye(self.n_embd) * 1e-6
            negative_id = -torch.eye(self.n_embd) * 1e-6
            q_list = [positive_id for _ in range(self.n_head // 2)] + [negative_id for _ in range(self.n_head // 2)]
            q_tensor = torch.stack(q_list, dim=0).view(self.n_head * self.n_embd, self.n_embd)
            k_list = [positive_id for _ in range(self.n_head)]
            k_tensor = torch.stack(k_list, dim=0).view(self.n_head * self.n_embd, self.n_embd)
            self.q_proj.weight.data = q_tensor.clone()
            self.k_proj.weight.data = k_tensor.clone()
        else:
            raise NotImplementedError("Unsupported initialization method specified.")
        
    def _attn(self, q, k, v):
        '''
            Compute the attention mechanism.
            Parameters:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, n_head * n_embd).
                k (torch.Tensor): Key tensor of the same shape as q.
                v (torch.Tensor): Value tensor of the same shape as q.
            Returns:
                attn_output (torch.Tensor): Output after applying attention.
                attn_weights (torch.Tensor): Attention weights.
        '''
        q_len = q.shape[1]
        ex_len = k.shape[1]
        # Reshape tensors to separate heads
        q = q.view(-1, q_len, self.n_head, self.n_embd)
        k = k.view(-1, ex_len, self.n_head, self.n_embd)
        v = v.view(-1, ex_len, self.n_head, self.n_embd)

        # Permute dimensions to prepare for attention computation
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.n_embd ** (-0.5)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # Normalize attention weights

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

    def forward(self, z_q, z):
        '''
            Forward pass through the multi-head attention layer.
            Parameters:
                z_q (torch.Tensor): Query input of shape (batch_size, 1, n_embd).
                z (torch.Tensor): Key/Value input of shape (batch_size, seq_len, n_embd).
               
            Returns:
                output (torch.Tensor): Attention output.
                
        '''
        batch_size, seq_len, q_len = z.size(0), z.size(1), z_q.size(1)
        q = self.q_proj(z_q)  # Project queries
        k = self.k_proj(z)    # Project keys
        v = self.v_proj(z)    # Project values

        # Compute attention
        attn_output, attn_weights = self._attn(q, k, v)

        # Reshape output to merge heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.n_embd * self.n_head)
        output = self.o_proj(attn_output)  # Project to output space
        return output

    def extract_qk(self):
        '''
            Extract the query and key weight matrices for analysis.
            Returns:
                qk (list of dict): List containing query and key weight matrices for each head.
        '''
        q_matrix = self.q_proj.weight.detach().cpu().transpose(0, 1)
        k_matrix = self.k_proj.weight.detach().cpu().transpose(0, 1)
        q_matrix = q_matrix.view(self.n_embd, self.n_head, self.n_embd)
        k_matrix = k_matrix.view(self.n_embd, self.n_head, self.n_embd)

        qk = []
        for i in range(self.n_head):
            qk.append({'W_Q': q_matrix[:, i, :], 'W_K': k_matrix[:, i, :]})
        return qk

    def extract_ov(self):
        '''
            Extract the output and value weight matrices for analysis.
            Returns:
                ov (list of dict): List containing output and value weight matrices for each head.
        '''
        o_matrix = self.o_proj.weight.detach().cpu().transpose(0, 1)
        v_matrix = self.v_proj.weight.detach().cpu().transpose(0, 1)
        if self.n_out is None:
            o_matrix = o_matrix.view(self.n_head, self.n_embd, self.n_embd)
        else:
            o_matrix = o_matrix.view(self.n_head, self.n_embd, self.n_out)
        v_matrix = v_matrix.view(self.n_embd, self.n_head, self.n_embd)

        ov = []
        for i in range(self.n_head):
            ov.append({'W_O': o_matrix[i, :, :], 'W_V': v_matrix[:, i, :]})
        return ov

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, init_method="random"):
        """
        Multi-layer causal multi-head attention mechanism.
        
        Parameters:
            n_embd (int): Embedding dimension for the input.
            n_head (int): Number of attention heads.
            num_layers (int): Number of layers in the attention stack.
        Output shape: 
        Batchsize * (Length+1) * (d+1)
        """
        super(CausalMultiHeadAttention, self).__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_head = n_embd 
        self.n_out = n_embd
        
        # Define the linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(n_embd, n_embd * n_head, bias=None)
        self.k_proj = nn.Linear(n_embd, n_embd * n_head, bias=None)
        self.v_proj = nn.Linear(n_embd, n_embd * n_head, bias=None)
        self.o_proj = nn.Linear(n_embd * n_head, n_embd, bias=None)

        # buffer for causal masking
        self.register_buffer("bias", torch.tensor(-1e9))
    
    
        # Handle different weight initialization strategies
        if init_method == "random":
            # Default random initialization; no additional action needed
            pass
        elif init_method == "small_id":
            # Initialize projections with a small identity matrix
            self.q_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            self.k_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            self.v_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-6
            if self.n_out is None:
                self.o_proj.weight.data = torch.eye(self.n_embd).repeat(1, self.n_head) * 1e-6
        elif init_method == "small_id_qk":
            # Initialize only the query and key projections as small identity matrices
            self.q_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-4
            self.k_proj.weight.data = torch.eye(self.n_embd).repeat(self.n_head, 1) * 1e-4
        elif init_method == "oppo_small_id":
            # Initialize half heads with positive small identity and half with negative
            assert self.n_head % 2 == 0, "The number of heads must be divisible by 2 for 'oppo_small_id' initialization."
            positive_id = torch.eye(self.n_embd) * 1e-6
            negative_id = -torch.eye(self.n_embd) * 1e-6
            weights_list = [positive_id for _ in range(self.n_head // 2)] + [negative_id for _ in range(self.n_head // 2)]
            weights_tensor = torch.stack(weights_list, dim=0).view(self.n_head * self.n_embd, self.n_embd)
            self.q_proj.weight.data = weights_tensor.clone()
            self.k_proj.weight.data = weights_tensor.clone()
            self.v_proj.weight.data = weights_tensor.clone()
            weights_tensor_o = torch.cat(weights_list, dim=1)  # Shape: (n_embd, n_embd * n_head)
            self.o_proj.weight.data = weights_tensor_o.clone()
        else:
            raise NotImplementedError("Unsupported initialization method specified.")



    
    def forward(self, x):
        """
        Forward pass through the multi-layer attention mechanism with causal masking.
        
        Parameters:
            z_q (torch.Tensor): Query input of shape (batch_size, 1, n_embd).
            z (torch.Tensor): Key/Value input of shape (batch_size, L, n_embd).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, L+1, n_embd).
        """

        batch_size, seq_len, _ = x.size()

        # Project the input to get keys, queries, and values
        k = self.k_proj(x)  # Shape: (batch_size, seq_len, n_embd * n_head)
        q = self.q_proj(x)  # Shape: (batch_size, seq_len, n_embd * n_head)
        v = self.v_proj(x)  # Shape: (batch_size, seq_len, n_embd * n_head)
        
        # Reshape the tensors to separate heads
        k = k.view(batch_size, seq_len, self.n_head, self.d_head)  # Shape: (batch_size, seq_len, n_head, d_head)
        q = q.view(batch_size, seq_len, self.n_head, self.d_head)  # Shape: (batch_size, seq_len, n_head, d_head)
        v = v.view(batch_size, seq_len, self.n_head, self.d_head)  # Shape: (batch_size, seq_len, n_head, d_head)
        
        # Permute the dimensions to get (batch_size, n_head, seq_len, d_head)
        k = k.permute(0, 2, 1, 3)  # Shape: (batch_size, n_head, seq_len, d_head)
        q = q.permute(0, 2, 1, 3)  # Shape: (batch_size, n_head, seq_len, d_head)
        v = v.permute(0, 2, 1, 3)  # Shape: (batch_size, n_head, seq_len, d_head)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.n_embd ** (-0.5)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # Normalize attention weights

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.to(attn_weights.device)

        # Apply causal mask
        causal_mask = causal_mask.bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.bias.to(attn_weights.dtype))

        # Caculate attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # Shape: (batch_size, n_head, seq_len, d_head)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # Shape: (batch_size, n_head, seq_len, d_head)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_len, n_head, d_head)
        attn_output = attn_output.view(batch_size, seq_len, self.n_head * self.d_head)  # Merge heads

        # Project the output
        output = self.o_proj(attn_output)  # Shape: (batch_size, seq_len, n_embd)
         
        return output


    def extract_qk(self):
        '''
            Extract the query and key weight matrices for analysis.
            Returns:
                qk (list of dict): List containing query and key weight matrices for each head.
        '''
        q_matrix = self.q_proj.weight.detach().cpu().transpose(0, 1)
        k_matrix = self.k_proj.weight.detach().cpu().transpose(0, 1)
        q_matrix = q_matrix.view(self.n_embd, self.n_head, self.n_embd)
        k_matrix = k_matrix.view(self.n_embd, self.n_head, self.n_embd)

        qk = []
        for i in range(self.n_head):
            qk.append({'W_Q': q_matrix[:, i, :], 'W_K': k_matrix[:, i, :]})
        return qk

    def extract_ov(self):
        '''
            Extract the output and value weight matrices for analysis.
            Returns:
                ov (list of dict): List containing output and value weight matrices for each head.
        '''
        o_matrix = self.o_proj.weight.detach().cpu().transpose(0, 1)
        v_matrix = self.v_proj.weight.detach().cpu().transpose(0, 1)
        if self.n_out is None:
            o_matrix = o_matrix.view(self.n_head, self.n_embd, self.n_embd)
        else:
            o_matrix = o_matrix.view(self.n_head, self.n_embd, self.n_out)
        v_matrix = v_matrix.view(self.n_embd, self.n_head, self.n_embd)

        ov = []
        for i in range(self.n_head):
            ov.append({'W_O': o_matrix[i, :, :], 'W_V': v_matrix[:, i, :]})

class MultiHeadAttentionStack(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, n_out=1, weight_tying=False):
        """
        Stack multiple layers of attention mechanisms with optional weight tying.
        
        Parameters:
            n_layers (int): Number of attention layers.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            n_out (int): Output dimension for the final linear transformation.
            weight_tying (bool): Whether to tie weights across all layers.
        """
        super(MultiHeadAttentionStack, self).__init__()
        self.n_layers = n_layers
        self.weight_tying = weight_tying
        self.d = d_model - 1
        
        if n_layers == 1:
            # Use the original MultiHeadAttention if there's only one layer
            self.attention_layer = MultiHeadAttention(d_model, num_heads, init_method="random", bias=False, n_out=n_out)
            self.is_single_layer = True
        else:
            self.is_single_layer = False
            if weight_tying:
                # Use a single shared instance of CausalMultiHeadAttention across all layers
                shared_attention_layer = CausalMultiHeadAttention(d_model, num_heads)
                self.attention_layers = nn.ModuleList([shared_attention_layer for _ in range(n_layers)])
            else:
                # Create independent instances of CausalMultiHeadAttention for each layer
                self.attention_layers = nn.ModuleList([CausalMultiHeadAttention(d_model, num_heads) for _ in range(n_layers)])
            self.final_linear = nn.Linear(d_model, n_out, bias=False)

    def forward(self, z_q, z):
        """
        Forward pass through the attention stack.
    
        Parameters:
            inputs (torch.Tensor): Input tensors.
                - Always expect (z_q, z) as inputs (query and key/value pair).
    
        Returns:
            torch.Tensor: Output tensor.
        """
        
    
        if self.is_single_layer:
            # For single-layer case, use the existing MultiHeadAttention logic
            return self.attention_layer(z_q, z)
        else:
            # For multi-layer case, combine z_q and z into one tensor along the sequence dimension
            x = torch.cat((z, z_q), dim=1)  # Shape: (batch_size, seq_len + 1, d_model)
            for layer in self.attention_layers:
                x = layer(x)
            
            # Apply final linear layer transformation
            output = self.final_linear(x)  # Shape: (batch_size, seq_len + 1, n_out)
    
            # Extract the output at the last position (corresponding to z_q)
            output = output[:, -1:, :]  # Shape: (batch_size, 1, n_out)
            return output

    def extract_qk_stack(self):
        """
        Extract the QK matrices for each layer in the stacked MultiHeadAttention model.
        
        Returns:
            list of list: Outer list corresponds to layers, and each inner list contains
                          QK matrices for each head in that layer.
        """
        if self.is_single_layer:
            return [self.attention_layer.extract_qk()]
        else:
            all_qk_matrices = []
            for layer in self.attention_layers:
                qk_matrices = layer.extract_qk()  # Call the extract_qk() method of CausalMultiHeadAttention
                all_qk_matrices.append(qk_matrices)
            return all_qk_matrices

    
    def extract_ov_stack(self):
        """
        Extract the OV matrices for each layer in the stacked MultiHeadAttention model.
        
        Returns:
            list of list: Outer list corresponds to layers, and each inner list contains
                          OV matrices for each head in that layer.
        """
        if self.is_single_layer:
            return [self.attention_layer.extract_ov()]
        else:
            all_ov_matrices = []
            for layer in self.attention_layers:
                ov_matrices = layer.extract_ov()  # Call the extract_ov() method of CausalMultiHeadAttention
                all_ov_matrices.append(ov_matrices)
            return all_ov_matrices

    def extract_final_linear(self):
        """
        Extract the parameters of the final linear layer in the attention stack.
        
        Returns:
            torch.nn.Parameter or None: If multi-layer, returns the parameter tensor of the final linear layer.
                                        If single-layer, prints a message and returns None.
        """
        if self.is_single_layer:
            
            return torch.eye(1).cpu()
        else:
            return self.final_linear.weight.detach().cpu()
