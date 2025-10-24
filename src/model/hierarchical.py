import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    """
    Improved Hierarchical Attention Module with Cross-Attention Mechanism
    1. Intra-type Attention: Compute prototype vectors for each node type
    2. Inter-type Cross-Attention: Information exchange between different node types
    3. Node-Prototype Cross-Attention: Interaction of each node with prototypes of all types
    """
    def __init__(self, hidden_channels, node_types, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = list(node_types)
        self.num_types = len(node_types)
        
        # intra-type attention for each node type
        self.intra_type_attention = nn.ModuleDict()
        for node_type in node_types:
            self.intra_type_attention[node_type] = nn.MultiheadAttention(
                hidden_channels, num_heads, batch_first=True, dropout=dropout
            )

        # inter-type cross-attention
        self.inter_type_cross_attention = nn.MultiheadAttention(
            hidden_channels, num_heads, batch_first=True, dropout=dropout
        )

        # node-prototype cross-attention 
        self.node_prototype_cross_attention = nn.ModuleDict()
        for node_type in node_types:
            self.node_prototype_cross_attention[node_type] = nn.MultiheadAttention(
                hidden_channels, num_heads, batch_first=True, dropout=dropout
            )
        
        # Fusion layer
        self.fusion_layer = nn.ModuleDict()
        for node_type in node_types:
            self.fusion_layer[node_type] = nn.Linear(hidden_channels * 2, hidden_channels)
    
    def forward(self, x_dict, return_weights=False):
        # --- 1. Intra-type Attention: Generate Enhanced Prototypes ---
        enhanced_prototypes = {}
        intra_attention_weights = {}
        
        for node_type, x in x_dict.items():
            # Apply self-attention within each node type
            x_normalized = F.normalize(x, p=2, dim=-1)
            x_input = x_normalized.unsqueeze(0)  # [1, num_nodes, hidden_channels]
            
            enhanced_x, attn_weights = self.intra_type_attention[node_type](
                x_input, x_input, x_input
            )
            
            # enhanced_prototype
            prototype = enhanced_x.mean(dim=1)  # [1, hidden_channels]
            enhanced_prototypes[node_type] = prototype
            intra_attention_weights[node_type] = attn_weights

        # --- 2. Inter-type Cross-Attention ---
        prototype_list = [enhanced_prototypes[nt] for nt in self.node_types]
        prototypes_tensor = torch.cat(prototype_list, dim=0).unsqueeze(0)  # [1, num_types, hidden_channels]

        # Cross-attention: each prototype as query, other prototypes as key and value
        cross_enhanced_prototypes, inter_attention_weights = self.inter_type_cross_attention(
            prototypes_tensor, prototypes_tensor, prototypes_tensor
        )
        cross_enhanced_prototypes = cross_enhanced_prototypes.squeeze(0)  # [num_types, hidden_channels]

        # --- 3. Node-Prototype Cross-Attention ---
        personalized_global_features = {}
        node_prototype_attention_weights = {}
        cross_prototypes_dict = {nt: p for nt, p in zip(self.node_types, cross_enhanced_prototypes)}

        for node_type_q in self.node_types:
            query_nodes = x_dict[node_type_q].unsqueeze(0)  # [1, num_nodes_q, hidden_channels]
            
            kv_prototypes = cross_enhanced_prototypes.unsqueeze(0) # [1, num_types, hidden_channels]

            attn_output, attn_weights = self.node_prototype_cross_attention[node_type_q](
                query_nodes, kv_prototypes, kv_prototypes
            )
            
            original_features = x_dict[node_type_q]
            global_features = attn_output.squeeze(0)
            
            combined = torch.cat([original_features, global_features], dim=-1)
            fused_features = self.fusion_layer[node_type_q](combined)
            
            personalized_global_features[node_type_q] = fused_features
            node_prototype_attention_weights[node_type_q] = attn_weights.squeeze(0) # [num_nodes_q, num_types]

        if return_weights:
            # Return inter-type attention and node-prototype attention weights
            return personalized_global_features, {
                'inter_attention': inter_attention_weights.squeeze(0), 
                'node_prototype_attention': node_prototype_attention_weights
            }
        
        return personalized_global_features