import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

import sys
import os
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]       
_MODEL = _SRC / "model"                           
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_MODEL) not in sys.path:
    sys.path.insert(0, str(_MODEL))

from residual import Residual                   
from hierarchical import HierarchicalAttention
from utils.loss import combined_loss
from utils.components import extract_components


class KGCModel(nn.Module):
    """
    Combined with HierarchicalAttention and HGT for node representation learning and link prediction.
    This model integrates:
    1. Node Semantic Feature Extraction (FFN)
    2. Global Semantic Feature Extraction (Hierarchical Attention)
    3. Topological Feature Extraction (HGT with Residuals)
    4. Feature Fusion with Learnable Weights
    5. Final Output Layer
    6. DistMult Decoder for Link Prediction
    7. Loss Functions: Weighted BCE and BPR Loss
    """
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int, num_hgt_layers: int, num_hgt_heads: int, dropout: float = 0.1):
        """
        Args:
            data (HeteroData): graph structure and metadata
            hidden_channels (int): hidden dimension for GNN and FFN.
            out_channels (int): output dimension for final node representations, used in decoder.
            num_hgt_layers (int): number of HGT layers.
            num_hgt_heads (int): number of attention heads in HGT.
            dropout (float): dropout rate for regularization, default 0.1.
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_hgt_layers = num_hgt_layers
        self.num_hgt_heads = num_hgt_heads
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(dropout)

        # --- 1. Node Semantic Feature Extraction (FFN) ---
        # Construct a linear layer for each node type to project input features to hidden dimension
        self.proj_linears = nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data[node_type].x.shape[1]
            self.proj_linears[node_type] = Linear(in_channels, hidden_channels)

        # --- 2. Global Feature Extraction (Hierarchical Attention) ---
        self.global_attention = HierarchicalAttention(hidden_channels, 
                                                      data.node_types, 
                                                      num_heads=num_hgt_heads, 
                                                      dropout=dropout)

        # --- 3. Topological Feature Extraction (HGT with Residuals) ---
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_hgt_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_hgt_heads)
            self.hgt_layers.append(Residual(conv))

        # --- 4. Feature Fusion Weights ---
        # Define fusion weights for each node type: [w_global, w_semantic, w_topology]
        self.fusion_weights = nn.ParameterDict()
        for node_type in data.node_types:
            self.fusion_weights[node_type] = nn.Parameter(torch.randn(3))

        # --- 5. Final Output Layer ---
        # Map the fused features (hidden_channels) to the final output dimension (out_channels)
        self.out_linears = nn.ModuleDict()
        for node_type in data.node_types:
            self.out_linears[node_type] = Linear(hidden_channels, out_channels)

        # --- 6. Relation Representation for Decoder ---
        # We only care about the 'treats' relation
        self.relation_embedding = nn.ParameterDict({
            'treats': nn.Parameter(torch.randn(1, out_channels))
        })

        # Initialize relation embedding
        nn.init.xavier_uniform_(self.relation_embedding['treats'])

    def forward(self, x_dict, edge_index_dict, return_attention_weights=False):
        """Model forward propagation"""

        # --- 1. Node Semantic Feature Extraction ---
        semantic_features = {
            node_type: self.dropout_layer(self.proj_linears[node_type](x).relu())
            for node_type, x in x_dict.items()
        }

        # --- 2. Global Semantic Feature Extraction ---
        inter_type_attn_weights = None
        node_proto_attn_weights = None
        if return_attention_weights:
            global_features, attention_weights = self.global_attention(
                semantic_features, return_weights=True
            )
            
            inter_type_attn_weights = attention_weights['inter_attention']
            node_proto_attn_weights = attention_weights['node_prototype_attention']
        else:
            global_features = self.global_attention(semantic_features)

        # --- 3. Topological Feature Extraction ---
        topology_input = {k: v.clone() for k, v in semantic_features.items()}
        for conv in self.hgt_layers:
            topology_input = conv(topology_input, edge_index_dict)
            topology_input = {
                node_type: self.dropout_layer(feat) 
                for node_type, feat in topology_input.items()
            }
        topology_features = topology_input
    
        # --- 4. Adaptive Feature Fusion ---
        fused_features = {}
        for node_type, sem_feat in semantic_features.items():
            weights = F.softmax(self.fusion_weights[node_type], dim=0)
            w_global, w_semantic, w_topology = weights[0], weights[1], weights[2]
            
            global_feat = global_features[node_type]

            fused_feat = (
                w_global * global_feat +
                w_semantic * sem_feat +
                w_topology * topology_features[node_type]
            )
            
            fused_features[node_type] = self.dropout_layer(fused_feat)
    
        final_x_dict = {
            node_type: self.out_linears[node_type](x)
            for node_type, x in fused_features.items()
        }
            
        if return_attention_weights:
            return final_x_dict, {
                'inter_attention': inter_type_attn_weights,
                'node_prototype_attention': node_proto_attn_weights,
                'fusion_weights': {nt: F.softmax(self.fusion_weights[nt], dim=0).detach().cpu() for nt in self.fusion_weights}
            }
        else:
            return final_x_dict

    def extract_components(self, x_dict, edge_index_dict, return_attention_weights=True):
        """
        Extract and return the core components of the model for analysis or visualization.
        """

        return extract_components(
            self, 
            x_dict, 
            edge_index_dict, 
            return_attention_weights
        )
    
    def decode(self, z_dict, edge_label_index):
        """
        DistMult decoder for link prediction 
        Args:
            z_dict (Dict[str, Tensor]): Contains the final representations of the nodes.
            edge_label_index (Tensor): Shape [2, num_edges] edge index for scoring.
                                       Here the edges are of type ('drug', 'treats', 'disease').

        Returns:
            Tensor: Predicted scores for each edge.
        """ 

        drug_emb = z_dict['drug'][edge_label_index[0]]
        disease_emb = z_dict['disease'][edge_label_index[1]]
        relation_emb = self.relation_embedding['treats']
        
        score = (drug_emb * relation_emb * disease_emb).sum(dim=-1)
        
        return score

    def compute_all_scores(self, head_embeds, all_tail_embeds):
        """
        Compute scores for all possible tail entities for link prediction.
        
        Args:
            head_embeds (Tensor): [batch_size, dim]
            all_tail_embeds (Tensor): [num_entities, dim]
            
        Returns:
            Tensor: [batch_size, num_entities]
        """

        relation_emb = self.relation_embedding['treats']
        head_rel = head_embeds * relation_emb  # [batch_size, dim]
        scores = torch.matmul(head_rel, all_tail_embeds.t())
        
        return scores

    def loss(self, 
             z_dict,
             edge_label_index,
             edge_label,
             neg_sampling_ratio: float = 1.0, 
             bce_weight: float = 1.0,
             bpr_weight: float = 0.0,
             bpr_neg_per_pos: int = 1,
             bpr_all_pairs: bool = False
             ):
        scores = self.decode(z_dict, edge_label_index)

        total_loss, _ = combined_loss(
            scores, 
            edge_label, 
            neg_sampling_ratio=neg_sampling_ratio,
            bce_weight=bce_weight,
            bpr_weight=bpr_weight,
            bpr_neg_per_pos=bpr_neg_per_pos,
            bpr_all_pairs=bpr_all_pairs
        )
        return total_loss