import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_components(model, x_dict, edge_index_dict, return_attention_weights: bool = False):
    """ 
    Extract and return the core components of the model for analysis or visualization.
    """
    
    # 1. Semantic Features
    semantic_features = {
        node_type: model.dropout_layer(model.proj_linears[node_type](x).relu())
        for node_type, x in x_dict.items()
    }
    
    # 2. Global Semantic Features
    inter_type_attn_weights = None
    node_proto_attn_weights = None
    if return_attention_weights:
        global_features, attention_weights = model.global_attention(semantic_features, return_weights=True)
        inter_type_attn_weights = attention_weights['inter_attention']
        node_proto_attn_weights = attention_weights['node_prototype_attention']
    else:
        global_features = model.global_attention(semantic_features)

    # 3. Topological Features
    topology_input = {k: v.clone() for k, v in semantic_features.items()}
    for conv in model.hgt_layers:
        topology_input = conv(topology_input, edge_index_dict)
        topology_input = {nt: model.dropout_layer(feat) for nt, feat in topology_input.items()}
    topology_features = topology_input

    # 4. Adaptive Feature Fusion
    fused_features = {}
    for nt, sem_feat in semantic_features.items():
        weights = F.softmax(model.fusion_weights[nt], dim=0)
        w_global, w_sem, w_topo = weights[0], weights[1], weights[2]
        fused = w_global * global_features[nt] + w_sem * sem_feat + w_topo * topology_features[nt]
        fused_features[nt] = model.dropout_layer(fused)

    # 5. Final Output
    final_x_dict = {nt: model.out_linears[nt](feat) for nt, feat in fused_features.items()}

    result = {
        'semantic': semantic_features,
        'global': global_features,
        'topology': topology_features,
        'fused': fused_features,
        'final': final_x_dict,
    }
    if return_attention_weights:
        result['attention'] = {
            'inter_attention': inter_type_attn_weights,
            'node_prototype_attention': node_proto_attn_weights,
            'fusion_weights': {nt: F.softmax(model.fusion_weights[nt], dim=0).detach().cpu() for nt in model.fusion_weights}
        }
    return result