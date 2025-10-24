import sys
import os
from pathlib import Path

import yaml
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.model import KGCModel


def get_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description="inference script for knowledge graph complement model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file (YAML format)')
    parser.add_argument('--graph_data', type=str, help='path to the graph data file')
    parser.add_argument('--disease_mapping', type=str, help='path to the disease mapping file')
    parser.add_argument('--drug_mapping', type=str, help='path to the drug mapping file')
    parser.add_argument('--model_path', type=str, help='path to the trained model file')
    parser.add_argument('--disease', type=str, help='disease name for inference')
    parser.add_argument('--hidden_channels', type=int, help='hidden channels for HGT')
    parser.add_argument('--out_channels', type=int, help='output channels')
    parser.add_argument('--num_hgt_layers', type=int, help='number of HGT layers')
    parser.add_argument('--num_hgt_heads', type=int, help='number of attention heads in HGT')
    parser.add_argument('--top_k', type=int, help='top k predictions to return')
    parser.add_argument('--output', type=str, help='path to save recommendations (csv/tsv). If omitted, only print.')
    parser.add_argument('--verbose', action='store_true', help='whether to print detailed logs')
    parser.add_argument('--use_raw_scores', action='store_true', help='whether to use raw scores instead of probabilities')

    temp_args, _ = parser.parse_known_args()

    if os.path.exists(temp_args.config):
        print(f"Loading the config from '{temp_args.config}' ...")

        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f)

        defaults = {}
        if config:
            for key, value in config.items():
                if isinstance(value, dict):
                    defaults.update(value)
                else:
                    defaults[key] = value
        parser.set_defaults(**defaults)
    return parser.parse_args()

def load_mappings(file_path, name_col='Name', index_col='Graph_Index'):
    """ Load entity mappings from a CSV file """
    try:
        df = pd.read_csv(file_path)
        mapping = df.set_index(name_col)[index_col].to_dict()
        return mapping
    except FileNotFoundError:
        raise FileNotFoundError(f"Not found the mapping file: '{file_path}'")
    except KeyError as e:
        raise KeyError(f"CSV file '{file_path}' is missing required column: {e}")
    except Exception as e:
        raise Exception(f"Error occurred while loading mapping file '{file_path}': {e}")
    
def recommend_drugs(model, full_graph, disease_mapping, drug_mapping, disease_name, top_k=10, use_probability=True, verbose=False):
    """
    Use the trained model to recommend drugs for a given disease.

    Args:
        model (KGCModel): Trained knowledge graph completion model
        full_graph (HeteroData): Complete graph data object
        disease_mapping (dict): Mapping from disease names to graph indices
        drug_mapping (dict): Mapping from drug names to graph indices
        disease_name (str): Name of the disease to find treatment for
        top_k (int): Number of top drug recommendations to return
        use_probability (bool): Whether to convert scores to probabilities
        verbose (bool): Whether to print detailed information

    Returns:
        list: A list of tuples containing the top_k recommended (drug_name, score, probability)
    """
    model.eval()

    # Create a reverse mapping from drug indices to names
    drug_rev_mapping = {v: k for k, v in drug_mapping.items()}

    # Validate disease name
    if disease_name not in disease_mapping:
        available_diseases = list(disease_mapping.keys())[:5]
        raise ValueError(f"Not found the disease '{disease_name}' in the mapping. Available examples: {available_diseases}")

    disease_idx = disease_mapping[disease_name]
    
    with torch.no_grad():
        if verbose:
            print(f"Calculating recommendations for disease '{disease_name}' (Index: {disease_idx})...")

        # 1. Get embeddings for all nodes
        z_dict = model(full_graph.x_dict, full_graph.edge_index_dict)

        # 2. Prepare input for batch prediction
        drug_embeddings = z_dict['drug']
        num_drugs = drug_embeddings.size(0)

        # Create edge indices for all drug-disease pairs
        drug_indices = torch.arange(num_drugs, device=drug_embeddings.device)
        disease_indices = torch.tensor([disease_idx], device=z_dict['disease'].device).repeat(num_drugs)
        edge_label_index = torch.stack([drug_indices, disease_indices], dim=0)

        # 3. Use decoder to compute treatment scores for all drugs
        raw_scores = model.decode(z_dict, edge_label_index)

        # 4. Process scores
        edge_type_to_filter = ('drug', 'treats', 'disease')
        if edge_type_to_filter in full_graph.edge_index_dict:
            all_known_edges = full_graph.edge_index_dict[edge_type_to_filter]
            
            disease_mask = (all_known_edges[1] == disease_idx)
            known_drug_indices = all_known_edges[0][disease_mask]
            raw_scores[known_drug_indices] = -float('inf')

        if use_probability:
            # Convert logits to probabilities
            probability_scores = torch.sigmoid(raw_scores)
            scores_for_ranking = probability_scores
            score_type = "Probability"
        else:
            # Use raw scores
            scores_for_ranking = raw_scores
            score_type = "Raw Score"

        # 5. Get Top-K recommendations
        top_scores, top_indices = torch.topk(scores_for_ranking, k=min(top_k, num_drugs), largest=True)
        
        if verbose:
            scores_std = raw_scores.std().item()
            unique_scores = len(torch.unique(raw_scores))
            print(f"Calculating recommendations for disease '{disease_name}' (Index: {disease_idx})...")
            print(f"  - Raw Score Std: {scores_std:.6f}")
            print(f"  - Unique Raw Scores: {unique_scores}/{num_drugs}")
            print(f"  - Raw Score Range: [{raw_scores.min().item():.6f}, {raw_scores.max().item():.6f}]")

            if use_probability:
                prob_scores = torch.sigmoid(raw_scores)
                print(f"  - Probability Range: [{prob_scores.min().item():.6f}, {prob_scores.max().item():.6f}]")

    # 6. Build recommendations
    recommendations = []
    for i in range(len(top_scores)):
        drug_idx = top_indices[i].item()
        drug_name = drug_rev_mapping.get(drug_idx, f"Unknown_Drug_{drug_idx}")
        
        if use_probability:
            score = top_scores[i].item()
            raw_score = raw_scores[drug_idx].item()
            recommendations.append((drug_name, score, raw_score))
        else:
            score = top_scores[i].item()
            prob_score = torch.sigmoid(raw_scores[drug_idx]).item()
            recommendations.append((drug_name, score, prob_score))
    
    return recommendations, score_type

def main():
    args = get_args()

    # Load mappings
    disease_mapping = load_mappings(args.disease_mapping)
    drug_mapping = load_mappings(args.drug_mapping)

    # Load graph data
    if not os.path.exists(args.graph_data):
        raise FileNotFoundError(f"Graph data file '{args.graph_data}' does not exist.")
    
    try:
        graph_data = torch.load(args.graph_data, weights_only=False)
        if not isinstance(graph_data, HeteroData):
            raise ValueError(f"Loaded graph data is not a HeteroData object.")
    except Exception as e:
        raise Exception(f"Error loading graph data from '{args.graph_data}': {e}")

    # Initialize model
    model = KGCModel(
        data = graph_data,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_hgt_layers=args.num_hgt_layers,
        num_hgt_heads=args.num_hgt_heads
    )

    # Load trained model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file '{args.model_path}' does not exist.")
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        if args.verbose:
            print(f"Loaded model weights from '{args.model_path}'.")
    except Exception as e:
        raise Exception(f"Error loading model weights from '{args.model_path}': {e}")

    # Perform recommendation
    recommendations, score_type = recommend_drugs(
        model=model,
        full_graph=graph_data,
        disease_mapping=disease_mapping,
        drug_mapping=drug_mapping,
        disease_name=args.disease,
        top_k=args.top_k,
        use_probability=not args.use_raw_scores,
        verbose=args.verbose
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, (drug_name, score, other) in enumerate(recommendations, start=1):
            if score_type == "Probability":
                rows.append({
                    "Rank": rank,
                    "Disease": args.disease,
                    "Drug": drug_name,
                    "Probability": float(score),
                    "RawScore": float(other)
                })
            else:
                rows.append({
                    "Rank": rank,
                    "Disease": args.disease,
                    "Drug": drug_name,
                    "RawScore": float(score),
                    "Probability": float(other)
                })
        import pandas as pd
        df_out = pd.DataFrame(rows)
        sep = '\t' if out_path.suffix.lower() == '.tsv' else ','
        df_out.to_csv(out_path, index=False, sep=sep)
        print(f"Saved recommendations to: {out_path}")
        
    # Print recommendations
    print(f"\nTop-{args.top_k} drug recommendations for disease '{args.disease}':")
    print(f"{'Rank':<5} {'Drug Name':<30} {score_type:<15} {'Raw Score':<15}")
    print("-" * 70)
    for rank, (drug_name, score, raw_score) in enumerate(recommendations, start=1):
        print(f"{rank:<5} {drug_name:<30} {score:<15.6f} {raw_score:<15.6f}")
    print("-" * 70)

if __name__ == "__main__":
    main()