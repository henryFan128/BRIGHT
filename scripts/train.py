import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.model import KGCModel
from utils.evaluation import monitor_gradients, compute_ranking_metrics

def get_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description="train script for knowledge graph complement model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file (YAML format)')
    parser.add_argument('--data_dir', type=str, help='path to the data directory')
    parser.add_argument('--model_save_path', type=str, help='path to save the best model')
    parser.add_argument('--hidden_channels', type=int, help='hidden channels for HGT')
    parser.add_argument('--out_channels', type=int, help='output channels')
    parser.add_argument('--num_hgt_layers', type=int, help='number of HGT layers')
    parser.add_argument('--num_hgt_heads', type=int, help='number of attention heads in HGT')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay for AdamW')
    parser.add_argument('--epochs', type=int, help='maximum number of training epochs')
    parser.add_argument('--neg_sampling_ratio', type=float, help='negative sampling ratio')
    parser.add_argument('--warmup_epochs', type=int, help='number of warmup epochs')
    parser.add_argument('--early_stopping_patience', type=int, help='early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, help='early stopping minimum improvement threshold')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use, -1 for CPU')
    parser.add_argument('--k_folds', type=int, help='number of cross-validation folds, set to 1 to disable')
    parser.add_argument('--cv_seed', type=int, help='random seed for cross-validation data splitting')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--grad_clip_norm', type=float, help='maximum norm for gradient clipping')
    parser.add_argument('--monitor_gradients', action='store_true', help='whether to monitor gradient norms')
    parser.add_argument('--ranking_ks', type=str, help='ranking metrics K values, comma separated')
    parser.add_argument('--bce_weight', type=float, help='BCE loss weight (0 to disable)')
    parser.add_argument('--bpr_weight', type=float, help='BPR loss weight (0 to disable)')
    parser.add_argument('--bpr_neg_per_pos', type=int, help='BPR negative samples per positive sample (effective when all_pairs=False)')
    parser.add_argument('--bpr_all_pairs', action='store_true', help='BPR use all positive-negative pairs (may be slower/more memory intensive)')

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
        
        if 'ranking_ks' in defaults and isinstance(defaults['ranking_ks'], list):
            defaults['ranking_ks'] = ','.join(map(str, defaults['ranking_ks']))

        parser.set_defaults(**defaults)
    return parser.parse_args()


args = get_args()
DATA_DIR = Path(args.data_dir)
MODEL_SAVE_PATH = Path(args.model_save_path)
HIDDEN_CHANNELS = args.hidden_channels
OUT_CHANNELS = args.out_channels
NUM_HGT_LAYERS = args.num_hgt_layers
NUM_HGT_HEADS = args.num_hgt_heads
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
EPOCHS = args.epochs
WARMUP_EPOCHS = args.warmup_epochs
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
EARLY_STOPPING_DELTA = args.early_stopping_delta
K_FOLDS = args.k_folds
CV_SEED = args.cv_seed
DROPOUT = args.dropout
GRAD_CLIP_NORM = args.grad_clip_norm
MONITOR_GRADIENTS = args.monitor_gradients
RANKING_KS = [int(k.strip()) for k in args.ranking_ks.split(',') if k.strip().isdigit()]
BCE_WEIGHT = args.bce_weight
BPR_WEIGHT = args.bpr_weight
BPR_NEG_PER_POS = args.bpr_neg_per_pos
BPR_ALL_PAIRS = args.bpr_all_pairs

if args.gpu_id >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.gpu_id}')
else:
    DEVICE = torch.device('cpu')


def build_all_true_tails(datasets, edge_type):
    """ 
    Construct the mapping from (head, relation) to all true tail entities
        head -> set(true tails)
    Returns:
        all_true_tails: dict mapping (head, relation) to set of true tail entities
    """

    all_true = defaultdict(set)
    for data in datasets:
        if edge_type not in data.edge_types:
            continue
        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label
        if edge_label_index is None or edge_label is None:
            continue
        pos_mask = (edge_label == 1)
        if pos_mask.sum() == 0:
            continue
        pos_edges = edge_label_index[:, pos_mask]
        for h, t in pos_edges.t().tolist():
            all_true[h].add(t)
    return all_true


print("--- Configurations of training ---")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print(f"Using device: {DEVICE}")
print("-----------------------------------")

# --- Data loader  ---
print("Loading data...")
try:
    train_data = torch.load(DATA_DIR / "train_data.pt", map_location='cpu', weights_only=False)
    valid_data = torch.load(DATA_DIR / "val_data.pt", map_location='cpu', weights_only=False)
    test_data = torch.load(DATA_DIR / "test_data.pt", map_location='cpu', weights_only=False)
except FileNotFoundError:
    print(f"Error: Data files not found in {DATA_DIR}. Please ensure the data is prepared and placed correctly.")
    exit()

train_data = train_data.to(DEVICE)
valid_data = valid_data.to(DEVICE)
test_data = test_data.to(DEVICE)    

print("Data loaded successfully.")
print(f"Train data: {train_data}")
print(f"Valid data: {valid_data}")
print(f"Test data: {test_data}")

print("\nThe graph structure of the valid and test to prevent data leakage.")
edge_type_to_predict = ('drug', 'treats', 'disease')
if edge_type_to_predict in valid_data.edge_types:
    valid_data[edge_type_to_predict].edge_index = train_data[edge_type_to_predict].edge_index

if edge_type_to_predict in test_data.edge_types:
    test_data[edge_type_to_predict].edge_index = train_data[edge_type_to_predict].edge_index

print("\nFinished fix the graph structure")


# --- Calculate the ratio of positive and negative samples in the training set ---
train_labels = train_data[edge_type_to_predict].edge_label
num_pos_samples = torch.sum(train_labels == 1).item()
num_neg_samples = torch.sum(train_labels == 0).item()

if num_pos_samples > 0:
    neg_sampling_ratio = num_neg_samples / num_pos_samples
else:
    neg_sampling_ratio = 1.0 

print(f"\nRatio of negative to positive samples: {neg_sampling_ratio:.2f}")

# --- Construct the full set of positive samples for Filtered Ranking ---
print("\n Building all true tails for filtered ranking...")
all_true_tails = build_all_true_tails([train_data, valid_data, test_data], edge_type_to_predict)
print("Finished building all true tails.")

@torch.no_grad()
def evaluate(model, data, edge_index_dict_for_message_passing, neg_sampling_ratio, all_true_tails):
    """ 
    Args:
        model: the KGCModel to evaluate
        data: the data object containing the graph and edge labels
        edge_index_dict_for_message_passing: dict of edge_index for message passing
        neg_sampling_ratio: negative sampling ratio used in loss computation
        all_true_tails: dict mapping (head, relation) to set of true tail entities for filtered ranking  
    """    

    model.eval()

    z_dict = model(data.x_dict, edge_index_dict_for_message_passing)
    edge_label_index = data[edge_type_to_predict].edge_label_index
    edge_label = data[edge_type_to_predict].edge_label

    loss = model.loss(
        z_dict,
        edge_label_index,
        edge_label,
        neg_sampling_ratio=neg_sampling_ratio,
        bce_weight=BCE_WEIGHT,
        bpr_weight=BPR_WEIGHT,  
        bpr_neg_per_pos=BPR_NEG_PER_POS,
        bpr_all_pairs=BPR_ALL_PAIRS
    )

    predicted_scores_logits = model.decode(z_dict, edge_label_index)
    pred_probs = predicted_scores_logits.sigmoid()

    true_labels = data[edge_type_to_predict].edge_label
    
    y_true = true_labels.cpu().numpy()
    y_pred_probs = pred_probs.cpu().numpy()

    auc = roc_auc_score(y_true, y_pred_probs)
    aupr = average_precision_score(y_true, y_pred_probs)

    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    base_metrics = {
        'Loss': loss.item(),
        'AUC': auc,
        'AUPR': aupr,
        'Accuracy': acc,
        'Recall': recall
    }

    if (edge_label == 1).any():
        ranking_metrics = compute_ranking_metrics(
            model, 
            data, 
            edge_index_dict_for_message_passing, 
            RANKING_KS, 
            all_true_tails, 
            edge_type = edge_type_to_predict
        )

        base_metrics.update(ranking_metrics)
    else:
        for k in RANKING_KS:
            base_metrics[f'Hits@{k}'] = 0.0
            base_metrics[f'Precision@{k}'] = 0.0
            base_metrics[f'Recall@{k}'] = 0.0
            base_metrics[f'NDCG@{k}'] = 0.0
        base_metrics['MRR'] = 0.0
    return base_metrics


def init_model_and_optimizer():
    """ Initialize the model, optimizer, and learning rate scheduler """

    model = KGCModel(
        data = train_data,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_hgt_layers=NUM_HGT_LAYERS,
        num_hgt_heads=NUM_HGT_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )

    return model, optimizer, scheduler


def train_and_evaluate_model(train_data, valid_data, test_data, model_save_path):
    """ Train and evaluate the model """
    model, optimizer, scheduler = init_model_and_optimizer()
    
    print("Finish the initialization of model and optimizer.")
    print(model)
    
    print("\n--- Start training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    val_neg_sampling_ratio = 1.0  
    
    train_labels = train_data[edge_type_to_predict].edge_label
    num_pos_samples = torch.sum(train_labels == 1).item()
    num_neg_samples = torch.sum(train_labels == 0).item()
    neg_sampling_ratio = num_neg_samples / num_pos_samples if num_pos_samples > 0 else 1.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # forward 
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        # calculate loss
        edge_label_index = train_data[edge_type_to_predict].edge_label_index
        edge_label = train_data[edge_type_to_predict].edge_label

        loss = model.loss(z_dict, edge_label_index, edge_label, neg_sampling_ratio,
                          bce_weight=BCE_WEIGHT, bpr_weight=BPR_WEIGHT,
                          bpr_neg_per_pos=BPR_NEG_PER_POS, bpr_all_pairs=BPR_ALL_PAIRS)
        
        # backward
        loss.backward()
        
        # monitor gradients (optional)
        if MONITOR_GRADIENTS:
            grad_norm_before = monitor_gradients(model)

        # gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

        # monitor clipped gradient norm (optional)
        if MONITOR_GRADIENTS:
            grad_norm_after = monitor_gradients(model)
            if epoch <= 10 or epoch % 50 == 0:  # only print for first 10 epochs or every 50 epochs
                print(f"  [Epoch {epoch}] Gradient Norm: {grad_norm_before:.4f} -> {grad_norm_after:.4f}")

        optimizer.step()

        # update learning rate
        scheduler.step()

        # --- validation and early stopping ---
        if epoch % 5 == 0 or epoch == 1:
            # during evaluation, use validation labels and edges, but with training graph structure
            val_metrics = evaluate(model, valid_data, train_data.edge_index_dict, val_neg_sampling_ratio,
                                   all_true_tails=build_all_true_tails([train_data, valid_data, test_data], edge_type_to_predict))
            val_loss = val_metrics["Loss"]
            print(
                f"[Epoch {epoch:03d}] | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_metrics['Loss']:.4f} | "
                f"Val AUC: {val_metrics['AUC']:.4f} | "
                f"Val AUPR: {val_metrics['AUPR']:.4f} | "
                f"Val Acc: {val_metrics['Accuracy']:.4f} | "
                f"Val Recall: {val_metrics['Recall']:.4f} | "
                f"Val MRR: {val_metrics.get('MRR',0):.4f} | "
                f"Val Hits@10: {val_metrics.get('Hits@10',0):.4f}"
            )
            
            if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # save the best model
                save_dir = Path(model_save_path).parent
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), str(model_save_path))
                print(f"  -> Loss of the optimal valid: {best_val_loss:.4f}; Model has saved:'{model_save_path}'")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nValidation loss has not improved for {EARLY_STOPPING_PATIENCE}*5 epochs, triggering early stopping.")
                break

    print("--- Training complete ---")

    # Load the best model
    print(f"\n Loading the best model from '{model_save_path}'...")
    try:
        model.load_state_dict(torch.load(str(model_save_path), map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Could not find the saved model file: {model_save_path}. Using the model at the end of training for testing.")
    
    # Evaluate the best model on the test set
    test_neg_sampling_ratio = 1.0
    test_metrics = evaluate(model, test_data, train_data.edge_index_dict, test_neg_sampling_ratio,
                            all_true_tails=build_all_true_tails([train_data, valid_data, test_data], edge_type_to_predict))

    print("\n--- Test Set Evaluation Results (Classification + Ranking) ---")
    for metric_name, metric_value in test_metrics.items():
        print(f"  - {metric_name}:".ljust(16) + f"{metric_value:.4f}")
    
    return test_metrics


def run_cross_validation(full_train_data, val_data, test_data, k_folds, seed):
    """ Run cross-validation for the model. """

    if k_folds <= 1:
        print("\n Single run without cross-validation.")
        
        test_metrics = train_and_evaluate_model(
            train_data=full_train_data,
            valid_data=val_data,
            test_data=test_data,
            model_save_path=MODEL_SAVE_PATH
        )
        return
    
    print(f"\n {k_folds} cross-validation in progress...")
    
    edge_label_index = torch.cat([
        full_train_data[edge_type_to_predict].edge_label_index,
        val_data[edge_type_to_predict].edge_label_index
    ], dim=1)
    
    edge_label = torch.cat([
        full_train_data[edge_type_to_predict].edge_label,
        val_data[edge_type_to_predict].edge_label
    ])
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    edge_label_np = edge_label.cpu().numpy()
    
    all_fold_metrics = {
        "Loss": [], "AUC": [], "AUPR": [], "Accuracy": [], "Recall": [], "MRR": [],
        **{f"Hits@{k}": [] for k in RANKING_KS},
        **{f"Precision@{k}": [] for k in RANKING_KS},
        **{f"Recall@{k}": [] for k in RANKING_KS},
        **{f"NDCG@{k}": [] for k in RANKING_KS},
    }
    
    best_global_val_loss = float('inf')
    best_global_test_metrics = None
    best_global_fold = -1
    best_fold_train_data = None
    best_fold_val_data = None
    best_fold_test_data = None
    
    # K folds for cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(edge_label_np)), edge_label_np)):
        print(f"\n======== The {fold + 1}/{k_folds} fold ========")
        
        fold_train_data = full_train_data.clone()
        fold_val_data = val_data.clone()
        
        fold_train_edge_index = edge_label_index[:, train_idx]
        fold_train_edge_label = edge_label[train_idx]
        
        fold_val_edge_index = edge_label_index[:, val_idx]
        fold_val_edge_label = edge_label[val_idx]
        
        fold_train_data[edge_type_to_predict].edge_label_index = fold_train_edge_index
        fold_train_data[edge_type_to_predict].edge_label = fold_train_edge_label
        
        fold_val_data[edge_type_to_predict].edge_label_index = fold_val_edge_index
        fold_val_data[edge_type_to_predict].edge_label = fold_val_edge_label
        
        fold_val_data[edge_type_to_predict].edge_index = fold_train_data[edge_type_to_predict].edge_index
        
        fold_test_data = test_data.clone()
        fold_test_data[edge_type_to_predict].edge_index = fold_train_data[edge_type_to_predict].edge_index
        
        temp_model_path = str(MODEL_SAVE_PATH).replace(".pt", f"_fold{fold+1}_temp.pt")
        
        fold_metrics = train_and_evaluate_model(
            train_data=fold_train_data,
            valid_data=fold_val_data,
            test_data=fold_test_data,
            model_save_path=temp_model_path
        )
        
        for metric_name, metric_value in fold_metrics.items():
            if metric_name in all_fold_metrics:
                all_fold_metrics[metric_name].append(metric_value)
        
        model, _, _ = init_model_and_optimizer()
        try:
            model.load_state_dict(torch.load(temp_model_path, map_location=DEVICE))
            val_metrics = evaluate(model, fold_val_data, fold_train_data.edge_index_dict, 1.0,
                                   all_true_tails=build_all_true_tails([fold_train_data, fold_val_data, fold_test_data], edge_type_to_predict))
            current_val_loss = val_metrics["Loss"]

            print(f"The test metrics of fold {fold+1}:")
            
            # Update the global best model if the current fold's validation loss is lower
            if current_val_loss < best_global_val_loss:
                best_global_val_loss = current_val_loss
                best_global_test_metrics = fold_metrics
                best_global_fold = fold + 1
                best_fold_train_data = fold_train_data.clone()
                best_fold_val_data = fold_val_data.clone()
                best_fold_test_data = fold_test_data.clone()
                
                # Copy the current best model to the global best model path
                save_dir = Path(MODEL_SAVE_PATH).parent
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
                print(f" -> New global best model (from fold {fold+1}) saved to '{MODEL_SAVE_PATH}'")
                
            # Delete temporary model file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
        except Exception as e:
            print(f"Error evaluating fold {fold+1} model: {str(e)}")

    print("\n============ Cross-validation Summary ============")

    print("Average metrics and standard deviation on the test set for each fold:")

    for metric_name, metric_values in all_fold_metrics.items():
        mean_value = np.mean(metric_values)
        std_value = np.std(metric_values)
        print(f"  - {metric_name}:".ljust(14) + f"{mean_value:.4f} ± {std_value:.4f}")

    print(f"\nThe global best model comes from fold {best_global_fold}, saved to '{MODEL_SAVE_PATH}'")
    print("The global best model's performance on the test set:")
    for metric_name, metric_value in best_global_test_metrics.items():
        print(f"  - {metric_name}:".ljust(14) + f"{metric_value:.4f}")

    print("\nCross-validation completed.")


edge_type_to_predict = ('drug', 'treats', 'disease')

def main():
    if K_FOLDS > 1:
        print(f"\n--- Start {K_FOLDS}-Fold Cross-Validation ---")
        run_cross_validation(train_data, valid_data, test_data, K_FOLDS, CV_SEED)
    else:
        print("\n--- Start Single Training and Evaluation ---")
        test_metrics = train_and_evaluate_model(train_data, valid_data, test_data, MODEL_SAVE_PATH)

        print("\n--- Single Run Final Test Results ---")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
