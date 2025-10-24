import os
import sys
import torch
import esm
import numpy as np
from tqdm import tqdm

class ESM2Extractor:
    """ESM2 feature extractor using the ESM2 model from Facebook AI Research."""
    
    def __init__(self, model_path):
        """
        Initialize the ESM2 model

        Args:
            model_path: Path to the model
        """
        try:
            # Load ESM2 model file directly from local path
            if os.path.exists(model_path):
                logger.info(f"Found ESM2 model file: {model_path}")
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                logger.info(f"Successfully loaded ESM2 model from local path: {model_path}")
            else:
                # Try to load from default path
                model_dir = os.path.dirname(model_path)
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.startswith("esm2_t33_650M_UR50D") and file.endswith(".pt"):
                            full_path = os.path.join(model_dir, file)
                            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(full_path)
                            logger.info(f"Successfully loaded ESM2 model from local path: {full_path}")
                            break
                    else:
                        raise FileNotFoundError(f"ESM2 model file not found in directory: {model_dir}")
                else:
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ESM2 model from local path: {e}")
            try:
                # Finally, try to use the pretrained function (may download the model)
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                logger.info("Successfully loaded ESM2 model from pretrained function")
            except Exception as e2:
                logger.error(f"Failed to load ESM2 model: {e2}")
                raise e2
        
        self.batch_converter = self.alphabet.get_batch_converter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"ESM2 model has been loaded to device: {self.device}")

    def batch_extract_features(self, protein_data, batch_size=8):
        """
        Batch extract ESM2 features

        Args:
            protein_data: [(protein_id, sequence), ...]
            batch_size: Batch size

        Returns:
            Feature dictionary {protein_id: feature vector}
        """
        results = {}

        for i in tqdm(range(0, len(protein_data), batch_size), desc="Extracting ESM2 features"):
            batch = protein_data[i:i+batch_size]

            # ESM input format conversion
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_tokens = batch_tokens.to(self.device)

            # Extract features
            with torch.no_grad():
                output = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                embeddings = output["representations"][33].mean(dim=1)

            # Save results
            for j, (protein_id, _) in enumerate(batch):
                results[protein_id] = embeddings[j].cpu().numpy()
        
        return results
    
    def extract_features(self, proteins_dict, batch_size=8):
        """
        Extract protein features

        Args:
            proteins_dict: Protein dictionary {protein_id: {name, sequence}}
            batch_size: Batch size

        Returns:
            Feature dictionary {protein_id: feature vector}
        """

        # Organize input data
        protein_data = []
        missing_seq_proteins = []
        
        for protein_id, protein_info in proteins_dict.items():
            sequence = protein_info.get("sequence")

            if sequence and sequence.strip().upper() not in ["", "UNKNOWN"]:
                # Limit sequence length, ESM2 supports a maximum of 1024
                if len(sequence) > 1024:  # Leave space for <cls> and <eos>
                    logger.warning(f"Protein {protein_id} sequence is too long and will be truncated: {len(sequence)} -> 1022")
                    sequence = sequence[:1024]
                
                protein_data.append((protein_id, sequence))
            else:
                logger.warning(f"Protein {protein_id} has no sequence data, using default features")
                missing_seq_proteins.append(protein_id)

        # Extract features for proteins with sequences
        features = self.batch_extract_features(protein_data, batch_size) if protein_data else {}

        # Handle proteins with missing sequences
        if missing_seq_proteins:
            logger.info(f"Generating default features for {len(missing_seq_proteins)} proteins with missing sequences")

            # Create a zero vector of dimension 1280 as default features
            default_feature = np.zeros(1280, dtype=np.float32)

            # If existing features are available, use their mean as the default feature
            if features:
                existing_features = np.stack(list(features.values()))
                default_feature = np.mean(existing_features, axis=0)
                logger.info("Using mean of existing protein features as default feature")

            # Assign default features to proteins with missing sequences
            for protein_id in missing_seq_proteins:
                features[protein_id] = default_feature + np.random.normal(0, 0.01, 1280)  # Add a small amount of noise to make each feature unique

        return features