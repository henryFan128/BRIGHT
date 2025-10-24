import os
import sys
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class BioClinicalBERTExtractor:
    """Bio_ClinicalBERT feature extractor"""
    
    def __init__(self, model_path):
        """
        Initialize the Bio_ClinicalBERT feature extractor.
        
        Args:
            model_path: Path to the model directory
        """
        try:
            if os.path.exists(model_path):
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    local_files_only=True,
                    trust_remote_code=False
                )
                self.model = AutoModel.from_pretrained(
                    model_path, 
                    local_files_only=True,
                    trust_remote_code=False
                )
                logger.info(f"Successfully loaded Bio_ClinicalBERT model from local path: {model_path}")
            else:
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Bio_ClinicalBERT model from local path: {e}")
            try:
                logger.warning("Attempting to load model from Hugging Face, this may require downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                logger.info("Successfully loaded Bio_ClinicalBERT model from Hugging Face")
            except Exception as e2:
                logger.error(f"Failed to load Bio_ClinicalBERT model from Hugging Face: {e2}")
                raise e2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Bio_ClinicalBERT model has been loaded to device: {self.device}")

    def batch_extract_features(self, descriptions, batch_size=32):
        """
        Batch extract Bio_ClinicalBERT features
        
        Args:
            descriptions: List of disease descriptions
            batch_size: Batch size

        Returns:
            Feature array (n_samples, 768)
        """
        all_features = []

        for i in tqdm(range(0, len(descriptions), batch_size), desc="Extracting Bio_ClinicalBERT features"):
            batch_descriptions = descriptions[i:i+batch_size]

            # Handle empty descriptions
            batch_descriptions = [desc if desc else "Unknown disease" for desc in batch_descriptions]
            
            inputs = self.tokenizer(batch_descriptions, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.pooler_output.cpu().numpy()
            
            all_features.append(batch_features)
        
        return np.vstack(all_features) if all_features else np.array([])
    
    def extract_features(self, diseases_dict, batch_size=32):
        """
        Extract disease features

        Args:
            diseases_dict: Disease dictionary {disease_id: {name, description}}
            batch_size: Batch size

        Returns:
            Feature dictionary {disease_id: feature vector}
        """

        # Organize description list and ID mapping
        disease_ids = list(diseases_dict.keys())
        missing_desc_diseases = []
        valid_disease_ids = []
        descriptions = []
        
        for disease_id in disease_ids:
            desc = diseases_dict[disease_id].get("description", diseases_dict[disease_id].get("name", ""))
            if desc:
                valid_disease_ids.append(disease_id)
                descriptions.append(desc)
            else:
                missing_desc_diseases.append(disease_id)
                logger.warning(f"Disease {disease_id} has no description or name data, using default features")

        # Extract features for diseases with descriptions
        if descriptions:
            features = self.batch_extract_features(descriptions, batch_size)

            # Combine results
            disease_features = {}
            for i, disease_id in enumerate(valid_disease_ids):
                disease_features[disease_id] = features[i]

            # Handle diseases with missing descriptions
            if missing_desc_diseases:
                logger.info(f"Generating default features for {len(missing_desc_diseases)} diseases with missing descriptions")

                # Create a default feature vector of dimension 768, using the mean of existing features
                default_feature = np.mean(features, axis=0)

                # Assign default features to diseases with missing descriptions
                for disease_id in missing_desc_diseases:
                    disease_features[disease_id] = default_feature + np.random.normal(0, 0.01, 768)  # Add a small amount of noise to make each feature unique

            return disease_features
            
        else:
            # If all diseases have no descriptions, create a zero vector of dimension 768
            logger.warning("All diseases have no description data, using zero vector as default features")
            disease_features = {}
            default_feature = np.zeros(768, dtype=np.float32)
            
            for disease_id in disease_ids:
                disease_features[disease_id] = default_feature + np.random.normal(0, 0.01, 768)
                
            return disease_features
