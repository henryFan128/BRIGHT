import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from transformers import AutoTokenizer, AutoModel


RDLogger.DisableLog('rdApp.*')  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChemBERTaExtractor:
    """Extracts ChemBERTa features and Morgan fingerprints for drugs."""
    
    def __init__(self, model_path):
        """
        Initialize the ChemBERTa model and tokenizer from a local path or Hugging Face.
        
        Args:
            model_path: path to the local ChemBERTa model directory
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
                logger.info(f"Successfully loaded ChemBERTa model from local path: {model_path}")
            else:
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ChemBERTa model from local path: {e}")
            try:
                # Only attempt to load from Hugging Face if local loading fails
                logger.warning("Attempting to load model from Hugging Face, this may require downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                logger.info("Successfully loaded ChemBERTa model from Hugging Face")
            except Exception as e2:
                logger.error(f"Failed to load ChemBERTa model from Hugging Face: {e2}")
                raise e2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"ChemBERTa model has been loaded to device: {self.device}")

    def extract_morgan_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """
        Extract Morgan fingerprint features

        Args:
            smiles_list: SMILES string list
            radius: Morgan fingerprint radius
            nBits: Fingerprint dimension

        Returns:
            Fingerprint feature array (n_samples, nBits)
        """
        fingerprints = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fingerprints.append(np.array(fp))
                else:
                    logger.warning(f"Unable to parse SMILES: {smi}")
                    fingerprints.append(np.zeros(nBits))
            except Exception as e:
                logger.warning(f"Failed to compute Morgan fingerprint: {smi}, Error: {e}")
                fingerprints.append(np.zeros(nBits))
        
        return np.array(fingerprints)
    
    def batch_extract_features(self, smiles_list, batch_size=32):
        """
        Batch extract ChemBERTa features

        Args:
            smiles_list: SMILES string list
            batch_size: Batch size

        Returns:
            Feature array (n_samples, 768)
        """
        all_features = []

        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting ChemBERTa features"):
            batch_smiles = smiles_list[i:i+batch_size]
            
            processed_batch = []
            for smi in batch_smiles:
                if smi and smi.strip().upper() not in ["", "UNKNOWN"]:
                    processed_batch.append(smi)
                else:
                    processed_batch.append("C")

            inputs = self.tokenizer(processed_batch, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.pooler_output.cpu().numpy()
            
            all_features.append(batch_features)
        
        return np.vstack(all_features) if all_features else np.array([])
    
    def extract_features(self, drugs_dict, batch_size=32):
        """
        Extract ChemBERTa features and Morgan fingerprints for drugs.
        Args:
            drugs_dict: {drug_id: {name, smiles}}
            batch_size: Batch size
            
        Returns:
            Feature dictionary {drug_id: {chembert: vector, morgan: vector}}
        """

        # Organize SMILES list and ID mapping
        drug_ids = list(drugs_dict.keys())
        smiles_list = []
        valid_drug_ids = []
        missing_smiles_drugs = []
        
        for drug_id in drug_ids:
            smiles = drugs_dict[drug_id].get("smiles")
            if smiles and smiles.strip().upper() not in ["", "UNKNOWN"]:
                smiles_list.append(smiles)
                valid_drug_ids.append(drug_id)
            else:
                missing_smiles_drugs.append(drug_id)
                logger.warning(f"Drug {drug_id} has no SMILES data, using default features")

        # Extract features
        if smiles_list:
            chembert_features = self.batch_extract_features(smiles_list, batch_size)
            morgan_features = self.extract_morgan_fingerprints(smiles_list)
            
            drug_features = {}
            for i, drug_id in enumerate(valid_drug_ids):
                drug_features[drug_id] = {
                    "chembert": chembert_features[i],
                    "morgan": morgan_features[i]
                }

            # Handle drugs with missing SMILES
            if missing_smiles_drugs:
                logger.info(f"Generating default features for {len(missing_smiles_drugs)} drugs with missing SMILES")

                # Use the mean of existing features as default features
                default_chembert = np.mean(chembert_features, axis=0)
                default_morgan = np.mean(morgan_features, axis=0)
                
                for drug_id in missing_smiles_drugs:
                    drug_features[drug_id] = {
                        "chembert": default_chembert + np.random.normal(0, 0.01, 768),  # Add a small amount of noise to make each feature unique
                        "morgan": default_morgan + np.random.normal(0, 0.01, 2048)
                    }
            
            return drug_features
            
        else:
            # If all drugs have no SMILES, create default vectors
            logger.warning("All drugs have no SMILES data, using zero vectors as default features")
            drug_features = {}
            default_chembert = np.zeros(768, dtype=np.float32)
            default_morgan = np.zeros(2048, dtype=np.float32)
            
            for drug_id in drug_ids:
                drug_features[drug_id] = {
                    "chembert": default_chembert + np.random.normal(0, 0.01, 768),
                    "morgan": default_morgan + np.random.normal(0, 0.01, 2048)
                }
                
            return drug_features