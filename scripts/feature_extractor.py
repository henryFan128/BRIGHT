import os
import esm
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
from tqdm import tqdm
import json
from neo4j import GraphDatabase
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from transformers import AutoTokenizer, AutoModel

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.feature_dr_extra import ChemBERTaExtractor
from utils.feature_pr_extra import ESM2Extractor
from utils.feature_di_extra import BioClinicalBERTExtractor

RDLogger.DisableLog('rdApp.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Neo4jExtractor:
    
    def __init__(self, uri, user, password):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI
            user: Username
            password: Password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close connection"""

        if self.driver:
            self.driver.close()
            
    def extract_drugs(self, limit=None):
        """
        Extract drug information
        
        Args:
            limit: Limit the number of extracted records for testing
            
        Returns:
            Drug dictionary: {drug_id: {name: Name, smiles: SMILES string}}
        """

        with self.driver.session() as session:
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"""
            MATCH (d:Drug)
            RETURN id(d) as id, d.Drug_name as name, d.SMILES as smiles
            {limit_clause}
            """
            result = session.run(query)
            
            drugs = {}
            for record in result:
                drug_id = record["id"]
                drugs[drug_id] = {
                    "name": record["name"],
                    "smiles": record["smiles"]
                }

            logger.info(f"Extracted {len(drugs)} drug entities")
            return drugs
    
    def extract_proteins(self, limit=None):
        """
        Extract protein information

        Args:
            limit: Limit the number of extracted records for testing

        Returns:
            Protein dictionary: {protein_id: {name: Name, sequence: Amino acid sequence}}
        """

        with self.driver.session() as session:
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"""
            MATCH (p:Protein)
            RETURN id(p) as id, p.Protein_name as name, p.Sequence as sequence
            {limit_clause}
            """
            result = session.run(query)
            
            proteins = {}
            for record in result:
                protein_id = record["id"]
                proteins[protein_id] = {
                    "name": record["name"],
                    "sequence": record["sequence"]
                }

            logger.info(f"Extracted {len(proteins)} protein entities")
            return proteins
    
    def extract_diseases(self, limit=None):
        """
        Extract disease information

        Args:
            limit: Limit the number of extracted records for testing

        Returns:
            Disease dictionary: {disease_id: {name: Name, Description: Description}}
        """
        
        with self.driver.session() as session:
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"""
            MATCH (d:Disease)
            RETURN id(d) as id, d.Disease_name as name, 
                   COALESCE(d.Description, d.Disease_name) as description
            {limit_clause}
            """
            result = session.run(query)
            
            diseases = {}
            for record in result:
                disease_id = record["id"]
                description = record["description"] or record["name"]
                diseases[disease_id] = {
                    "name": record["name"],
                    "description": description
                }

            logger.info(f"Extracted {len(diseases)} disease entities")
            return diseases


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Pre-trained model feature extraction and caching")

    # Neo4j connection parameters
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687",
                        help="Neo4j database URI")
    parser.add_argument("--neo4j_user", type=str, default="neo4j",
                        help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default="password",
                        help="Neo4j password")

    # Data limit parameters
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of extracted records for testing")

    # Model path parameters
    parser.add_argument("--chembert_path", type=str, 
                        default="./pretrained_models/ChemBERTa/",
                        help="ChemBERTa model path")
    parser.add_argument("--esm2_path", type=str, 
                        default="./pretrained_models/ESM2/",
                        help="ESM2 model path")
    parser.add_argument("--clinicalbert_path", type=str, 
                        default="./pretrained_models/Bio_ClinicalBERT",
                        help="Bio_ClinicalBERT model path")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./cache/features",
                        help="Feature cache output directory")

    # Batch processing parameters
    parser.add_argument("--drug_batch_size", type=int, default=32,
                        help="Drug feature extraction batch size")
    parser.add_argument("--protein_batch_size", type=int, default=16,
                        help="Protein feature extraction batch size")
    parser.add_argument("--disease_batch_size", type=int, default=32,
                        help="Disease feature extraction batch size")

    # GPU parameters
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specify the GPU ID to use, such as 0, 1, etc., default is to use the system-assigned GPU")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Connect to Neo4j and extract data
    try:
        logger.info("Connecting to Neo4j database and extracting data...")
        extractor = Neo4jExtractor(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

        # Extract entity data
        drugs = extractor.extract_drugs(limit=args.limit)
        proteins = extractor.extract_proteins(limit=args.limit)
        diseases = extractor.extract_diseases(limit=args.limit)

        # Close connection
        extractor.close()

        # Save raw data
        with open(os.path.join(args.output_dir, "entities.pkl"), "wb") as f:
            pickle.dump({
                "drugs": drugs,
                "proteins": proteins, 
                "diseases": diseases
            }, f)

        logger.info("Entity data has been saved")

        # Feature extraction
        logger.info("Starting feature extraction...")

        # Set CUDA_VISIBLE_DEVICES (if GPU ID is specified)
        if args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            logger.info(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_id}")

        # 1. Extract drug features
        logger.info("Extracting drug features...")
        chembert_extractor = ChemBERTaExtractor(args.chembert_path)
        drug_features = chembert_extractor.extract_features(drugs, batch_size=args.drug_batch_size)

        # Save drug features
        drug_chembert = {drug_id: feature["chembert"] for drug_id, feature in drug_features.items()}
        drug_morgan = {drug_id: feature["morgan"] for drug_id, feature in drug_features.items()}
        
        np.savez_compressed(
            os.path.join(args.output_dir, "drug_features.npz"),
            chembert=drug_chembert,
            morgan=drug_morgan
        )
        logger.info(f"Drug features have been saved, total {len(drug_features)} items")

        # 2. Extract protein features
        logger.info("Extracting protein features...")
        esm2_extractor = ESM2Extractor(args.esm2_path)
        protein_features = esm2_extractor.extract_features(proteins, batch_size=args.protein_batch_size)

        # Save protein features
        np.savez_compressed(
            os.path.join(args.output_dir, "protein_features.npz"),
            esm2=protein_features
        )
        logger.info(f"Protein features have been saved, total {len(protein_features)} items")

        # 3. Extract disease features
        logger.info("Extracting disease features...")
        biobert_extractor = BioClinicalBERTExtractor(args.biobert_path)
        disease_features = biobert_extractor.extract_features(diseases, batch_size=args.disease_batch_size)

        # Save disease features
        np.savez_compressed(
            os.path.join(args.output_dir, "disease_features.npz"),
            biobert=disease_features
        )
        logger.info(f"Disease features have been saved, total {len(disease_features)} items")

        # Save feature statistics
        feature_stats = {
            "drug_count": len(drug_features),
            "protein_count": len(protein_features),
            "disease_count": len(disease_features),
            "chembert_dim": 768,
            "morgan_dim": 2048,
            "esm2_dim": 1280,
            "clinicalbert_dim": 768,
            "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(args.output_dir, "feature_stats.json"), "w") as f:
            json.dump(feature_stats, f, indent=2)

        logger.info("All feature extraction completed!")

    except Exception as e:
        logger.error(f"Error occurred during feature extraction: {e}")
        raise
        

if __name__ == "__main__":
    main()
