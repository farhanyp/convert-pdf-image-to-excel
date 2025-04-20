#!/usr/bin/env python3
"""
Script to check PaddleOCR model directories and files.
Run this to verify if models are correctly downloaded and extracted.
"""

import os
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_checker")

def check_model_files(model_dir, model_type):
    """
    Check if required model files exist in the specified directory.
    
    Args:
        model_dir: Model directory to check
        model_type: Type of model (det, rec, cls, table, layout)
        
    Returns:
        Boolean indicating if model files are valid
    """
    if not os.path.exists(model_dir):
        logger.error(f"{model_type.upper()} model directory not found: {model_dir}")
        return False
    
    # Check for any files or subdirectories
    all_items = os.listdir(model_dir)
    if not all_items:
        logger.error(f"No files or subdirectories found in {model_dir}")
        return False
    
    # Check for model files in subdirectories or directly in the model_dir
    model_files_found = False
    
    # First check subdirectories
    subdirs = [d for d in all_items if os.path.isdir(os.path.join(model_dir, d))]
    if subdirs:
        # Check each subdirectory for model files
        for subdir in subdirs:
            subdir_path = os.path.join(model_dir, subdir)
            logger.info(f"Checking model files in subdirectory: {subdir_path}")
            
            found_files = os.listdir(subdir_path)
            logger.info(f"Found files in subdirectory: {found_files}")
            
            # Check for various model file patterns
            if any(f.endswith('.pdmodel') for f in found_files):
                model_files_found = True
                break
    
    # If no model files found in subdirectories, check the main directory
    if not model_files_found:
        logger.info(f"Checking model files directly in: {model_dir}")
        found_files = all_items
        logger.info(f"Found files in main directory: {found_files}")
        
        # Check for various model file patterns
        if any(f.endswith('.pdmodel') for f in found_files):
            model_files_found = True
    
    if not model_files_found:
        logger.error(f"No model files (*.pdmodel) found in {model_dir} or its subdirectories")
        return False
    
    logger.info(f"{model_type.upper()} model appears to be valid with model files")
    return True

def check_all_models(models_dir):
    """
    Check all model types in the specified directory.
    
    Args:
        models_dir: Base directory containing model subdirectories
        
    Returns:
        Dictionary with validation results for each model type
    """
    logger.info(f"Checking PaddleOCR models in {models_dir}")
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return {}
    
    model_types = ["det", "rec", "cls", "table", "layout"]
    validation_results = {}
    
    for model_type in model_types:
        model_dir = os.path.join(models_dir, model_type)
        validation_results[model_type] = check_model_files(model_dir, model_type)
    
    # Print summary
    logger.info("Model validation summary:")
    for model_type, is_valid in validation_results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID"
        logger.info(f"{model_type.upper()}: {status}")
    
    return validation_results

def main():
    # Use app/models as the default models directory
    default_models_dir = os.path.join('app', 'models')
    
    parser = argparse.ArgumentParser(description="Check PaddleOCR model directories and files")
    parser.add_argument("--models_dir", type=str, default=default_models_dir,
                        help=f"Base directory containing model subdirectories (default: {default_models_dir})")
    
    args = parser.parse_args()
    
    check_all_models(args.models_dir)

if __name__ == "__main__":
    main()