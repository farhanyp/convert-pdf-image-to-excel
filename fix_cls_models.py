#!/usr/bin/env python3
"""
Script to fix the structure of the cls model directory.
This script copies the model files from subdirectory to the main directory.
"""

import os
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cls_model_fixer")

def fix_cls_model_directory():
    """Fix the structure of the cls model directory."""
    # Path to the cls model directory
    cls_dir = os.path.join('app', 'models', 'cls')
    
    if not os.path.exists(cls_dir):
        logger.error(f"Cls model directory not found: {cls_dir}")
        return False
    
    # Find subdirectory with model files
    subdirs = [d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))]
    
    if not subdirs:
        logger.error(f"No subdirectories found in {cls_dir}")
        return False
    
    # Assume the first subdirectory is the model directory
    model_subdir = os.path.join(cls_dir, subdirs[0])
    logger.info(f"Found model subdir: {model_subdir}")
    
    # Check if model files exist in the subdirectory
    model_files = [
        "inference.pdiparams",
        "inference.pdiparams.info",
        "inference.pdmodel"
    ]
    
    subdir_files = os.listdir(model_subdir)
    logger.info(f"Files in subdir: {subdir_files}")
    
    missing_files = [f for f in model_files if f not in subdir_files]
    if missing_files:
        logger.error(f"Missing model files in {model_subdir}: {missing_files}")
        return False
    
    # Copy model files to the main directory
    logger.info(f"Copying model files from {model_subdir} to {cls_dir}")
    
    for file in model_files:
        source = os.path.join(model_subdir, file)
        dest = os.path.join(cls_dir, file)
        
        if os.path.exists(source):
            shutil.copy2(source, dest)
            logger.info(f"Copied {file} to {dest}")
        else:
            logger.warning(f"Source file not found: {source}")
    
    # Verify files were copied
    cls_files = os.listdir(cls_dir)
    logger.info(f"Files in cls directory after copy: {cls_files}")
    
    # Check for required files
    still_missing = [f for f in model_files if f not in cls_files]
    if still_missing:
        logger.error(f"Still missing model files in {cls_dir}: {still_missing}")
        return False
    
    logger.info("Successfully fixed cls model directory structure")
    return True

if __name__ == "__main__":
    fix_cls_model_directory()