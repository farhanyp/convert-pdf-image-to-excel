#!/usr/bin/env python3
"""
Script to download PaddleOCR models and save them to app/models directory.
"""

import os
import sys
import argparse
import logging
import requests
import tarfile
import tempfile
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_downloader")

# Define model URLs for PaddleOCR
MODEL_URLS = {
    "det": {
        "en": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
    },
    "rec": {
        "en": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
        "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"
    },
    "cls": {
        "common": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    },
    "table": {
        "en": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar"
    },
    "layout": {
        "en": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar"
    }
}

def download_file(url, output_path):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL of the file to download
        output_path: Path where the file should be saved
    """
    try:
        logger.info(f"Downloading {url} to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize request with stream=True to download in chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB
        
        # Initialize progress bar
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        # Download and save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # Verify file size
        if total_size != 0 and progress_bar.n != total_size:
            logger.warning(f"Downloaded file size ({progress_bar.n} bytes) doesn't match expected size ({total_size} bytes)")
        
        # Verify the file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully downloaded {url} to {output_path} ({os.path.getsize(output_path)} bytes)")
            return True
        else:
            logger.error(f"Download failed: File {output_path} does not exist or is empty")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def extract_tar_with_python(tar_path, extract_dir):
    """
    Extract a tar file using Python's tarfile module.
    
    Args:
        tar_path: Path to the tar file
        extract_dir: Directory where the tar should be extracted
    """
    try:
        logger.info(f"Extracting {tar_path} to {extract_dir} using Python tarfile")
        
        # Create extract directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract using tarfile module
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_dir)
        
        # Log extracted files
        extracted_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                extracted_files.append(os.path.join(root, file))
        
        logger.info(f"Extracted {len(extracted_files)} files to {extract_dir}")
        
        # Remove the tar file to save space
        os.remove(tar_path)
        logger.info(f"Removed tar file: {tar_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {tar_path} with Python tarfile: {str(e)}")
        return False

def extract_tar(tar_path, extract_dir):
    """
    Extract a tar file using system command or Python as fallback.
    
    Args:
        tar_path: Path to the tar file
        extract_dir: Directory where the tar should be extracted
    """
    try:
        logger.info(f"Extracting {tar_path} to {extract_dir}")
        
        # Create extract directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # First try system tar command
        import subprocess
        result = subprocess.run(['tar', '-xf', tar_path, '-C', extract_dir], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.warning(f"System tar command failed with error: {result.stderr.decode()}")
            logger.info("Falling back to Python tarfile module")
            return extract_tar_with_python(tar_path, extract_dir)
        
        # Remove the tar file to save space
        os.remove(tar_path)
        
        # List contents of extract directory to verify
        files = []
        for root, dirs, filenames in os.walk(extract_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        
        logger.info(f"Extracted {len(files)} files to {extract_dir}")
        logger.info(f"Successfully extracted and removed {tar_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error with system tar extraction: {str(e)}")
        # Try Python extraction as fallback
        return extract_tar_with_python(tar_path, extract_dir)

def download_model_type(model_type, lang, output_dir):
    """
    Download a specific model type.
    
    Args:
        model_type: Type of model (det, rec, cls, table, layout)
        lang: Language of the model (en, ch)
        output_dir: Directory where models should be saved
    """
    try:
        if model_type == "cls":
            # cls model is language independent
            url = MODEL_URLS[model_type]["common"]
        else:
            # Other models are language dependent
            url = MODEL_URLS[model_type][lang]
        
        # Define paths
        model_dir = os.path.join(output_dir, model_type)
        tar_filename = url.split("/")[-1]
        tar_path = os.path.join(output_dir, tar_filename)
        
        logger.info(f"Starting download of {model_type} model from {url}")
        
        # Download and extract
        if download_file(url, tar_path):
            extract_success = extract_tar(tar_path, model_dir)
            
            if extract_success:
                # List and log model directory contents
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logger.info(f"Files in {model_dir}: {files}")
                    
                    # Check subdirectories too
                    for item in files:
                        item_path = os.path.join(model_dir, item)
                        if os.path.isdir(item_path):
                            subdir_files = os.listdir(item_path)
                            logger.info(f"Files in {item_path}: {subdir_files}")
                    
                    # Special handling for rec and table models
                    if model_type in ["rec", "table"] and not any(".pdmodel" in f for f in files):
                        logger.warning(f"No .pdmodel files found directly in {model_dir}. Checking subdirectories...")
                        # Check if extraction created a subdirectory with the model
                        has_model_files = False
                        for item in files:
                            item_path = os.path.join(model_dir, item)
                            if os.path.isdir(item_path) and any(".pdmodel" in f for f in os.listdir(item_path)):
                                has_model_files = True
                                logger.info(f"Found model files in subdirectory: {item_path}")
                                break
                        
                        if not has_model_files:
                            logger.error(f"No model files found for {model_type}. Download might have failed.")
                            return False
                
                return True
            else:
                logger.error(f"Failed to extract {tar_path}")
                return False
        else:
            logger.error(f"Failed to download {url}")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading {model_type} model: {str(e)}")
        return False

def download_all_models(output_dir, lang="en"):
    """
    Download all required models for PaddleOCR.
    
    Args:
        output_dir: Directory where models should be saved
        lang: Language of the models (en, ch)
    """
    logger.info(f"Starting download of all PaddleOCR models to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download each model type
    success = True
    model_types = ["det", "rec", "cls", "table", "layout"]
    
    for model_type in model_types:
        logger.info(f"Processing model type: {model_type}")
        if not download_model_type(model_type, lang, output_dir):
            success = False
            logger.error(f"Failed to download {model_type} model")
        else:
            logger.info(f"Successfully downloaded {model_type} model")
    
    if success:
        logger.info(f"Successfully downloaded all models to {output_dir}")
    else:
        logger.warning("Some models failed to download. Check the logs for details.")
    
    # Verify the model directory structure
    verify_model_directory_structure(output_dir, model_types)
    
    return success

def verify_model_directory_structure(output_dir, model_types):
    """
    Verify that the model directory structure is correct.
    
    Args:
        output_dir: Base directory containing model subdirectories
        model_types: List of model types to verify
    """
    logger.info("Verifying model directory structure")
    
    for model_type in model_types:
        model_dir = os.path.join(output_dir, model_type)
        
        if not os.path.exists(model_dir):
            logger.error(f"{model_type} model directory not found: {model_dir}")
            continue
        
        # Check for model files either directly or in subdirectories
        model_files_found = False
        
        # Check for .pdmodel files directly in model directory
        files = os.listdir(model_dir)
        if any(f.endswith('.pdmodel') for f in files):
            logger.info(f"Found model files directly in {model_dir}")
            model_files_found = True
        else:
            # Check subdirectories for model files
            for item in files:
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    subdir_files = os.listdir(item_path)
                    if any(f.endswith('.pdmodel') for f in subdir_files):
                        logger.info(f"Found model files in subdirectory: {item_path}")
                        model_files_found = True
                        break
        
        if not model_files_found:
            logger.error(f"No model files found for {model_type}")

def main():
    # Use app/models as the default output directory
    default_output_dir = os.path.join('app', 'models')
    
    parser = argparse.ArgumentParser(description="Download PaddleOCR models")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help=f"Directory to save the models (default: {default_output_dir})")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ch"],
                        help="Language of the models (en, ch)")
    parser.add_argument("--model", type=str, choices=["det", "rec", "cls", "table", "layout", "all"],
                        default="all", help="Specific model to download (default: all)")
    
    args = parser.parse_args()
    
    if args.model == "all":
        download_all_models(args.output_dir, args.lang)
    else:
        download_model_type(args.model, args.lang, args.output_dir)

if __name__ == "__main__":
    main()