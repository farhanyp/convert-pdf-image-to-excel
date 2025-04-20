import os
import logging
from typing import List, Dict, Any
from werkzeug.datastructures import FileStorage
from app.services.ocr_service import OCRService

class ScheduleParserService:
    """Service for parsing PDF schedule files."""
    
    def __init__(self, logger=None):
        """Initialize the parser service with a logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.ocr_service = OCRService(logger)
    
    def parse_schedule_pdf(self, file: FileStorage) -> List[Dict[str, Any]]:
        """
        Parse the uploaded PDF file and extract employee schedule data.
        
        Args:
            file: The uploaded PDF file
            
        Returns:
            List of extracted employee data with schedules
        """
        try:
            self.logger.info(f"Processing PDF file: {file.filename}")
            
            # Save the uploaded file temporarily
            temp_path = os.path.join('storage/schedules', 'temp_' + file.filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            
            # Log the temporarily saved file
            self.logger.info(f"Temporarily saved file to: {temp_path}")
            
            # Use OCR service to process PDF and extract employee data
            extracted_employees = self.ocr_service.process_pdf_and_extract_table_data(temp_path)
            
            # Clean up the temporary file
            os.remove(temp_path)
            self.logger.info(f"Removed temporary file: {temp_path}")
            
            self.logger.info(f"Successfully extracted {len(extracted_employees)} employees")
            return extracted_employees
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            raise