import os
import secrets
from flask import request, jsonify, session, current_app
from werkzeug.utils import secure_filename
from app.services.schedule_parser_service import ScheduleParserService
from app.services.excel_generator_service import ExcelGeneratorService

# Initialize services
schedule_parser = ScheduleParserService()
excel_generator = ExcelGeneratorService()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def upload_schedule():
    # Check if the POST request has the file part
    if 'schedule_pdf' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No file part in the request'
        }), 400
    
    file = request.files['schedule_pdf']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No file selected'
        }), 400
    
    # Check if the file is a PDF
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'message': 'Only PDF files are allowed'
        }), 400
    
    try:
        current_app.logger.info('Schedule file received', {
            'original_name': file.filename,
            'size': file.content_length,
            'mime_type': file.mimetype
        })
        
        # Parse the PDF and extract employee data
        current_app.logger.info('Starting OCR processing of PDF file')
        extracted_employees = schedule_parser.parse_schedule_pdf(file)
        
        # Log the extracted data for debugging
        current_app.logger.info('Extracted employees data', {
            'count': len(extracted_employees),
            'sample': extracted_employees[0] if extracted_employees else 'No employees found'
        })
        
        # Store data in session
        session['file_info'] = {
            'name': file.filename,
            'size': file.content_length,
            'uploaded_at': None  # Use datetime.now().strftime('%Y-%m-%d %H:%M:%S') if needed
        }
        session['extracted_employees'] = extracted_employees
        
        return jsonify({
            'success': True,
            'message': 'File successfully processed with OCR',
            'data': {
                'file_info': {
                    'name': file.filename,
                    'size': file.content_length,
                },
                'employees': extracted_employees,
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error('Error processing PDF file with OCR', {
            'error': str(e)
        })
        
        return jsonify({
            'success': False,
            'message': f'Failed to process file: {str(e)}'
        }), 500

def get_employee_selection():
    """
    Get extracted employee data
    
    Returns:
        JSON response with employee data
    """
    if 'extracted_employees' not in session:
        return jsonify({
            'success': False,
            'message': 'No data found. Please upload a schedule file first'
        }), 400
    
    employees = session.get('extracted_employees', [])
    file_info = session.get('file_info', {})
    
    return jsonify({
        'success': True,
        'employees': employees,
        'fileInfo': file_info
    }), 200

def generate_excel():
    """
    Generate Excel files based on selected employees
    
    Returns:
        JSON response with download links
    """
    # Validate request
    if not request.is_json:
        return jsonify({
            'success': False,
            'message': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    if not data or 'selected_employees' not in data:
        return jsonify({
            'success': False,
            'message': 'Selected employees data is required'
        }), 400
    
    selected_employee_ids = data.get('selected_employees', [])
    
    if not isinstance(selected_employee_ids, list) or not selected_employee_ids:
        return jsonify({
            'success': False,
            'message': 'Selected employees must be a non-empty array of IDs'
        }), 400
    
    try:
        if 'extracted_employees' not in session:
            return jsonify({
                'success': False,
                'message': 'No schedule data has been processed. Please upload a file first.'
            }), 400
        
        extracted_employees = session.get('extracted_employees', [])
        
        # Debug: Log employee data
        current_app.logger.info('Selected employee data: ', {
            'ids': selected_employee_ids,
            'count': len(selected_employee_ids)
        })
        
        # Filter selected employees
        selected_employees = [
            employee for employee in extracted_employees 
            if employee.get('id') in selected_employee_ids
        ]
        
        # Debug: Log filtered data
        current_app.logger.info(f'Employee count after filtering: {len(selected_employees)}')
        
        # Create storage directory if it doesn't exist
        storage_path = current_app.config['UPLOAD_FOLDER']
        os.makedirs(storage_path, exist_ok=True)
        
        # Generate unique filenames
        file_identifier = secrets.token_hex(4)
        standard_filename = f'Schedule_Standard_{file_identifier}.xlsx'
        printable_filename = f'Schedule_Print_A5_{file_identifier}.xlsx'
        
        standard_file_path = os.path.join(storage_path, standard_filename)
        printable_file_path = os.path.join(storage_path, printable_filename)
        
        # Debug: Log paths
        current_app.logger.info('Files to be created: ', {
            'standard': standard_file_path,
            'printable': printable_file_path
        })
        
        # Create Standard Excel file
        excel_generator.create_standard_excel(selected_employees, standard_file_path)
        current_app.logger.info(f'Standard Excel file successfully created: {standard_file_path}')
        
        # Create printable Excel file (A5)
        excel_generator.create_printable_excel(selected_employees, printable_file_path)
        current_app.logger.info(f'Printable Excel file successfully created: {printable_file_path}')
        
        # Verify files
        if not os.path.exists(standard_file_path) or not os.path.exists(printable_file_path):
            raise Exception('Excel files were not successfully saved to storage.')
        
        # Check file sizes for debugging
        standard_file_size = os.path.getsize(standard_file_path)
        printable_file_size = os.path.getsize(printable_file_path)
        
        current_app.logger.info('Excel file sizes: ', {
            'standard': excel_generator.format_bytes(standard_file_size),
            'printable': excel_generator.format_bytes(printable_file_size)
        })
        
        # Create URLs for downloading these files
        base_url = request.host_url.rstrip('/') + '/api/download'
        
        return jsonify({
            'success': True,
            'message': 'Excel files successfully created',
            'debug': {
                'employee_count': len(selected_employees),
                'standard_file_size': excel_generator.format_bytes(standard_file_size),
                'printable_file_size': excel_generator.format_bytes(printable_file_size),
            },
            'files': {
                'standard': {
                    'name': standard_filename,
                    'url': f"{base_url}/{standard_filename}",
                },
                'printable': {
                    'name': printable_filename,
                    'url': f"{base_url}/{printable_filename}",
                }
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error('Error creating Excel files', {
            'error': str(e)
        })
        
        return jsonify({
            'success': False,
            'message': f'Failed to create Excel files: {str(e)}'
        }), 500