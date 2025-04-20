from flask import Blueprint
from app.controllers.schedule_controller import (
    upload_schedule, 
    get_employee_selection,
    generate_excel
)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Schedule routes
api_bp.route('/schedule/upload', methods=['POST'])(upload_schedule)
api_bp.route('/schedule/employees', methods=['GET'])(get_employee_selection)
api_bp.route('/schedule/generate-excel', methods=['POST'])(generate_excel)