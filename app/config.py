import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-schedule-api'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'storage/schedules')
MAX_CONTENT_LENGTH = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}

PERMANENT_SESSION_LIFETIME = 1800