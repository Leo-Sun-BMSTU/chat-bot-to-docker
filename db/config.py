import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TOKEN")
admin_id = int(os.getenv("ADMIN_ID"))
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
host = "localhost"
port = 1111

I18N_DOMAIN = 'testbot'
BASE_DIR = Path(__file__).parent
LOCALES_DIR = BASE_DIR / 'locales'
