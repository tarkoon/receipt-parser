from pathlib import Path

# Load .env file so tests pick up GOOGLE_CLOUD_PROJECT etc.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
