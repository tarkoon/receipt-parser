import sys
from pathlib import Path

# Add project root to sys.path so tests can import modules directly
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file so tests pick up GOOGLE_CLOUD_PROJECT etc.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
