import os

from dotenv import load_dotenv


_PACKAGE_DIR = os.path.dirname(__file__)
# Load .env from the package parent (prompts/.env) then the package itself.
load_dotenv(os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".env")))
load_dotenv(os.path.join(_PACKAGE_DIR, ".env"))
