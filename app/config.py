"""
App configuration.

DB credentials are provided as five discrete environment variables
(DB_DRIVER, DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD) -- never a single
pre-built connection string, and never hardcoded here.

Entra ID / JWT / Anthropic settings work the same way: all environment
variables. The ones below default to placeholder values so the app can
still start up -- login and chat simply won't work until you replace them
in .env with real values (see the comments in .env).

load_dotenv() is called with no arguments, so it uses its default search
behavior starting from the current working directory. The backend is
expected to be run from the `backend/` folder (same folder as .env and
requirements.txt), e.g. `uvicorn app.main:app` invoked from within
`backend/`.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- Database ---
    DB_DRIVER: str
    DB_SERVER: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # --- Entra ID (Microsoft identity platform), spec §3.1 ---
    # PLACEHOLDER values -- replace with your app registration's real
    # tenant ID and client ID before testing login. No client secret is
    # needed here: the Angular app authenticates as a public client
    # (authorization code + PKCE), and the backend only *validates* the
    # tokens Entra issues, via Entra's published JWKS -- it never talks to
    # Entra to issue tokens itself.
    ENTRA_TENANT_ID: str = "REPLACE_WITH_ENTRA_TENANT_ID"
    ENTRA_CLIENT_ID: str = "REPLACE_WITH_ENTRA_CLIENT_ID"

    # --- App's own JWT, issued after Entra validation (spec §3.1) ---
    # PLACEHOLDER -- replace with a real random secret before testing
    # login; anyone with this value can forge valid session tokens.
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    JWT_SECRET_KEY: str = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 480  # ~8 hours, per spec §3.1

    # --- Anthropic Claude API (chat generation) ---
    ANTHROPIC_API_KEY: str = "REPLACE_WITH_ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"


settings = Settings()
