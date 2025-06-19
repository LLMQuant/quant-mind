"""Environment configuration utilities with dotenv support."""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class EnvConfig:
    """Environment configuration manager with dotenv support."""

    _loaded = False

    @classmethod
    def load_dotenv(cls, dotenv_path: Optional[str] = None) -> bool:
        """Load environment variables from .env file.

        Args:
            dotenv_path: Path to .env file. If None, searches for .env in current
                        directory and parent directories.

        Returns:
            True if .env file was found and loaded, False otherwise
        """
        if cls._loaded:
            return True

        if not DOTENV_AVAILABLE:
            logger.warning(
                "python-dotenv not available. Install with: pip install python-dotenv"
            )
            return False

        try:
            if dotenv_path:
                # Load specific file
                env_path = Path(dotenv_path)
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    cls._loaded = True
                    return True
                else:
                    logger.warning(f"Dotenv file not found: {env_path}")
                    return False
            else:
                # Auto-discover .env file
                current_dir = Path.cwd()
                env_paths = [
                    current_dir / ".env",
                    current_dir.parent / ".env",  # Check parent directory
                    Path.home()
                    / ".quantmind"
                    / ".env",  # User config directory
                ]

                for env_path in env_paths:
                    if env_path.exists():
                        load_dotenv(env_path)
                        logger.info(f"Loaded environment from {env_path}")
                        cls._loaded = True
                        return True

                logger.debug("No .env file found in standard locations")
                return False

        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")
            return False

    @classmethod
    def get_api_key(cls, service: str, required: bool = True) -> Optional[str]:
        """Get API key for a service with automatic dotenv loading.

        Args:
            service: Service name (e.g., 'LLAMA_CLOUD', 'OPENAI')
            required: Whether the API key is required

        Returns:
            API key string or None if not found

        Raises:
            ValueError: If required=True and API key not found
        """
        # Ensure dotenv is loaded
        cls.load_dotenv()

        # Construct environment variable name
        env_var = f"{service}_API_KEY"
        api_key = os.getenv(env_var)

        if api_key:
            logger.debug(f"Found {env_var} in environment")
            return api_key
        elif required:
            raise ValueError(
                f"{env_var} is required but not found. "
                f"Set it as environment variable or in .env file."
            )
        else:
            logger.debug(f"{env_var} not found (optional)")
            return None

    @classmethod
    def get_env_var(
        cls,
        var_name: str,
        default: Optional[str] = None,
        required: bool = False,
    ) -> Optional[str]:
        """Get environment variable with automatic dotenv loading.

        Args:
            var_name: Environment variable name
            default: Default value if not found
            required: Whether the variable is required

        Returns:
            Environment variable value or default

        Raises:
            ValueError: If required=True and variable not found
        """
        # Ensure dotenv is loaded
        cls.load_dotenv()

        value = os.getenv(var_name, default)

        if value is None and required:
            raise ValueError(
                f"Environment variable {var_name} is required but not found. "
                f"Set it as environment variable or in .env file."
            )

        return value


def create_sample_env_file(path: str = ".env") -> None:
    """Create a sample .env file with common configuration options.

    Args:
        path: Path where to create the .env file
    """
    env_path = Path(path)

    if env_path.exists():
        logger.warning(f".env file already exists at {env_path}")
        return

    sample_content = """# QuantMind Configuration
# Copy this file to .env and fill in your actual values

# API Keys
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# QuantMind Settings
QUANTMIND_LOG_LEVEL=INFO
QUANTMIND_DATA_DIR=./data
QUANTMIND_TEMP_DIR=/tmp

# Workflow Settings
QUANTMIND_MAX_WORKERS=4
QUANTMIND_RETRY_ATTEMPTS=3
QUANTMIND_TIMEOUT=300

# ArXiv Settings
QUANTMIND_ARXIV_MAX_RESULTS=100

# Optional: Override specific service endpoints
# LLAMA_CLOUD_BASE_URL=https://api.llamaindex.ai
# OPENAI_BASE_URL=https://api.openai.com
"""

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        logger.info(f"Created sample .env file at {env_path}")
        print(f"✅ Created sample .env file at {env_path}")
        print("📝 Please edit it with your actual API keys and settings")
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")
        raise


# Convenience functions for common use cases
def get_llama_cloud_api_key(required: bool = True) -> Optional[str]:
    """Get LlamaParse Cloud API key."""
    return EnvConfig.get_api_key("LLAMA_CLOUD", required=required)


def get_openai_api_key(required: bool = True) -> Optional[str]:
    """Get OpenAI API key."""
    return EnvConfig.get_api_key("OPENAI", required=required)


def load_environment() -> bool:
    """Load environment configuration. Call this at startup."""
    return EnvConfig.load_dotenv()
