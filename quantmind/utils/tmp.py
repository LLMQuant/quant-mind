"""Template system for QuantMind prompts.

This module provides infrastructure to load and render prompt templates
from YAML files using Jinja2 templating engine.
"""

import inspect
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FunctionLoader, StrictUndefined

from quantmind.utils.logger import get_logger

logger = get_logger(__name__)

# Get the current file's directory
DIRNAME = Path(__file__).absolute().resolve().parent
# Get the project root (quant-mind directory)
PROJ_PATH = DIRNAME.parent.parent.parent


def get_caller_dir(upshift: int = 0) -> Path:
    """Get the directory of the calling module.

    Args:
        upshift: Number of stack frames to go up (default: 0 for immediate caller)

    Returns:
        Path to the caller's directory
    """
    stack = inspect.stack()
    caller_frame = stack[1 + upshift]
    caller_module = inspect.getmodule(caller_frame[0])
    if caller_module and caller_module.__file__:
        caller_dir = Path(caller_module.__file__).parent
    else:
        caller_dir = DIRNAME
    return caller_dir


def load_content(
    uri: str, caller_dir: Path | None = None, ftype: str = "yaml"
) -> Any:
    """Load content from file based on URI.

    Args:
        uri: URI in format "path:key1.key2" or "path" for direct file content
        caller_dir: Directory to resolve relative paths from
        ftype: File type extension (yaml, txt, etc.)

    Returns:
        Loaded content (dict for YAML, str for text files)

    Raises:
        FileNotFoundError: If file cannot be found
    """
    if caller_dir is None:
        caller_dir = get_caller_dir(upshift=1)

    # Parse the URI
    path_part, *yaml_trace = uri.split(":")
    assert (
        len(yaml_trace) <= 1
    ), f"Invalid uri {uri}, only one yaml trace is allowed."
    yaml_trace = [key for yt in yaml_trace for key in yt.split(".")]

    # Build file paths with priorities
    if path_part.startswith("."):
        # Relative to caller directory
        file_path_l = [
            caller_dir / f"{path_part[1:].replace('.', '/')}.{ftype}"
        ]
    else:
        # Try current directory first, then project root
        file_path_l = [
            Path(path_part.replace(".", "/")).with_suffix(f".{ftype}"),
            (PROJ_PATH / path_part.replace(".", "/")).with_suffix(f".{ftype}"),
        ]

    for file_path in file_path_l:
        try:
            if ftype == "yaml":
                with file_path.open() as file:
                    yaml_content = yaml.safe_load(file)
                # Traverse the YAML content to get the desired template
                for key in yaml_trace:
                    yaml_content = yaml_content[key]
                return yaml_content
            else:
                return file_path.read_text()
        except FileNotFoundError:
            continue  # File doesn't exist, try next path
        except KeyError:
            continue  # File exists but key is missing
    else:
        raise FileNotFoundError(f"Cannot find {uri} in {file_path_l}")


class QuantMindTemplate:
    """QuantMind Template system for loading and rendering prompt templates.

    Usage:
        T = QuantMindTemplate

        # Load from YAML file
        prompt = T("prompts.analysis:system_prompt").r(
            context="financial analysis",
            model="gpt-4"
        )

        # Load from text file
        prompt = T("prompts.analysis:user_prompt", ftype="txt").r(
            query="analyze this data"
        )
    """

    def __init__(self, uri: str, ftype: str = "yaml"):
        """Initialize template with URI.

        Args:
            uri: URI in format "path:key1.key2" or "path" for direct content
            ftype: File type (yaml, txt, etc.)

        URI Examples:
            - "prompts.analysis:system_prompt" - Load from prompts/analysis.yaml,
                key "system_prompt"
            - ".local:custom_prompt" - Load from caller's local.yaml, key "custom_prompt"
            - "prompts.analysis:user_prompt" - Load from prompts/analysis.yaml, key "user_prompt"
        """
        self.uri = uri
        caller_dir = get_caller_dir(1)

        # Handle relative paths
        if uri.startswith("."):
            try:
                # Convert to project-relative path for easier finding
                self.uri = f"{str(caller_dir.resolve().relative_to(PROJ_PATH)).replace('/', '.')}{uri}"
            except ValueError:
                pass

        self.template = load_content(uri, caller_dir=caller_dir, ftype=ftype)

    def r(self, **context: Any) -> str:
        """Render the template with the given context.

        Args:
            **context: Context variables for template rendering

        Returns:
            Rendered template string
        """
        # Create Jinja2 environment with support for includes
        env = Environment(
            undefined=StrictUndefined, loader=FunctionLoader(load_content)
        )

        # Render the template
        rendered = env.from_string(self.template).render(**context).strip("\n")

        # Clean up multiple newlines
        while "\n\n\n" in rendered:
            rendered = rendered.replace("\n\n\n", "\n\n")

        # Log for debugging
        logger.debug(
            "Template rendered",
            extra={
                "uri": self.uri,
                "template": self.template,
                "context": context,
                "rendered": rendered,
            },
        )

        return rendered


# Convenience alias
T = QuantMindTemplate
