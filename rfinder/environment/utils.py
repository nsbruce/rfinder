import warnings
from pathlib import Path
from typing import Dict, List

from .defaults import BLOB_VARS, DEFAULT_VARS


def get_env_dir() -> Path:
    """
    Get the directory of the .env file
    """
    return Path(__file__).parent.parent.parent


def get_env_files() -> List[Path]:
    """
    Get the list of environment files
    """
    env_dir = get_env_dir()
    env_files = env_dir.glob("**.env**")
    return list(env_files)


def load_env() -> Dict[str, str]:
    """
    Load environment variables from .env file
    """
    env_dir = get_env_dir()

    try:
        env = file_to_dict(env_dir / ".env.local")
    except FileNotFoundError:
        warnings.warn(
            "No .env.local file found in {}. Defaulting to"
            " rfind_web.environment.defaults".format(env_dir)
        )
        env = get_all_defaults()

    return env


def file_to_dict(p: Path) -> Dict[str, str]:
    """
    Convert a env formatted file to a dictionary
    TODO could be replaced with python-dotenv package
    """
    env = {}
    with open(p) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            key, val = line.split("=")
            env[key] = val.strip()
    return env


def get_all_defaults() -> Dict[str, str]:
    """
    Get the default values for all environment variables
    *Stringification is done to match the output of file_to_dict
    """

    all_vals = {
        **DEFAULT_VARS,
        **BLOB_VARS,
    }
    stringified_vals = {k: str(v) for k, v in all_vals.items()}
    return stringified_vals
