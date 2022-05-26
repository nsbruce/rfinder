import warnings
from pathlib import Path

from .defaults import DEFAULT_VARS


def get_env_dir() -> Path:
    """
    Get the directory of the .env file
    """
    return Path(__file__).parent.parent.parent


def get_env_files() -> list[Path]:
    """
    Get the list of environment files
    """
    env_dir = get_env_dir()
    env_files = env_dir.glob("**.env**")
    return list(env_files)


def load_env() -> dict[str, str]:
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


def file_to_dict(p: Path) -> dict[str, str]:
    """
    Convert a env formatted file to a dictionary
    TODO could be replaced with python-dotenv package
    """
    env = {}
    with open(p) as f:
        for line in f:
            if line.startswith("#"):
                continue
            key, val = line.split("=")
            env[key] = val.strip()
    return env


def get_all_defaults() -> dict[str, str]:
    """
    Get the default values for all environment variables
    *Stringification is done to match the output of file_to_dict
    """

    stringified_vals = {k: str(v) for k, v in DEFAULT_VARS.items()}
    return stringified_vals
