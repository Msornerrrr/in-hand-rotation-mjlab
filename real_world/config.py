import functools
from pathlib import Path


@functools.lru_cache()
def get_package_root_dir() -> Path:

    return Path(__file__).parent.absolute()

@functools.lru_cache()
def get_repo_root_dir() -> Path:
    return get_package_root_dir().parent


@functools.lru_cache()
def get_data_dir() -> Path:
    return get_repo_root_dir() / "data"


CTRL_HZ = 20
LEAP_CTRL_HZ = 200  # after upsampling
