import glob
from pathlib import Path

from ultralytics.utils.checks import check_suffix

from duoYolo.utils import ROOT

def check_file(file, suffix="", hard=True):
    """
    Search file (if necessary), check suffix (if provided), and return path.

    Args:
        file (str): File name or path.
        suffix (str | tuple): Acceptable suffix or tuple of suffixes to validate against the file.
        hard (bool): Whether to raise an error if the file is not found.

    Returns:
        (str): Path to the file.
    """
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileExistsError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """
    Search/download YAML file (if necessary) and return path, checking suffix.

    Args:
        file (str | Path): File name or path.
        suffix (tuple): Tuple of acceptable YAML file suffixes.
        hard (bool): Whether to raise an error if the file is not found or multiple files are found.

    Returns:
        (str): Path to the YAML file.
    """
    return check_file(file, suffix, hard=hard)