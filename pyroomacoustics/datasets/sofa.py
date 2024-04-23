import json
from pathlib import Path

from .utils import download_multiple, AttrDict

_pra_data_folder = Path(__file__).parents[1] / "data"
DEFAULT_SOFA_PATH = _pra_data_folder / "sofa"
SOFA_INFO = _pra_data_folder / "sofa_files.json"


def get_sofa_db_info():
    with open(SOFA_INFO, "r") as f:
        sofa_info = json.load(f)
    return sofa_info


def download_sofa_files(path=None, overwrite=False, verbose=False):
    """
    Download the SOFA files containing source/receiver impulse responses

    Parameters
    ----------
    path: str or Path, optional
        A path to a directory where the files will be downloaded
    overwrite: bool, optional
        If set to `True`, forces the download even if the files already exist
    verbose: bool
        Print some information about the download status (default `False`)

    Returns
    -------
    files: list of Path
        The list of path to the files downloaded
    """
    if path is None:
        path = DEFAULT_SOFA_PATH

    path = Path(path)

    path.mkdir(exist_ok=True, parents=True)

    sofa_info = get_sofa_db_info()

    files = {
        path / name: info["url"]
        for name, info in sofa_info.items()
        if info["supported"]
    }
    download_multiple(files, overwrite=overwrite, verbose=verbose)

    return list(files.keys())


def SOFADatabase(AttrDict):
    def __init__(self):
        self._db = get_sofa_db_info()
        super().__init__(self._db)
