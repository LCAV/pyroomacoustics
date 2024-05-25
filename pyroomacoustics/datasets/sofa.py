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
        path / f"{name}.sofa": info["url"]
        for name, info in sofa_info.items()
        if info["supported"]
    }
    download_multiple(files, overwrite=overwrite, verbose=verbose)

    return list(files.keys())


class SOFADatabase(dict):
    def __init__(self, download=True):
        super().__init__()

        if download:
            download_sofa_files(path=self.root)

        self._db = {}
        for name, info in get_sofa_db_info().items():
            path = self.root / f"{name}.sofa"
            if path.exists():
                dict.__setitem__(self, name, AttrDict(info))
                self[name]["path"] = path

    def list(self):
        for name, info in self.items():
            print(f"- {name} ({info.type})")
            for channel in info.contains:
                print(f"  - {channel}")

    @property
    def root(self):
        return DEFAULT_SOFA_PATH

    @property
    def db_info_path(self):
        return SOFA_INFO

    def __setitem__(self, key, val):
        # disallow writing elements
        raise RuntimeError(f"{self.__class__} is not writable")
