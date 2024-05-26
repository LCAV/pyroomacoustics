import json
from pathlib import Path

from .utils import AttrDict, download_multiple

_pra_data_folder = Path(__file__).parents[1] / "data"
DEFAULT_SOFA_PATH = _pra_data_folder / "sofa"
SOFA_INFO = _pra_data_folder / "sofa_files.json"


def get_sofa_db_info():
    with open(SOFA_INFO, "r") as f:
        sofa_info = json.load(f)
    return sofa_info


def download_sofa_files(path=None, overwrite=False, verbose=False, no_fail=False):
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
    download_multiple(files, overwrite=overwrite, verbose=verbose, no_fail=no_fail)

    return list(files.keys())


class SOFADatabase(dict):
    def __init__(self, download=True):
        super().__init__()

        if download:
            # specify "no_fail" to avoid errors if internet is not available
            download_sofa_files(path=self.root, no_fail=True)

        self._db = {}
        for name, info in get_sofa_db_info().items():
            path = self.root / f"{name}.sofa"
            if path.exists():
                dict.__setitem__(self, name, AttrDict(info))
                self[name]["path"] = path

        for path in DEFAULT_SOFA_PATH.glob("*.sofa"):
            name = path.stem
            if name not in self:
                dict.__setitem__(
                    self,
                    name,
                    AttrDict(
                        {
                            "path": path,
                            "supported": "???",
                            "type": "unknown",
                            "url": "???",
                            "homepage": "???",
                            "license": "???",
                            "contains": None,
                        }
                    ),
                )

    def list(self):
        for name, info in self.items():
            print(f"- {name} ({info.type})")
            if info.contains is not None:
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
