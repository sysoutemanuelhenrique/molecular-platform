import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PUBCHEM_SDF_URL_TEMPLATE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=2d"
)

DEFAULT_TIMEOUT_SECONDS = 30.0


class PubChemNotFoundError(Exception):
    """Raised when PubChem has no record for the given CID (HTTP 404)."""


class PubChemAcquisitionError(Exception):
    """Raised for any other PubChem acquisition failure: timeout, network
    failure, non-404 HTTP error, or an empty response body."""


def build_pubchem_sdf_url(cid: int) -> str:
    """Build the exact PubChem PUG REST endpoint for a compound's 2D SDF."""

    return PUBCHEM_SDF_URL_TEMPLATE.format(cid=cid)


def fetch_pubchem_sdf_2d(cid: int, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Path:
    """Download the 2D SDF for a PubChem CID into a new temporary file.

    Returns the path to the temporary file; the caller owns its cleanup.
    This function only acquires bytes - it never writes into experiment
    storage and never invokes any molecular preparation.
    """

    url = build_pubchem_sdf_url(cid)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()

    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise PubChemNotFoundError(f"PubChem has no record for CID {cid}.") from error
        raise PubChemAcquisitionError(
            f"PubChem request failed with HTTP {error.code} for CID {cid}."
        ) from error

    except TimeoutError as error:
        raise PubChemAcquisitionError(
            f"PubChem request timed out after {timeout}s for CID {cid}."
        ) from error

    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise PubChemAcquisitionError(
                f"PubChem request timed out after {timeout}s for CID {cid}."
            ) from error
        raise PubChemAcquisitionError(
            f"Failed to reach PubChem for CID {cid}: {error.reason}"
        ) from error

    if not data:
        raise PubChemAcquisitionError(f"PubChem returned an empty response for CID {cid}.")

    temp_dir = tempfile.mkdtemp(prefix=f"pubchem_{cid}_")
    temp_path = Path(temp_dir) / f"pubchem_cid_{cid}_2d.sdf"

    try:
        temp_path.write_bytes(data)
    except OSError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return temp_path


def cleanup_temp_artifact(temp_path: Path) -> None:
    """Remove the temporary directory created by fetch_pubchem_sdf_2d.

    The RAW bytes live in a dedicated temp directory (not a bare temp file)
    so that the file's own basename can be deterministic; cleanup therefore
    removes that whole directory, not just the file.
    """

    shutil.rmtree(temp_path.parent, ignore_errors=True)
