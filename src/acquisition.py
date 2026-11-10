"""Download the NASA C-MAPSS turbofan degradation benchmark.

Acquisition is code, not a manual download: anyone cloning this repo can
reproduce the exact input by running this module. It records provenance (which
URL actually served, retrieval timestamp, SHA-256 of the archive and of every
extracted file, row and unit counts) so any number in the analysis traces back
to specific bytes.

What this data is
-----------------
NASA's Prognostics Center of Excellence ran the C-MAPSS engine simulator
(Commercial Modular Aero-Propulsion System Simulation) to failure on fleets of
simulated turbofan engines. Each engine starts with a different, unstated
amount of initial wear, runs under a control schedule, and degrades until it
no longer meets its operating specification. Every row is one engine on one
flight cycle: 3 operational settings and 21 sensor channels.

Four subsets vary two things — how many operating conditions the fleet flies
under, and how many independent fault modes can develop:

    FD001   100 train units   1 condition    1 fault  (HPC degradation)
    FD002   260 train units   6 conditions   1 fault  (HPC degradation)
    FD003   100 train units   1 condition    2 faults (HPC + fan)
    FD004   249 train units   6 conditions   2 faults (HPC + fan)

They are kept as four labelled subsets rather than merged blindly, because the
regime is a covariate the analysis needs: a six-condition fleet has sensor
readings that jump between operating points, and pooling it with a
single-condition fleet without a regime indicator would attribute that jump to
degradation.

Why this dataset, and what it does and does not stand for
--------------------------------------------------------
This project is about *when to pull a unit from service*, an industrial
maintenance-timing question. The honest position on the data:

* C-MAPSS is **simulated**, not measured. The degradation dynamics come from a
  physics model with injected noise, not from a real fleet. Nothing here should
  be read as a claim about real turbofans, and the README says so.
* It is used because it has the one property this analysis needs and almost no
  public maintenance dataset has: **complete run-to-failure trajectories for
  every unit**. That is what makes it possible to censor the data on purpose
  and then check the censoring-aware estimate against a known truth. A real
  fleet snapshot would not permit that check, which is the whole methodological
  spine of the project (see ``panel.py``).
* It is a **stated analogue**, not a substitute, for tooling-consumption and
  replenishment-timing work on real industrial equipment. It is not
  proprietary data from any employer and does not stand in for any.

Train vs test, and why the analysis uses the training trajectories
------------------------------------------------------------------
The benchmark ships each subset twice. ``train_FDxxx.txt`` runs every unit to
failure. ``test_FDxxx.txt`` truncates each unit some time before failure, with
the true remaining cycles supplied separately in ``RUL_FDxxx.txt`` — the
competition's held-out answer key.

The published leaderboard framing is "predict the RUL number in the answer
key". This project does not use that framing (``panel.py`` argues why at
length), so the training trajectories are what it models: they are the only
files with an observable failure event, which is what a survival model needs.
The test files and RUL keys are still downloaded and checksummed — they are
used once, as an external check that the induced-censoring construction
produces a fleet whose truncation pattern resembles the benchmark's own.

Source
------
Saxena, A. and Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data
Set*. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett
Field, CA. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Licence: work of the U.S. Government, public domain. Redistribution is
permitted, but the data is not committed to this repository — downloading at
runtime is what guarantees the provenance record describes the file actually
analysed.

Usage
-----
    python src/acquisition.py
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "CMAPSSData"
ARCHIVE_PATH = RAW_DIR / "CMAPSSData.zip"
PROVENANCE_PATH = RAW_DIR / "provenance.json"
TRAIN_PARQUET = RAW_DIR / "cmapss_train.parquet"
TEST_PARQUET = RAW_DIR / "cmapss_test.parquet"
RUL_PARQUET = RAW_DIR / "cmapss_test_rul.parquet"

SOURCE_CITATION = (
    "Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation Simulation "
    "Data Set. NASA Prognostics Data Repository, NASA Ames Research Center, "
    "Moffett Field, CA."
)
SOURCE_PAGE = (
    "https://www.nasa.gov/intelligent-systems-division/"
    "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
)
SOURCE_LICENCE = "U.S. Government work — public domain"

# NASA has moved this file more than once: the original ti.arc.nasa.gov host was
# retired, and data.nasa.gov has reorganised since. Rather than pin one URL and
# break, the module tries known-good mirrors in order and records which one
# actually served. All three are expected to deliver byte-identical archives;
# the checksum in the provenance record is what proves it for a given run.
DOWNLOAD_URLS = [
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]

# The four subsets, with the design NASA published for each. Carried in code
# rather than looked up later because the regime is a modelling covariate: the
# number of operating conditions decides whether sensor jumps are degradation
# or just a change of operating point.
SUBSETS = {
    "FD001": {"conditions": 1, "faults": 1, "fault_modes": "HPC degradation"},
    "FD002": {"conditions": 6, "faults": 1, "fault_modes": "HPC degradation"},
    "FD003": {"conditions": 1, "faults": 2, "fault_modes": "HPC degradation, fan degradation"},
    "FD004": {"conditions": 6, "faults": 2, "fault_modes": "HPC degradation, fan degradation"},
}

# Unit counts published with the benchmark. Asserted after parsing, so a
# truncated download or a repackaged mirror that quietly dropped units fails
# loudly instead of shrinking the fleet by a few engines.
EXPECTED_TRAIN_UNITS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
EXPECTED_TEST_UNITS = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
DATA_COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]

CHUNK_BYTES = 1 << 16
TIMEOUT = 300

USER_AGENT = (
    "turbofan-maintenance-timing/1.0 (portfolio data analysis project; "
    "contact via GitHub Cena-S)"
)


def make_session() -> requests.Session:
    """A session that backs off rather than hammering a mirror that is failing."""
    retry = Retry(
        total=4,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _sha256(path: Path) -> str:
    """Checksum a file without reading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(force: bool = False) -> tuple[Path, str]:
    """Fetch the archive from the first mirror that serves it.

    Returns the path and the URL that worked. Recording *which* mirror served
    matters: if two runs produce different checksums, the provenance record is
    what identifies the mirror that changed rather than leaving it a mystery.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists() and not force:
        recorded = _recorded_source_url()
        print(f"Archive already present: {ARCHIVE_PATH.name}")
        return ARCHIVE_PATH, recorded or "cached (source URL not recorded)"

    session = make_session()
    failures: list[str] = []

    for url in DOWNLOAD_URLS:
        try:
            print(f"Trying {url}")
            response = session.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            with ARCHIVE_PATH.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    handle.write(chunk)
            # A mirror that returns an HTML error page with a 200 would
            # otherwise be caught much later, as an unreadable data file.
            if not zipfile.is_zipfile(ARCHIVE_PATH):
                ARCHIVE_PATH.unlink(missing_ok=True)
                raise RuntimeError("served content is not a zip archive")
            print(f"  served {ARCHIVE_PATH.stat().st_size / 1e6:.1f} MB")
            return ARCHIVE_PATH, url
        except Exception as error:  # noqa: BLE001 - the point is to try the next mirror
            failures.append(f"{url}: {error}")
            print(f"  failed: {error}")

    raise RuntimeError(
        "Every known mirror failed. Tried:\n  " + "\n  ".join(failures)
        + "\n\nFallback: download the C-MAPSS archive manually from\n  "
        + SOURCE_PAGE
        + f"\nand place it at {ARCHIVE_PATH}, then re-run this module."
    )


def _recorded_source_url() -> str | None:
    """The URL a previous run recorded, if there is one."""
    if not PROVENANCE_PATH.exists():
        return None
    try:
        return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["source_url"]
    except (json.JSONDecodeError, KeyError):
        return None


def extract_archive(archive: Path) -> dict[str, dict]:
    """Unpack the data files, following one level of nesting if present.

    The mirrors differ in packaging: data.nasa.gov serves ``CMAPSSData.zip``
    directly, while the PHM S3 bucket serves an outer zip with that archive
    inside it. Handling the nested case is four lines and removes an entire
    class of "worked on my machine" failure.

    Only the ``*_FDxxx.txt`` data files are extracted; directory structure in
    the archive is flattened so downstream code has one place to look.
    """
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, dict] = {}

    def _extract_from(source: zipfile.ZipFile, depth: int = 0) -> None:
        for info in source.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name.lower().endswith(".zip") and depth == 0:
                with source.open(info) as nested_handle:
                    payload = io.BytesIO(nested_handle.read())
                with zipfile.ZipFile(payload) as nested:
                    _extract_from(nested, depth + 1)
                continue
            if not name.endswith(".txt"):
                continue
            if not any(name.startswith(prefix) for prefix in ("train_", "test_", "RUL_")):
                continue
            target = EXTRACT_DIR / name
            with source.open(info) as handle, target.open("wb") as out:
                out.write(handle.read())
            extracted[name] = {
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
            }

    with zipfile.ZipFile(archive) as archive_zip:
        _extract_from(archive_zip)

    missing = [
        f"{kind}_{subset}.txt"
        for subset in SUBSETS
        for kind in ("train", "test", "RUL")
    ]
    absent = [name for name in missing if name not in extracted]
    if absent:
        raise RuntimeError(
            f"Archive is missing expected data files: {absent}. "
            "The mirror may have repackaged the dataset."
        )

    print(f"Extracted {len(extracted)} data files to {EXTRACT_DIR.name}/")
    return extracted


def read_trajectory_file(path: Path, subset: str) -> pd.DataFrame:
    """Read one ``train_`` or ``test_`` file into the canonical 26-column frame.

    The files are space-delimited with *trailing* whitespace on every line, so
    a naive read produces two phantom all-NaN columns at the end. Using
    ``sep=r"\\s+"`` with explicit names and dropping the surplus is more
    honest than silently accepting 28 columns and indexing positionally.
    """
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if frame.shape[1] < len(DATA_COLUMNS):
        raise RuntimeError(
            f"{path.name}: expected at least {len(DATA_COLUMNS)} columns, "
            f"found {frame.shape[1]}."
        )

    frame = frame.iloc[:, : len(DATA_COLUMNS)]
    frame.columns = DATA_COLUMNS
    frame["unit"] = frame["unit"].astype("int32")
    frame["cycle"] = frame["cycle"].astype("int32")
    frame["subset"] = subset

    # Cycles must be a complete 1..n run per unit. A gap would silently change
    # every duration in the analysis, so it is checked rather than assumed.
    per_unit = frame.groupby("unit")["cycle"]
    if not (per_unit.min() == 1).all():
        raise RuntimeError(f"{path.name}: some units do not start at cycle 1.")
    if not (per_unit.max() == per_unit.count()).all():
        raise RuntimeError(f"{path.name}: some units have gaps in their cycle index.")

    return frame


def read_rul_file(path: Path, subset: str) -> pd.DataFrame:
    """Read the answer key: true remaining cycles for each truncated test unit."""
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    return pd.DataFrame({
        "subset": subset,
        "unit": range(1, len(values) + 1),
        "true_rul": values.astype("int32").to_numpy(),
    })


def _assemble(kind: str, expected_units: dict[str, int]) -> tuple[pd.DataFrame, list[dict]]:
    """Read all four subsets of one kind and check the published unit counts."""
    frames: list[pd.DataFrame] = []
    records: list[dict] = []

    for subset in SUBSETS:
        frame = read_trajectory_file(EXTRACT_DIR / f"{kind}_{subset}.txt", subset)
        units = frame["unit"].nunique()

        if units != expected_units[subset]:
            raise RuntimeError(
                f"{kind}_{subset}.txt: expected {expected_units[subset]} units, "
                f"found {units}. The mirror may not be the published dataset."
            )

        frames.append(frame)
        records.append({
            "subset": subset,
            "kind": kind,
            "units": int(units),
            "rows": int(frame.shape[0]),
            "max_cycles": int(frame["cycle"].max()),
            "median_unit_length": float(frame.groupby("unit")["cycle"].max().median()),
            **SUBSETS[subset],
        })
        print(
            f"  {kind:>5} {subset}  {units:>3} units  {frame.shape[0]:>6,} rows  "
            f"median life {records[-1]['median_unit_length']:.0f} cycles"
        )

    return pd.concat(frames, ignore_index=True), records


def main(force: bool = False) -> pd.DataFrame:
    archive, served_url = download_archive(force=force)
    archive_sha = _sha256(archive)
    extracted = extract_archive(archive)

    print("\nParsing subsets:")
    train, train_records = _assemble("train", EXPECTED_TRAIN_UNITS)
    test, test_records = _assemble("test", EXPECTED_TEST_UNITS)

    ruls = pd.concat(
        [read_rul_file(EXTRACT_DIR / f"RUL_{subset}.txt", subset) for subset in SUBSETS],
        ignore_index=True,
    )
    for subset in SUBSETS:
        n_keys = int((ruls["subset"] == subset).sum())
        if n_keys != EXPECTED_TEST_UNITS[subset]:
            raise RuntimeError(
                f"RUL_{subset}.txt: {n_keys} answers for "
                f"{EXPECTED_TEST_UNITS[subset]} test units."
            )

    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    ruls.to_parquet(RUL_PARQUET, index=False)

    record = {
        "citation": SOURCE_CITATION,
        "source_page": SOURCE_PAGE,
        "source_url": served_url,
        "candidate_urls": DOWNLOAD_URLS,
        "licence": SOURCE_LICENCE,
        "data_caveat": (
            "C-MAPSS is simulator output, not measurements from a real fleet. "
            "Used here because complete run-to-failure trajectories are what "
            "make the censoring check possible; no claim about real turbofans "
            "is made or implied."
        ),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": {
            "filename": archive.name,
            "sha256": archive_sha,
            "bytes": archive.stat().st_size,
        },
        "files": extracted,
        "train": train_records,
        "test": test_records,
        "totals": {
            "train_units": int(train.groupby(["subset", "unit"]).ngroups),
            "train_rows": int(train.shape[0]),
            "test_units": int(test.groupby(["subset", "unit"]).ngroups),
            "test_rows": int(test.shape[0]),
        },
    }
    PROVENANCE_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    totals = record["totals"]
    print(
        f"\n{totals['train_units']} training units "
        f"({totals['train_rows']:,} unit-cycles) -> {TRAIN_PARQUET.name}"
    )
    print(
        f"{totals['test_units']} test units "
        f"({totals['test_rows']:,} unit-cycles) -> {TEST_PARQUET.name}"
    )
    print(f"Provenance -> {PROVENANCE_PATH.name}  (served by {served_url})")
    return train


if __name__ == "__main__":
    main()
