# `src/acquisition.py`

## What this module is for

Downloads the NASA C-MAPSS benchmark, unpacks it, parses it into typed frames, and records
enough provenance that any number downstream traces back to specific bytes. It is the first
thing that runs and the only thing that touches the network.

Nothing calls it except `run_analysis.py` and the capstone notebook's loader. It calls
`requests`, `zipfile` and `pandas`, and nothing else in `src/`.

## The public surface

| Function | In | Out |
|---|---|---|
| `main(force=False)` | — | `DataFrame` of training trajectories; writes 3 parquets + `provenance.json` |
| `download_archive(force)` | — | `(Path, str)` — the archive and the URL that served it |
| `extract_archive(archive)` | `Path` | `dict[filename, {sha256, bytes}]` |
| `read_trajectory_file(path, subset)` | `Path, str` | `DataFrame[unit, cycle, setting_1..3, sensor_1..21, subset]` |
| `read_rul_file(path, subset)` | `Path, str` | `DataFrame[subset, unit, true_rul]` |

Module constants worth knowing: `TRAIN_PARQUET` (where the parsed data lands),
`EXTRACT_DIR`, `SUBSETS` (the four designs), `EXPECTED_TRAIN_UNITS` / `EXPECTED_TEST_UNITS`.

## The walkthrough

**`download_archive` tries mirrors in order and records which one won.** NASA has moved this
file more than once — the original `ti.arc.nasa.gov` host was retired and `data.nasa.gov` has
reorganised since. Pinning one URL guarantees a future breakage; trying a list and *recording
the winner* means that if two runs ever produce different checksums, the provenance file
identifies which mirror changed instead of leaving it a mystery.

If every mirror fails, the error message tells the user where to download manually and where
to put the file. An acquisition script that dies with a bare `ConnectionError` has wasted the
one chance it had to be helpful.

**`extract_archive` follows one level of zip nesting.** The mirrors package the data
differently: `data.nasa.gov` serves `CMAPSSData.zip` directly, the PHM S3 bucket serves an
outer zip with that archive inside it. Four lines of recursion removes an entire class of
"worked on my machine".

**`read_trajectory_file` slices to 26 columns rather than trusting the parse.** Then it runs
two structural checks and raises on either.

**`_assemble` asserts the published unit counts** — 100/260/100/249 for train. This is the
check that matters most, and it is discussed below.

**`main` writes `provenance.json`** with the citation, licence, the served URL, retrieval
timestamp, SHA-256 of the archive and of all 12 extracted files, plus per-subset row/unit
counts and the data caveat.

## The non-obvious bits

**Trailing whitespace.** Every line of every C-MAPSS data file ends with a space. Read
naively, pandas produces **two phantom all-NaN columns** at the end. The module reads with
`sep=r"\s+"`, then explicitly slices to `len(DATA_COLUMNS)` and assigns names. Accepting 28
columns and indexing positionally would work today and break silently the first time the
upstream format changed.

**`if not zipfile.is_zipfile(...)` after a successful download.** This looks paranoid. It is
not: a mirror that has been reorganised commonly returns an HTML error page with HTTP 200.
Without this check the failure surfaces much later, as an unreadable data file, with no
indication that the download was the problem.

**The unit-count assertions.** These are the most valuable four lines in the file. Third-party
mirrors of public datasets get repackaged, re-exported through Excel, and subtly truncated all
the time. A run that quietly analysed 247 engines instead of 249 would produce plausible
numbers that are simply wrong. Failing loudly is the only acceptable behaviour.

**The test files are downloaded but never modelled.** They are truncated before failure with
the answers in a separate key — the competition's held-out set. Since they contain no
observable failure event, a survival model cannot learn from them. They are checksummed
anyway, because a provenance record that describes only the files you liked is not a
provenance record.

**`_recorded_source_url()` exists for the cached path.** When the archive is already on disk
the function still has to return *some* URL, and reading it back from the previous run's
provenance is more honest than returning the first candidate as though it had served.

## How it can break

- **Every mirror goes down.** Raises with manual-download instructions. Not silent.
- **A mirror repackages the data.** Caught by the extraction check (missing expected
  filenames) or the unit-count assertion.
- **Upstream adds a column.** The slice takes the first 26 and the extra is ignored — the one
  failure mode that would *not* raise. Acceptable, because C-MAPSS is a frozen 2008 dataset.
- **Partial download.** Caught by `is_zipfile`, or by the cycle-continuity check.

Pinned by `tests/test_acquisition.py` (6 tests): trailing whitespace, cycle gaps, units not
starting at 1, too few columns, the RUL key's 1-based indexing, and the published counts.
The network path is deliberately **not** tested — a test that needs NASA to be up is a test
that fails for reasons unrelated to the code.
