# `src/__init__.py`

## What this module is for

Marks `src/` as a package so `from src import panel, survival` works from the repo root, the
notebooks directory and the test suite alike. It holds one line: a docstring naming what the
package is.

It is included here because the `src-explained/` set is checked **mechanically** — the
filenames in `src/` and `src-explained/` must match one to one, with no "this one is
obvious" exemptions. That rule is what stops a genuinely important module from being skipped
on the grounds that someone assumed it was trivial.

## The public surface

Nothing. It exports no names and defines no behaviour.

## The walkthrough

```python
"""Turbofan maintenance-timing: censoring-aware survival modelling of engine failure."""
```

That is the file.

## The non-obvious bits

**It is deliberately empty of imports.** A common convenience is to re-export the submodules
here, so `import src` gives you `src.panel` for free. That is avoided on purpose: it would
make importing *any* module drag in all of them, including `figures` (matplotlib) and
`survival` (lifelines and scikit-survival). `src/acquisition.py` is meant to be runnable as a
standalone script on a machine that has only `requests` and `pandas` working, and eager
re-exports would break that.

**It matters for the Kaggle notebook.** `05-communication/tools/make_kaggle_notebook.py`
builds a synthetic `src` module and registers each inlined submodule under `sys.modules`
before exec-ing it. If this file had side effects or imports, that reconstruction would have
to reproduce them. Keeping it inert is what makes the generated notebook simple and reliable.

## How it can break

It cannot, short of deletion — which would break every `from src import ...` in the repo and
fail the entire test suite on collection, immediately and unmistakably.
