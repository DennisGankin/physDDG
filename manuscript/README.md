# Manuscript

- `main.tex` + `references.bib` — the paper, in the official OUP authoring template
  (`oup-authoring-template.cls` and `oup-abbrvnat.bst` are the CTAN files, included here so the source
  compiles standalone).
- `supplementary.md` — the supplement (source of truth).
- `supplementary.tex` — generated from the markdown by `python3 tools/md2tex.py`; do not edit by hand.
- `figures/make_figures.py` — rebuilds every figure from `results/ladder_physics/*.csv` and
  `preds_*.parquet`: `../.venv/bin/python figures/make_figures.py`.
- `CLARITY_CHECK.md` — the clarity-pass change list and the check of every number against the model card.

## Compiling

No TeX installation is needed. The PDFs here were built with a static Tectonic binary:

```bash
curl -sL -o tectonic.tar.gz \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz
mkdir -p tools && tar xzf tectonic.tar.gz -C tools
tools/tectonic -X compile main.tex
tools/tectonic -X compile supplementary.tex
```

Tectonic downloads the packages it needs on the first run and resolves the bibliography itself, so no separate
BibTeX pass is required. With a normal TeX Live installation, `pdflatex main; bibtex main; pdflatex main;
pdflatex main` produces the same document. The binary itself and the downloaded OUP template archive are not
committed (see `.gitignore`).
