# Releasing the code and data

## What can and cannot be shared

The LUCAS topsoil data are licensed to the recipient and may not be redistributed. The
licence grants a personal, non-transferable right to use the data and forbids
sub-licensing, so the raw campaign files stay out of any public release.

What can be shared: the analysis code, the derived panel once it no longer permits
reconstruction of licensed point data, and the result tables. The safest position is to
release code and result tables, and to describe how to rebuild the panel from files
obtained directly from ESDAC. `scripts/lucas_repeat_count.py` and `src/build_panel.py`
do exactly that, so a reader with their own ESDAC access can reproduce the study.

The derived covariate series are computed from Landsat, which carries no such restriction,
but they are keyed to LUCAS point identifiers and so should not be published with those
identifiers attached.

## GitHub

1. Create a repository, for example `soc-sequence-test`.
2. From this directory:

```
git init
git add -A
git commit -m "Test of a convolutional-recurrent model for SOC change"
git branch -M main
git remote add origin git@github.com:<user>/soc-sequence-test.git
git push -u origin main
```

`.gitignore` already excludes `data/raw/`, `data/interim/`, `data/processed/` and trained
models, so no licensed data leaves the machine. Verify before pushing:

```
git status --porcelain | grep -E "data/(raw|interim|processed)" && echo "STOP: licensed data staged"
```

## Zenodo

1. Log in to Zenodo with GitHub and enable the repository under Settings, GitHub.
2. Create a release on GitHub, tagged `v1.0.0`. Zenodo archives it and issues a DOI.
3. `.zenodo.json` supplies the metadata, including the ORCID.
4. Put the resulting DOI in the paper's Data and code availability statement, replacing
   the present wording.

Enabling the repository on Zenodo before the release is what triggers archiving; a release
made first is not captured.
