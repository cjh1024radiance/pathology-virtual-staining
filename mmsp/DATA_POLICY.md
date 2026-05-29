# Data Policy

This repository is prepared for public release, but the underlying pathology dataset remains private.

## Release rules

- Do not commit raw slides, tile exports, or patch pairs
- Do not commit patient identifiers, slide IDs that can be traced back to the source system, or private manifests
- Do not commit checkpoints trained on restricted data unless the project policy explicitly allows it
- Keep all local data under `data/` or another ignored directory

## Recommended local layout

- `data/private_dataset/` - your private WSI dataset
- `outputs/` - preprocessing, training, and inference artifacts

## Manifest guidance

If you share examples, anonymize them and replace file paths with placeholders such as:

- `data/private_dataset/<case_id>/<case_id>_unstained.svs`
- `data/private_dataset/<case_id>/<case_id>_he.svs`

## Review checklist before publishing

- Search for secrets and remove them from tracked files
- Verify `.gitignore` covers dataset and output directories
- Confirm that example manifests do not expose real patient or institution identifiers
