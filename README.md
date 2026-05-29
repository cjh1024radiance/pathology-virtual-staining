# Pathology Virtual Staining

This repository contains the code for a pathology-aware virtual staining pipeline used in the paper.

## What is included

- WSI preprocessing and paired patch export
- Training code for the virtual stain network
- Inference, ROI selection, and qualitative evaluation scripts
- Diagnostic-aware loss and visualization utilities

## What is not included

- The private dataset is not released
- Trained checkpoints are not bundled
- External API credentials are not stored in the repository

## Repository layout

- `config/` - runtime and training configuration templates
- `core.py` - slide loading and shared image utilities
- `diagnosis_aware_loss.py` - diagnostic-aware loss demo and mask export
- `model/` - model definitions, datasets, and training components
- `train/` - WSI pair preparation and training entrypoints
- `test/` - inference, batch evaluation, and reporting entrypoints

## Requirements

- Python 3.12+
- PyTorch 2.7
- OpenSlide for `.svs` support

Install with:

```bash
pip install -r requirements.txt
```

## Configuration

The repository ships with safe defaults in `config/config.json`.

Use environment variables for secrets and local overrides:

- `PATHOLOGY_LLM_API_URL`
- `PATHOLOGY_LLM_API_KEY`
- `PATHOLOGY_VLM_MODEL`
- `PATHOLOGY_API_PROMPT_HINT`
- `PATHOLOGY_VIRTUAL_STAIN_MODEL_PATH`
- `PATHOLOGY_VIRTUAL_STAIN_MODEL_DEVICE`

The training config template in `config/training_config.json` uses `data/private_dataset` as a placeholder. Replace it with your local dataset path.

## Data policy

The code expects the private dataset to stay local. Do not commit raw slides, patch exports, metadata files, or generated outputs.

See `DATA_POLICY.md` for the recommended release policy.

## Training

Prepare paired patches and train the model:

```bash
python train/run_virtual_stain_training.py --config config/training_config.json
```

For ablation runs:

```bash
python train/run_virtual_stain_ablations.py --config config/training_config.json
```

## Inference

Run the full closed-loop pathology pipeline on a slide:

```bash
python test/pathology_e2e.py --input <slide_path> --output-dir <output_dir>
```

Batch evaluation and sensitivity analysis are available under `test/`.

## Outputs

Generated artifacts are written under `outputs/` by default and are ignored by Git.

## Citation

If you use this code in a paper or project, please cite the corresponding publication.

## License

This repository is released under the MIT License. See `LICENSE`.
