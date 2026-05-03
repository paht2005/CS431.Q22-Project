### Zero-shot Object Counting with Good Exemplars
**Enhanced with Rich Prompts and YOLO-World — CS338.Q21 (Pattern Recognition, UIT)**

This folder contains the **main implementation** of the VA-Count model and the
project-specific extensions:

- Baseline VA-Count (GroundingDINO-based exemplar extraction + MAE + NSM)
- Rich Prompt pipeline (prompt enhancement + CLIP re-ranking)
- YOLO-World based exemplar extraction for faster inference

All instructions below assume the current working directory is
`code/source-code`.

---

##  Table of Contents
- [Project History](#project-history)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Checkpoints](#model-checkpoints)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Project History

This codebase was built from scratch for CS338.Q21 (Pattern Recognition,
UIT) as a re-implementation of the ECCV 2024 **VA-Count** paper, extended
with two independent additions from the literature:

- **Rich Prompts** (Zhu et al., 2025): Gemini-generated visual descriptions
  + CLIP re-ranking to obtain higher-quality positive/negative exemplars.
- **YOLO-World** (Cheng et al., 2024): a fast open-vocabulary detector used
  as a drop-in replacement for GroundingDINO in the exemplar-extraction
  stage.

The team's deliverables include:

- Repository structure (`docs/`, `configs/`, `scripts/`) with the Gemini API
  key kept out of source code (loaded from a local `.env`).
- Training, evaluation, exemplar-generation and Streamlit-demo scripts.
- A 26-page bilingual LaTeX report (`docs/report/Report.pdf`) with full
  methodology, figures and result tables.
- The `wandb` runs archived under `experiments/exp{2,3,4,5}/wandb/`, which
  are the source of every MAE / RMSE / latency number quoted below.

## Overview

VA-Count is a zero-shot object counting method that leverages good exemplars for accurate counting. The model combines:
- Vision Transformer backbone (MAE pretrained)
- Grounding DINO for exemplar detection
- Binary classifier for single/multiple object detection
- Cross-attention mechanism for feature matching

## Project Structure

```
source-code/
├── data/                           # Dataset directory
│   ├── FSC147/
│   │   ├── images_384_VarV2/      # Resized images
│   │   ├── gt_density_map_adaptive_384_VarV2/  # Density maps
│   │   ├── annotation_FSC147_384.json
│   │   ├── annotation_FSC147_pos.json  # Positive exemplars
│   │   ├── annotation_FSC147_neg.json  # Negative exemplars
│   │   ├── Train_Test_Val_FSC_147.json
│   │   ├── ImageClasses_FSC147.txt
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   ├── CARPK/
│   └── out/                        # Output directory
│       ├── classify/               # Classifier checkpoints
│       ├── results_base/           # Test results
│       └── pre_4_dir/              # Pretrain checkpoints
├── GroundingDINO/                  # Grounding DINO submodule (installed here)
│   ├── groundingdino/
│   ├── weights/
│   │   └── groundingdino_swint_ogc.pth
│   └── ...
├── util/                           # Utility functions
│   ├── FSC147.py                  # Dataset loader
│   ├── misc.py                    # Miscellaneous utilities
│   └── ...
├── models_crossvit.py             # Cross-ViT model
├── models_mae_cross.py            # MAE with cross-attention
├── models_mae_noct.py             # MAE without counting token
├── FSC_pretrain.py                # Pretraining script (MAE backbone)
├── FSC_train.py                   # Training / fine-tuning script
├── FSC_test.py                    # Testing script (MAE, NSM, MAE+YOLO etc.)
├── biclassify.py                  # Binary classifier training
├── datasetmake.py                 # Dataset preparation
├── grounding_pos.py               # Generate positive exemplars with GroundingDINO
├── grounding_neg.py               # Generate negative exemplars with GroundingDINO
├── yolo_pos_withPrompt.py         # YOLO-World positive exemplars with prompts
├── yolo_neg.py                    # YOLO-World negative exemplars
├── yolo_pos_withoutPrompt.py      # YOLO-World positive exemplars without prompts
├── demo_app_advanced.py           # Advanced demo application (UI + visualization)
├── demo_inference.py              # Basic command-line inference demo
├── demo_pipeline_advanced.py      # Advanced end-to-end pipeline demo
├── demo_visualization.py          # Standalone visualization demo
├── inference_official.py          # Script close to the official VA-Count pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Environment Setup

### Prerequisites
- Python 3.9+
- CUDA 11.7+ (for GPU support; CPU / Apple Silicon MPS also work)
- PyTorch 2.0+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/paht2005/CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers.git
cd CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers
```

2. **Create conda environment**
```bash
conda create -n vacount python=3.12
conda activate vacount
```

3. **Install Grounding DINO**
```bash
cd GroundingDINO
pip install -e .
cd ..
```

4. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

The default `requirements.txt` targets modern PyTorch (`torch>=2.0`) and works
on macOS (CPU / MPS), Linux CPU, and CUDA >= 11.7. To reproduce the exact
MAE / RMSE / latency numbers reported in `docs/report/Report.pdf` on a
CUDA 11.6 box, install the legacy pin file instead:

```bash
pip install -r requirements-cuda116.txt
```

5. **Download Grounding DINO weights**
```bash
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

## Dataset Preparation

### FSC147 Dataset

1. **Download FSC147**
   - Download from [FSC147](https://www.kaggle.com/datasets/xuncngng/fsc147-0)
   - Extract to `./data/FSC147/`

2. **Prepare data splits**
```bash
# Create train/val/test split files
python -c "
import json
from pathlib import Path

json_file = './data/FSC147/Train_Test_Val_FSC_147.json'
with open(json_file, 'r') as f:
    data = json.load(f)

for split in ['train', 'val', 'test']:
    with open(f'./data/FSC147/{split}.txt', 'w') as f:
        for item in data[split]:
            f.write(f'{item}\n')
"
```

### Expected Directory Structure
```
./data/FSC147/
├── images_384_VarV2/
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
├── gt_density_map_adaptive_384_VarV2/
│   ├── 2.npy
│   ├── 3.npy
│   └── ...
├── annotation_FSC147_384.json
├── Train_Test_Val_FSC_147.json
├── ImageClasses_FSC147.txt
├── train.txt
├── val.txt
└── test.txt
```

## Model Checkpoints

### Download Pretrained Models (Original paper)

1. **Main counting model** (required for inference)
   - Download from [Baidu Disk](https://pan.baidu.com/s/11sbdDYLDfTOIPx5pZvBpmw?pwd=paeh) (Password: `paeh`)
   - Save to `./data/checkpoint_FSC.pth`

2. **Binary classifier** (optional, for exemplar selection)
   - Download from [Baidu Disk](https://pan.baidu.com/s/1fOF0giI3yQpvGTiNFUI7cQ?pwd=psum) (Password: `psum`)
   - Save to `./data/out/classify/`

3. **MAE pretrained backbone** (required for training)
   - Download from official MAE repository
   - Save to `./weights/mae_pretrain_vit_base_full.pth`

##  Quick Start

### Advanced Demo with Visualization

```bash
# Run advanced demo with visual outputs
python demo_app_advanced.py \
    --resume ./data/checkpoint_FSC.pth \
    --data_path ./data/FSC147 \
    --output_dir ./demo_outputs \
    --visualize
```

### Official Testing

```bash
# Test on FSC147 test set
python FSC_test.py \
    --output_dir ./data/out/results_base \
    --resume ./data/checkpoint_FSC.pth \
    --data_path ./data/FSC147 \
    --split test
```

##  Training

### Step 1: Prepare Binary Classifier (Optional)

```bash
# Generate dataset for classifier
python datasetmake.py --data_path ./data/FSC147

# Train binary classifier
python biclassify.py \
    --data_path ./data/FSC147 \
    --output_dir ./data/out/classify \
    --epochs 100
```

### Step 2: Generate Exemplars

```bash
# Generate positive exemplars using Grounding DINO
python grounding_pos.py --root_path ./data/FSC147/

# Generate negative exemplars
python grounding_neg.py --root_path ./data/FSC147/
```

Alternative: Use YOLO for exemplar generation
```bash
# With prompts
python yolo_pos_withPrompt.py --root_path ./data/FSC147/

# Without prompts
python yolo_pos_withoutPrompt.py --root_path ./data/FSC147/
```

### Step 3: Pretraining (Optional)

```bash
# Pretrain the model
python FSC_pretrain.py \
    --data_path ./data/FSC147 \
    --output_dir ./data/out/pre_4_dir \
    --resume ./weights/mae_pretrain_vit_base_full.pth \
    --epochs 300 \
    --batch_size 8 \
    --lr 1e-4
```

### Step 4: Fine-tuning

```bash
# Fine-tune with positive exemplars
python FSC_train.py \
    --data_path ./data/FSC147 \
    --anno_file annotation_FSC147_pos.json \
    --output_dir ./data/out/finetune_pos \
    --resume ./data/out/pre_4_dir/checkpoint-latest.pth \
    --epochs 500 \
    --batch_size 8 \
    --lr 1e-5
```

##  Inference


### Batch Inference

```bash
python FSC_test.py
```

### Streamlit demo

```bash
# Configure the Gemini API key once (see ../.env.example) and then:
streamlit run demo_app_advanced.py
```



##  Citation

If you build on this code, please cite the original VA-Count paper and
acknowledge this CS338.Q21 fork:

```bibtex
@inproceedings{zhu2024zero,
  title={Zero-shot Object Counting with Good Exemplars},
  author={Zhu, Huilin and Yuan, Jingling and Yang, Zhengwei and Guo, Yu and Wang, Zheng and Zhong, Xian and He, Shengfeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

## Acknowledgement

This project is based on:
- [CounTR](https://github.com/Verg-Avesta/CounTR) - Base counting architecture
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Exemplar detection
- [MAE](https://github.com/facebookresearch/mae) - Vision Transformer backbone

The CS338.Q21 implementation builds on these works to deliver the
training, evaluation, exemplar-generation and demo pipeline described
above. We are very grateful for all of these excellent works.

## Contact

For questions about this CS338.Q21 project, please contact the team leader:
**Nguyen Cong Phat — `23521143@gm.uit.edu.vn`**.
For questions about the original VA-Count paper, please refer to the
authors of the upstream repository.

## License

This project is licensed under the MIT License — see [LICENSE.txt](../../LICENSE.txt) at the
repository root for details.
