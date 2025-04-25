# Models

This directory contains model architectures and instructions for obtaining trained models for the Agro Nexus application.

## Model Files

The trained model files are not included in the Git repository due to size constraints. You can either download pre-trained models or train them yourself.

### Pre-trained Models

1. Plant Disease Detection Model (CNN)
   - Download from: [provide download link]
   - Place in: `models/plant_disease_model.h5`
   - Architecture: CNN (defined in `CNN.py`)

2. Plant Height Estimation Model
   - Download from: [provide download link]
   - Place in: `models/plant_height_model.pkl`
   - Architecture: Defined in `plantheight.py`

### Training Your Own Models

1. Plant Disease Detection Model:
```bash
python train_disease_model.py --data_dir datasets/plant_disease --epochs 50
```

2. Plant Height Estimation Model:
```bash
python train_height_model.py --data_dir datasets/plant_height --epochs 30
```

## Model Architecture

The model architecture definitions are included in this repository:
- `CNN.py`: Contains the CNN architecture for plant disease detection
- `plantheight.py`: Contains the model architecture for plant height estimation

## Checkpoints

During training, model checkpoints will be saved in `models/checkpoints/`. These are also excluded from Git. 