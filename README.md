# Google Landmark Recognition

A machine learning project for landmark recognition using various deep learning models.

## Project Structure

```
Google-Landmark-Recognition/
├── data/                          # Data processing and analysis
│   ├── data_downloader.ipynb     # Dataset download utilities
│   ├── download-dataset.sh       # Dataset download script
│   └── utils/                    # Data utilities
│       ├── data_analysis.py      # Data analysis tools
│       ├── data_utils.ipynb      # Data processing utilities
│       ├── label_selection_train.py
│       └── train_label_selection.ipynb
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── config.py            # Configuration files
│   │   ├── dataset.py           # Dataset classes
│   │   ├── utils.py             # Utility functions
│   │   ├── saved_models/        # Trained model checkpoints
│   │   └── *.ipynb              # Model training notebooks
│   ├── train_logs/              # Training logs and metrics
│   ├── evaluate_metrics.ipynb   # Model evaluation
│   ├── Model_Training_Metrics_Visualization.ipynb
│   └── visualize_sampled_images.py
└── README.md                     # This file
```

## Models Implemented

- **ResNet**: ResNet-18 and ResNet-50
- **EfficientNet**: EfficientNet-B0 and EfficientNet-B7
- **MobileNetV2**: MobileNetV2 for landmark detection
- **DenseNet**: DenseNet for landmark detection
- **SqueezeNet**: SqueezeNet for landmark detection
- **Vision Transformer (ViT)**: Transformer-based model
- **Swin Transformer**: Swin Transformer implementation

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/WeiXia-0000/Google-Landmark-Recognition.git
   cd Google-Landmark-Recognition
   ```

2. **Set up the environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install torch torchvision
   pip install jupyter notebook
   pip install pandas numpy matplotlib seaborn
   pip install scikit-learn
   ```

3. **Download the dataset**:
   ```bash
   cd data
   chmod +x download-dataset.sh
   ./download-dataset.sh
   ```

4. **Run model training**:
   - Navigate to `src/models/`
   - Open any of the model training notebooks (e.g., `resnet-18.ipynb`)
   - Follow the instructions in the notebook

## Data Processing

The project includes comprehensive data processing utilities:
- Data download and preprocessing
- Label selection and analysis
- Data visualization tools

## Model Training

Each model has its own training notebook with:
- Model architecture definition
- Training loop implementation
- Evaluation metrics
- Visualization of training progress

## Evaluation

Use the evaluation notebooks to:
- Compare model performance
- Analyze training metrics
- Visualize results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Google Landmark Dataset
- PyTorch community
- Various research papers and implementations
