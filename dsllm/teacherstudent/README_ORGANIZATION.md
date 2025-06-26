# Teacher-Student Data Enhancement Project Organization

## 📁 Folder Structure

```
teacherstudent/
├── README_ORGANIZATION.md          # This file
├── teacher_student_model.py        # Main model implementations
├── run_data_enhancement.py         # Training script for data enhancement
├── enhancement_evaluation.ipynb    # Comprehensive evaluation notebook
├── trained/                        # 🎯 All trained models
│   ├── best_enhancement_model.pth  # Main data enhancement model
│   ├── best_teacher.pth           # Teacher model checkpoint
│   ├── best_student.pth           # Student model checkpoint
│   └── teacher_student_checkpoint.pth # Training checkpoint
└── evaluation/                     # 📊 All evaluation outputs
    ├── eval_run_YYYYMMDD_HHMMSS/  # Timestamped evaluation runs
    │   ├── enhancement_model_report.md
    │   ├── distribution_analysis.png
    │   ├── frequency_analysis.png
    │   ├── error_analysis.png
    │   ├── temporal_analysis.png
    │   └── sample_*_reconstruction.png
    └── training_demo_YYYYMMDD_HHMMSS/ # Training demo outputs
        └── enhancement_comparison.png
```

## 🚀 Usage

### Training
```bash
python run_data_enhancement.py
```
- Outputs trained model to: `trained/best_enhancement_model.pth`
- Saves demo visualizations to: `evaluation/training_demo_*/`

### Evaluation
```bash
jupyter notebook enhancement_evaluation.ipynb
```
- Creates new timestamped folder: `evaluation/eval_run_YYYYMMDD_HHMMSS/`
- Generates comprehensive analysis report and visualizations
- All outputs organized by evaluation run

### Model Loading
```python
# Load trained model
checkpoint = torch.load('trained/best_enhancement_model.pth')

# Use for enhancement
model = DataEnhancementTeacherStudentModel(config)
model.load_state_dict(checkpoint)
enhanced_data = model.enhance_data(low_res_input)
```

## 📊 Evaluation Outputs

Each evaluation run creates a timestamped folder containing:
- **Performance Report**: Comprehensive markdown analysis
- **Sample Reconstructions**: Visual comparisons of enhancement quality
- **Statistical Analysis**: Distribution and frequency domain analysis  
- **Error Analysis**: Outlier detection and quality metrics
- **Temporal Analysis**: Pattern preservation evaluation

## 🎯 Integration with SensorLLM

The data enhancement model creates higher-resolution sensor data suitable for SensorLLM input:
- **Input**: Low-res sensor data (30 timesteps, 0.1Hz sampling, 1 sample/10sec)
- **Output**: High-res sensor data (300 timesteps, 1Hz sampling)  
- **Ready for**: Direct integration with SensorLLM pipeline 