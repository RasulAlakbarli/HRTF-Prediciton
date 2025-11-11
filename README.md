# HRTF Prediction from Pinna Images

## Problem Statement

Head-Related Transfer Functions (HRTFs) describe how sound waves interact with an individual's head, torso, and particularly their outer ears (pinnae) before reaching the eardrums. These functions are crucial for creating realistic spatial audio experiences, encoding directional information that our auditory system uses to localize sounds in 3D space.

### Real-World Applications

HRTFs are fundamental to numerous audio technologies:
- **Virtual Reality (VR) and Augmented Reality (AR)**: Creating immersive, spatially accurate soundscapes
- **Gaming**: Delivering precise positional audio for enhanced gameplay
- **Hearing Aids**: Improving sound localization for users
- **Telecommunications**: Enhancing conference call experiences with spatial audio
- **Cinema and Entertainment**: Creating 3D audio experiences in theaters and streaming platforms

### Current Challenges

Traditional HRTF measurement is a significant bottleneck in personalized spatial audio adoption:

1. **Expensive Equipment**: Requires specialized anechoic chambers and precise measurement rigs
2. **Time-Consuming Process**: Individual measurements can take 30-60 minutes per person
3. **Expert Operation**: Needs trained personnel to conduct measurements properly
4. **Limited Accessibility**: Most users cannot access measurement facilities
5. **Discomfort**: Participants must remain still for extended periods

These limitations result in most applications using generic HRTFs, which provide suboptimal spatial audio quality. **Predicting HRTFs from simple pinna photographs offers a practical, scalable solution** that could democratize personalized spatial audio.

---

## Dataset

This project uses the **[SONICOM HRTF Dataset](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/)**, which provides high-quality HRTF measurements paired with corresponding pinna images.

### Dataset Specifications
- **Subjects**: 100 human subjects
  - Training set: 90 subjects
  - Test set: 10 subjects
- **Images per Subject**: 14 grayscale pinna images (7 per ear)
- **Image Format**: Grayscale, resized to 256 × 256 pixels
- **Labels**: HRTF measurements corresponding to each subject
- **HRTF Output Dimensions**: 793 angles × 2 ears × 256 frequency bins

---

## Model Description

### Architecture Overview

The model uses a two-stage architecture combining convolutional and recurrent neural networks:

1. **ResNet Encoder**: Extracts high-level visual features from pinna images
2. **HRTF Model**: Processes features through bidirectional LSTM and generates HRTFs

### Detailed Architecture

#### ResNet Encoder
- **Base Model**: Pre-trained ResNet-50, modified for single-channel (grayscale) input
- **Purpose**: Extract high-level spatial features from pinna images
- **Input**: Grayscale images (256 × 256 pixels)
- **Output Shape**: `[batch_size × sides × views, 2048]`

#### HRTF Model

The HRTF Model consists of three sequential components:

1. **Bidirectional LSTM (Feature Processing)**
   - **Purpose**: Process sequences of ResNet features from multiple views
   - **Input Shape**: `[batch_size, sides × views, 2048]`
   - **Output Shape**: `[batch_size, 2048]` (summary features)
   - Captures temporal/spatial relationships between different pinna views

2. **Feature Expansion LSTM**
   - **Purpose**: Expand summary features to match HRTF spatial resolution
   - **Input Shape**: `[batch_size, 1, 1]`
   - **Output Shape**: `[batch_size, 1586, 2048]`
   - Generates features for each spatial angle

3. **Fully Connected Layers**
   - **Purpose**: Generate final HRTF predictions
   - **Output Shape**: `[batch_size, 793, 2, 256]`
     - 793: Number of spatial angles
     - 2: Left and right ear
     - 256: Frequency bins

### Loss Function

**Mean Spectral Distortion (MSD)**: Measures the distortion between predicted and ground-truth HRTFs in the spectral domain. This metric is particularly suited for evaluating perceptual quality of spatial audio.

## Model Performance

| Metric | Score |
|--------|-------|
| Mean Spectral Distortion (MSD) | **-63.6 dB** |

The negative MSD value indicates that the model successfully reconstructs HRTFs with minimal spectral distortion, suitable for practical spatial audio applications.

---

## Project Architecture

```
├── inference.py           # Script for running inference on new images
├── metrics.py            # Evaluation metrics for HRTF prediction quality
├── model.py              # Model architecture definition
├── plot_results.py       # Visualization tools for predictions and comparisons
├── requirements.txt      # Python dependencies
├── transformations.py    # Data augmentation and preprocessing transforms
└── utils.py             # Helper functions and dataset class
```
---

## Usage


### Setup
```bash
# Clone the repository
git clone git@github.com:RasulAlakbarli/HRTF-Prediciton.git
cd HRTF-Prediciton

# Install dependencies
pip install -r requirements.txt
```


### Inference

The model supports three different evaluation tasks based on the number of input images:

#### Task 0: Full Image Set (19 images per ear)
```bash
python inference.py -l left_img1.jpg left_img2.jpg ... left_img19.jpg \
                    -r right_img1.jpg right_img2.jpg ... right_img19.jpg \
                    -o output.sofa
```

#### Task 1: Reduced Image Set (7 images per ear)
```bash
python inference.py -l left_img1.jpg left_img2.jpg ... left_img7.jpg \
                    -r right_img1.jpg right_img2.jpg ... right_img7.jpg \
                    -o output.sofa
```

#### Task 2: Minimal Image Set (3 images per ear)
```bash
python inference.py -l left_img1.jpg left_img2.jpg left_img3.jpg \
                    -r right_img1.jpg right_img2.jpg right_img3.jpg \
                    -o output.sofa
```

#### Command-Line Arguments

- `-l, --left`: List of left ear pinna image paths (space-separated)
- `-r, --right`: List of right ear pinna image paths (space-separated)
- `-o, --output_path`: Output path for the predicted HRTF in SOFA format

**Note**: The model accepts an arbitrary number of images per ear, but the three tasks above represent standard evaluation scenarios with different data availability constraints.

#### Output Format

The predicted HRTFs are saved in **SOFA (Spatially Oriented Format for Acoustics)** format, which is the standard file format for storing spatial audio data. The output file contains:
- Head-Related Impulse Responses (HRIRs) for 793 spatial directions
- Left and right ear measurements
- 256 time samples per impulse response

### Evaluation

To evaluate the model on all three tasks with the test dataset:

1. Ensure `config.py` points to your dataset directory
2. Uncomment the `evaluate()` function call in `inference.py`:
   ```python
   if __name__ == "__main__":
       evaluate()  # Uncomment this line
       # main()
   ```
3. Run the evaluation:
   ```bash
   python inference.py
   ```

This will:
- Load the trained model
- Process the test set for all three tasks (19, 7, and 3 images per ear)
- Calculate Mean Spectral Distortion (MSD) scores
- Generate evaluation metrics for each task

### Visualization

Plot and compare predicted HRTFs against ground truth:

```bash
python plot_results.py
```

---


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please contact me at alakbarlirasul@gmail.com
