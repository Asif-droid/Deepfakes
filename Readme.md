# Deepfake Generation and Detection

This repository contains Jupyter notebooks and scripts for both generating deepfake videos using face swapping techniques and detecting deepfakes using state-of-the-art deep learning models. The project leverages libraries such as InsightFace, PyTorch, Hugging Face Transformers, and OpenCV, and is designed to run efficiently on Google Colab with GPU acceleration.

---

## Features

### 1. Deepfake Generation ([deepfake.ipynb](deepfake.ipynb))
- **Face Swapping with InsightFace:**  
  Utilizes the InsightFace library and ONNX models to swap faces in images and videos.
- **Batch Processing:**  
  Automates the generation of multiple deepfake videos by iterating over folders of source images.
- **Custom Video Generation:**  
  Functions to generate deepfake videos by swapping a source face onto all faces detected in each frame of a target video.
- **VAE-based Face Modeling:**  
  Implements a Variational Autoencoder (VAE) for face image modeling and reconstruction.
- **GAN Prototyping:**  
  Includes prototype code for a GAN-based face swapper with identity and adversarial losses.

### 2. Deepfake Detection ([Deepfake_detector_v2.ipynb](Deepfake_detector_v2.ipynb))
- **Frame Extraction:**  
  Extracts frames from real and fake videos for dataset creation.
- **Dataset Preparation:**  
  Supports saving/loading datasets as compressed NumPy arrays for efficient storage and retrieval.
- **Model Architectures:**  
  - Fine-tuning of SigLIP and ViT transformer models for image classification.
  - Custom LSTM classifier for sequence modeling on video frames.
- **Training & Evaluation:**  
  - Training scripts with metrics reporting (accuracy, classification report).
  - Model checkpointing and best model saving.
- **Inference:**  
  Functions for classifying single images or batches, with support for both PIL and NumPy formats.

---

## Directory Structure

```
Deepfake/
│
├── Deepfake_detector_v2.ipynb   # Deepfake detection pipeline (training, evaluation, inference)
├── deepfake.ipynb               # Deepfake generation and face swapping
└── ...                          # (Other scripts, data, or checkpoints)
```

---

## Requirements

- Python 3.8+
- Google Colab (recommended for GPU support)
- PyTorch
- torchvision
- transformers
- datasets
- evaluate
- Pillow
- OpenCV
- tqdm
- imbalanced-learn
- InsightFace
- onnxruntime-gpu
- kagglehub (for dataset download)
- matplotlib

Install dependencies (in Colab or locally):
```bash
pip install torch torchvision transformers datasets evaluate pillow opencv-python tqdm imbalanced-learn insightface onnxruntime-gpu kagglehub matplotlib
```

---

## Usage

### Deepfake Generation

1. **Mount Google Drive:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install dependencies:**
    ```python
    !pip install numpy insightface onnxruntime-gpu kagglehub
    ```

3. **Download and prepare datasets:**
    - Use `kagglehub` to download face datasets.
    - Use provided scripts to copy and organize data.

4. **Run face swapping:**
    - Use the `generate_fake` function to create deepfake videos.
    - Use `automate_gen_folder` or `automate_all_generate` for batch processing.

5. **VAE and GAN Prototyping:**
    - Train the VAE model on face images for unsupervised representation learning.
    - Prototype GAN-based face swapping with identity and adversarial losses.

### Deepfake Detection

1. **Install dependencies:**
    ```python
    !pip install transformers torch pillow gradio hf_xet evaluate datasets accelerate huggingface_hub numpy opencv-python scikit-learn imbalanced-learn matplotlib
    ```

2. **Prepare datasets:**
    - Extract frames from real and fake videos.
    - Save datasets as compressed NumPy arrays.

3. **Train and evaluate models:**
    - Fine-tune transformer models or train custom LSTM classifier.
    - Use provided training loops and evaluation functions.

4. **Inference:**
    - Use `classify_image` or `predict_image` for single or batch predictions.

---

## Example: Generate a Deepfake Video

```python
source_img = cv2.imread('/path/to/source_face.jpg')
source_faces = app.get(source_img)
source_face = source_faces[0]
video_path = '/path/to/target_video.mp4'
output_video_path = '/path/to/output_deepfake.mp4'
generate_fake(video_path, output_video_path, source_face)
```

---

## Example: Detect Deepfakes

```python
from PIL import Image
image = Image.open('/path/to/frame.png').convert("RGB")
result = classify_image(image)
print(result)
```

---

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)

---

## License

This project is for research and educational