# Tomato-Leaf-Disease-Detection

## 🌱 Introduction

Agriculture plays a crucial role in sustaining society, providing food and resources worldwide. However, plant diseases pose a significant challenge by affecting the quality and yield of crops, often leading to substantial economic losses. Early detection and treatment are critical to minimizing damage, but traditional methods, which rely on expert visual inspections, are often time-consuming and impractical for large-scale farming operations.

Tomato plants, in particular, are susceptible to various diseases, such as **early blight**, **late blight**, and **bacterial spot**, caused by bacteria and fungi. Detecting these diseases early can significantly enhance crop productivity and reduce losses.

---

## 📋 Project Overview

This project focuses on the development and application of a **deep convolutional neural network (CNN)** to detect tomato leaf diseases effectively.

### ✨ Key Features:
- **Dataset**: A public dataset containing **11,102 images** of healthy and diseased plant leaves collected under controlled conditions, filtered for tomato-specific images.
- **Model**: A CNN-based model trained to classify tomato leaves as healthy or affected by specific diseases.
- **Goal**: Provide a practical, automated solution for detecting tomato crop diseases to assist farmers in managing their crops more effectively.

---

## 🌟 Benefits

By leveraging CNNs and large datasets, this project aims to:
- Improve productivity in tomato farming.
- Help farmers make informed decisions quickly.
- Reduce economic losses caused by plant diseases.

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure the following software and libraries are installed:
- Python 3.8 or higher
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

### 📦 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/peteksener/Tomato-Leaf-Disease-Detection.git
   cd Tomato-Leaf-Disease-Detection
   ```
2. **Prepare the Dataset:**
   - Download the dataset from [(https://www.kaggle.com/datasets/emmarex/plantdisease/)](#).
   - Split the dataset to 'train', 'val', 'test' folders.
   - Place the folders in directory
   - Replace datapaths in train.py with paths of your train, validation and test folders.
     
3. **Install Required Python Version:**
   ```bash
   brew install python@3.10
   ```
4. **Create Virtual Environment:**
   ```bash
   python3.10 -m venv tf_env
   source tf_env/bin/activate
   ```
5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the Model:**
   ```bash
   python train.py
   ```
5. **Test the Model:**
   ```bash
   python test_model.py
   ```

---

## 📊 Results

The CNN model demonstrated high accuracy in classifying tomato leaf diseases across 10 categories, achieving a test accuracy of 98.9%. Data augmentation techniques improved the model’s ability to generalize within the scope of the dataset. However, challenges arise when applying the model to images found online, where multiple leaves are often present in a single picture. In contrast, our dataset consists of images with only one leaf per picture, leading to mispredictions in such cases. Interestingly, when the background of online images is modified to resemble the backgrounds in our dataset, the model’s predictions improve significantly. This suggests that background consistency plays a critical role in the model’s performance and highlights a limitation of the current dataset’s diversity.Additionally, the dataset imbalance is a significant factor contributing to the high loss in the validation set. The imbalance between different disease classes causes the model to favor the majority class, resulting in poorer performance on the minority classes during validation. This further highlights the need for a more balanced and diverse dataset to ensure the model can generalize better to new, unseen data.These findings emphasize the importance of creating datasets that better reflect real-world conditions. Future work could address these limitations by incorporating more varied and realistic samples to enhance the model’s robustness. Overall, this project illustrates the potential of AI in agriculture,offering scalable solutions for early disease detection and crop management.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please:
- Fork the repository.
- Create a feature branch.
- Submit a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- The creators of the public dataset used in this project.
- Open-source frameworks like TensorFlow and Keras.



