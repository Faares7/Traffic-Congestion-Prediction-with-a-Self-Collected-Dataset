# Traffic Congestion Prediction with a Self-Collected Dataset using Image Analysis

## 📌 Project Overview
This project addresses the lack of localized traffic datasets in Egypt by introducing a novel, self-collected dataset for traffic congestion prediction . Using images captured from critical locations on the 26th of July Corridor, we employ Deep Learning and Computer Vision techniques to classify traffic into three levels: **High**, **Medium**, and **Low** .

The system utilizes Transfer Learning with state-of-the-art Convolutional Neural Networks (CNNs) and Ensemble Learning to achieve high accuracy, aiming to assist in intelligent transportation systems and urban planning .

## 🌍 Study Area & Dataset
Data was collected from two high-density locations in Giza, Egypt, chosen for their critical nature and traffic diversity :
1.  **Juhayna Square:** A major hub connecting residential and commercial zones .
2.  **MUST University Pedestrian Bridge:** chosen for its mix of vehicular and pedestrian activity .

### Dataset Statistics
* **Initial Size:** 370 images (highly imbalanced) .
* **Final Size:** 2,240 images (after augmentation) .
* **Classes:**
    * 🔴 **High Traffic:** 735 images .
    * 🟡 **Medium Traffic:** 742 images .
    * 🟢 **Low Traffic:** 763 images .
* **Split:** 60% Training, 20% Validation, 20% Testing .

## 🧠 Methodology
We implemented a transfer learning approach using pre-trained models on ImageNet . The pipeline includes:
1.  **Preprocessing:** Resizing to 224x224, normalization (0-1), and RGB conversion .
2.  **Augmentation:** Horizontal flipping, shifting, rotation, brightness adjustment, and zooming to balance classes .
3.  **Models Evaluated:** VGG16, Inception V3, MobileNet, ResNet, DenseNet, and EfficientNetB0 .
4.  **Ensemble Learning:** Combining predictions to improve robustness .

## 📊 Results
The models were evaluated on accuracy, precision, recall, and F1-score .

| Model | Test Accuracy | Training Accuracy | Key Findings |
| :--- | :--- | :--- | :--- |
| **Inception V3** | **98.65%** | 92.00% | Top performer; excellent balance of size and accuracy . |
| **VGG16** | 98.37% | 98.79% | High accuracy but computationally expensive . |
| **MobileNet** | 96.39% | 99.80% | High efficiency with minimal overhead; ideal for real-time use . |
| **DenseNet** | 95.95% | 99.85% | Robust performance with efficient feature reuse . |
| **ResNet50** | 85.14% | 87.40% | Struggled with "Medium" traffic classification due to feature capture limitations . |

## 🚀 Future Work
* Integration into **Speed Management Systems** to adjust speed limits dynamically .
* Expansion to **real-time video feeds** for continuous monitoring .
* Inclusion of additional locations to improve dataset generalizability .

## 👥 Authors
* **F. A. Moustafa**, **H. M. Nasr**, **M. Nashaat**, **M. Hossam**, **A. Hossam**
* *School of Information Technology and Computer Science (ITCS), Nile University, Egypt* .
