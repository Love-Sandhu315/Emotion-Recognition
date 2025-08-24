# 🎭 Real-Time Emotion Detection with ResNet18

This project uses **PyTorch** and **OpenCV** to perform **real-time facial emotion detection** through a webcam.  
A fine-tuned **ResNet18** model is used to classify emotions into 7 categories with a **validation accuracy of ~65%**:  

👉 `Neutral`, `Disgust`, `Fear`, `Happy`, `Angry`, `Sad`, `Surprise`


## ✨ Features
- 📷 Real-time face detection using **Haar Cascades**
- 🤖 Deep learning model (**ResNet18**) for emotion classification
- 🧠 Trained to recognize **7 emotions**
- 🖼 Overlay predictions directly on webcam feed


## 🚀 Installation

1. **Clone the repository**

git clone [https://github.com/your-username/emotion-detection.git](https://github.com/Love-Sandhu315/Emotion-Recognition.git)
cd emotion-detection


2. **Install dependencies**

pip install -r requirements.txt


3. **Add the trained model**


Run the script:

* A webcam window will open.
* Detected faces will be labeled with predicted emotions.
* Press **Q** to exit.


## 📂 Project Structure


emotion-detection/
│── main.py                  # Main script
│── best_resnet18_fer.pth    # Trained model weights (not uploaded if >100MB)
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── .gitignore


## 🛠 Tech Stack

* [PyTorch](https://pytorch.org/) – Deep Learning
* [Torchvision](https://pytorch.org/vision/stable/index.html) – Pretrained ResNet18
* [OpenCV](https://opencv.org/) – Face detection & video capture
* [NumPy](https://numpy.org/) – Array operations



## 📌 Future Improvements

* ✅ Use **MTCNN** or **Dlib** for more accurate face detection
* ✅ Deploy as a **web app** (Flask/Streamlit)
* ✅ Train on larger datasets for better accuracy



## 🙌 Acknowledgments

* [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* PyTorch & OpenCV community


## 👤 Author

**Lovepreet Singh**
🔗 [LinkedIn] (www.linkedin.com/in/lovepreet-singh-395b57231) 


