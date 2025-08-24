import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

# Define class labels
class_labels = ['neutral', 'disgust', 'fear', 'happy', 'angry', 'sad', 'surprise']

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet18 and modify the final layer
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load('best_resnet18_fer.pth', map_location=device))
model.to(device)
model.eval()

# Define image transformation (assuming trained on 3-channel normalized input)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop the face region
            if face.size == 0:
                continue

            face_input = transform(face).unsqueeze(0).to(device)

            outputs = model(face_input)
            _, predicted = torch.max(outputs, 1)
            label = class_labels[predicted.item()]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
