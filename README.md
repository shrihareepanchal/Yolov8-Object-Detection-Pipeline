# Yolov8-Pipeline

This project implements a simple yet modular pipeline using Ultralytics YOLOv8 for object detection. It loads a model, runs inference on an input image or video, and prints the detected classes with confidence scores.

# Setup Instructions

  1.Clone the Repository
  
      git clone <your-repo-url>
      cd Pipeline
          
  2.Create and Activate Virtual Environment

      python -m venv venv
      venv\Scripts\activate
          
  3.Install Dependencies

      pip install ultralytics

# How to Run the Code

      python main.py --source sample.jpeg --model yolov8s.pt

# Dependencies

    Python	3.9+
    torch	2.1.0+cpu
    numpy	1.26.4
    ultralytics	8.1.17
  


      
