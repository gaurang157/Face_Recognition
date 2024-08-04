# Image Upload and Recognition App

This is a Streamlit application for uploading images, detecting faces, and annotating them with predictions. The application uses the YOLO model for classification and DeepFace for face detection.

![Recognition App](https://raw.githubusercontent.com/gaurang157/Face_Recognition/main/Screenshot%20(5354).png)

## Features

- **Upload Image**: Allows users to upload images in PNG or JPG format.
- **Face Detection**: Detects faces in the uploaded image using the RetinaFace detector from the DeepFace library.
- **Image Annotation**: Annotates detected faces with bounding boxes and predicted class names using the YOLO model.
- **Sample Images**: Displays sample images for demonstration purposes.

## Installation

1. **Clone the Repository**

```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
```

This guide outlines the steps to set up a Streamlit application, including creating a virtual environment, installing dependencies, and running the application.

2. Create a Virtual Environment

- **Purpose:** A virtual environment isolates project dependencies, preventing conflicts with other Python projects.
- **Command:**
```bash
python -m venv env
```
- **Activate the Environment:**
- **Linux/macOS:**
```bash
source env/bin/activate
```
- **Windows:**
```bash
env\Scripts\activate
```

3. Install Dependencies

- **Purpose:** Install necessary packages listed in `requirements.txt`.
- **Command:**
```bash
pip install -r requirements.txt
```

4. Download Custom Face Recognition Model
- **Command:**
```bash
gdown 11UYuNj0cn2x56xZRHWdAQD09iuFf6r_f
```

### 4. Run the Application

- **Purpose:** Start the Streamlit app.
- **Command:**
```bash
streamlit run main.py
```
**Note:** Replace `app.py` with the actual name of your Streamlit app file.

This command will launch your Streamlit application in your browser. 

**Further Steps:**

- Replace `app.py` with the actual name of your Streamlit file.
- Create a `requirements.txt` file with a list of your project's dependencies.
- Build your Streamlit app within the `app.py` file.
