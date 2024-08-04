import pathlib
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deepface import DeepFace
import tempfile
import os
st.set_page_config(layout="wide")
# Load the YOLO model
model = YOLO(r'C:\Users\gaura\Desktop\iPython\Face_Recog\runs\classify\train3\weights\best.pt')

def annotate_and_plot(image_path, model):
    st.write(image_path)
    # Load the image
    image = Image.open(image_path)

    # Detect faces
    obj = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface')

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Annotate each detected face
    for i, face in enumerate(obj):
        # Get bounding box coordinates
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']

        # Create a rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Predict class using the model
        results = model(image_path)

        # Get the highest probability class name
        for r in results:
            probs = r.probs
            conf_ = probs.top1conf.item()
            max_prob_index = probs.top1
            class_name = model.names[max_prob_index]
            print(f"Predicted class: {class_name}: {conf_:.2f}")

        # Add text label
        ax.text(x, y-10, f'{class_name}: {conf_:.2f}', color='r')

    return fig

def main():
    st.title('Image Upload and Recognize')
    
    # Display sample images
    st.subheader('Sample Images')
    sample_images = [r"C:\Users\gaura\Downloads\image (1).jpg", r"C:\Users\gaura\Desktop\iPython\Face_Recog\image-retrieval\dataset\pa\Akshay\akshaykumar-20240804-0007.jpg"]  # Update with your sample image paths

    # for img_path in sample_images:
    col1, col2 = st.columns([1,2])
    with col1:
        image = Image.open(sample_images[0])
        st.image(image, caption=os.path.basename(sample_images[0]), use_column_width=True)
        image = Image.open(sample_images[1])
        st.image(image, caption=os.path.basename(sample_images[1]), use_column_width=True)
    with col2:
        st.subheader('Upload your image')
        img_file = st.file_uploader(label='Choose an image file', type=['png', 'jpg'])
        
        if img_file:
            img = Image.open(img_file)
            file_extension = os.path.splitext(img_file.name)[1]
            
            # Create a temporary file to save the uploaded image with the correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(img_file.getbuffer())
                temp_image_path = temp_file.name
            
            # Annotate and plot the image
            fig = annotate_and_plot(temp_image_path, model)
            
            # Display the plot
            st.pyplot(fig)
    # with col3:
    #     image = Image.open(sample_images[1])
    #     st.image(image, caption=os.path.basename(sample_images[1]), use_column_width=True)

if __name__ == '__main__':
    main()
