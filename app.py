import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import streamlit as st
from PIL import Image
from datetime import datetime

# Set Tesseract OCR path (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
MODEL_PATH = r"D:/PycharmProjects/Automated Vehicle Number Plate Detection/models/best.pt"
model = YOLO(MODEL_PATH)

# Initialize text file to store detected plate texts with timestamps
text_file_path = "detected_plates_with_timestamp.txt"

# Streamlit UI
st.title("Car Number Plate Detection and Recognition")

# File uploader for images or videos
upload_type = st.radio("Select Upload Type", ("Image", "Video"))


def save_to_text_file(detected_texts):
    with open(text_file_path, "a") as file:
        # Adding headers if file is empty
        if file.tell() == 0:
            file.write(f"{'Timestamp':<25}{'Detected Plate'}\n")
            file.write(f"{'-' * 25}{'-' * 20}\n")

        # Write detected plates with timestamp
        for text in detected_texts:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp:<25}{text}\n")


if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Read image
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        # Perform detection
        results = model(image_np)

        # Annotate image with detections
        detected_texts = []
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_plate = image_np[y1:y2, x1:x2]

            # OCR on cropped region
            plate_text = pytesseract.image_to_string(cropped_plate, config="--psm 7").strip()
            detected_texts.append(plate_text)

            # Draw bounding box and text
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save detected texts with timestamps to the text file
        save_to_text_file(detected_texts)

        # Convert to RGB for display in Streamlit
        result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        st.image(result_image, caption="Detected Number Plate", use_column_width=True)

        st.success("Detected number plate texts saved to text file.")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4"])
    if uploaded_video:
        # Save uploaded video temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Process video
        cap = cv2.VideoCapture(temp_video_path)

        # Output video writer
        output_file_path = "output_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

        stframe = st.empty()  # Placeholder for displaying frames

        detected_texts = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model(frame)

            # Annotate frame with detections and extract text
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cropped_plate = frame[y1:y2, x1:x2]

                # OCR on cropped region
                plate_text = pytesseract.image_to_string(cropped_plate, config="--psm 7").strip()
                detected_texts.append(plate_text)

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write frame to output video
            out.write(frame)

            # Display frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption="Processing Video...", channels="RGB", use_column_width=True)

        # Release resources
        cap.release()
        out.release()

        # Save detected texts with timestamps to the text file
        save_to_text_file(detected_texts)

        # Display output video
        st.video(output_file_path)

        # Provide download link for output video
        with open(output_file_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="output_detected.mp4")

        st.success("Detected number plate texts saved to text file.")

# Add a download button for the detected text file
with open(text_file_path, "rb") as text_file:
    st.download_button("Download Detected Plates Table", text_file, file_name="detected_plates_with_timestamp.txt")
