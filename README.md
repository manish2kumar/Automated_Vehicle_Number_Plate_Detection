# Automated_Vehicle_Number_Plate_Detection

An AI-powered system to detect and extract vehicle number plates from images and videos using YOLOv8 for detection and Tesseract OCR for text extraction. This project processes input files (images or videos), detects number plates, extracts the text, and stores it with timestamps in a tabular format.

---

## Features
- **Real-Time Detection**:- Detect vehicle number plates from images and videos.

- **Text Extraction**:- Recognize and extract text from detected number plates using OCR.

- **Result Logging**:- Stores detected number plates with timestamps in a text file.

- **Dual Input Modes**:- Supports both image and video uploads.

---

## Demo

![Home Page](https://github.com/manish2kumar/Automated_Vehicle_Number_Plate_Detection/blob/main/result/home_page.png)
![Output Image](https://github.com/manish2kumar/Automated_Vehicle_Number_Plate_Detection/blob/main/result/output_image.png)
![Output Timestamp](https://github.com/manish2kumar/Automated_Vehicle_Number_Plate_Detection/blob/main/result/output_timestamp.png)

---

## Table of Contents
1. [Installation](#installation)
2. [Technologies Used](#technologies-used)
3. [How to Use](#how-to-use)
4. [System Architecture](#system-architecture)
5. [Results](#results)
6. [Dataset](#dataset)
7. [Contributors](#contributors)

---

## Installation

### Prerequisites
- Python 3.9 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract): Ensure it is installed and added to your system PATH.
- Install required Python libraries:
```bash
  pip install -r requirements.txt
```
---

### Clone the Repository  

- Clone this repository to your local machine.

```http
  git clone https://github.com/manish2kumar/Automated_Vehicle_Number_Plate_Detection
```
---

## Technologies Used

- **YOLOv8**: For number plate detection.

- **Tesseract OCR**: For extracting text from detected plates.

- **Streamlit**: For building an interactive user interface.

- **OpenCV**: For image and video processing.

---

## How to Use

- **1. Run the Application**:

```bash
  streamlit run app.py
```
- **2. Upload Files**:

  - Upload an image or video for detection.
  - View the results on the app interface.

- **3. Output**:

  - Detected number plates with timestamps will be displayed in the app.
  - Data will also be saved in a text file in a tabular format.

---  

## System Architecture

The system processes inputs through the following steps:

   - Accepts image/video files via the Streamlit interface.
   - Detects vehicle number plates using YOLOv8.
   - Crops the detected plates and extracts text with Tesseract OCR.
   - Logs results (text and timestamp) into a text file.

---

## Results

### Accuracy
Detection and text extraction accuracy depend on:
- **Image Quality**: High-resolution images improve results.
- **YOLO Model Training**: Performance varies based on the quality and diversity of the training dataset.

### Example Output
The system logs detected number plates with timestamps in a tabular format:

| **Timestamp**       | **Detected Number Plate** |
|---------------------|---------------------------|
| 2024-11-23 14:12:01 | MH12AB1234               |
| 2024-11-23 14:13:05 | KA05CD5678               |


---

## Dataset
- This project uses the [Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).

---

## Contributors
- Manish Kumar - [GitHub Profile](https://github.com/manish2kumar)
- Ayush Kaushal - [GitHub Profile]( https://github.com/Ayushkaushal13)

