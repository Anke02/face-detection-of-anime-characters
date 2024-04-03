from numpy import ndarray
from supervision import BoxCornerAnnotator, LabelAnnotator
from ultralytics import YOLO
from path_setup import WEIGHTS_YOLOV9, WEIGHTS_YOLOV8, WEIGHTS_YOLOV5
import cv2
import streamlit as st
import tempfile
import supervision as sv
import os


# Get list files in directory
def list_files_in_directory(folder_path) -> list[str]:
    files_list = os.listdir(folder_path)
    return files_list


# Get path weight
def path_to_weight(weight_type) -> str:
    if weight_type in list_files_in_directory(WEIGHTS_YOLOV9):
        return WEIGHTS_YOLOV9 / weight_type

    if weight_type in list_files_in_directory(WEIGHTS_YOLOV8):
        return WEIGHTS_YOLOV8 / weight_type

    if weight_type in list_files_in_directory(WEIGHTS_YOLOV5):
        return WEIGHTS_YOLOV5 / weight_type


# Load the model
def load_model(weight_type) -> YOLO:
    weight = path_to_weight(weight_type)
    model = YOLO(weight)
    return model


# Save file to a temporary file
def save_to_tempfile(file) -> str:
    t_file = tempfile.NamedTemporaryFile(delete=False)
    t_file.write(file.read())
    return t_file.name


# Standard size frame or image
def resize_frame(frame):
    return cv2.resize(frame, (1280, 720))


# Visualization image
def image_visualization(image, weight_type, confidence):
    t_image = save_to_tempfile(image)
    image = cv2.imread(t_image)
    image = resize_frame(image)

    model = load_model(weight_type)
    results = model(image, conf=confidence)
    annotated_image = results[0].plot()

    st.image(annotated_image, channels='BGR')


# Visualization video
def video_visualization(video, weight_type, confidence):
    t_video = save_to_tempfile(video)
    cap = cv2.VideoCapture(t_video)
    model = load_model(weight_type)

    st_frame = st.empty()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame = resize_frame(frame)
            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            st_frame.image(annotated_frame, channels='BGR')
        else:
            cap.release()
            break


# Style to label for SuperVision by Roboflow
def load_style_annotator() -> tuple[BoxCornerAnnotator, LabelAnnotator]:
    box_corner_annotator = sv.BoxCornerAnnotator(
        color=sv.Color.red(),
        thickness=7
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.red(),
        text_scale=1,
        text_thickness=2,
        text_padding=10,
        text_position=sv.Position.BOTTOM_CENTER
    )
    return box_corner_annotator, label_annotator


# Display frame with style for Supervision
def display_frame(frame, model, confidence) -> ndarray | ndarray:
    frame = resize_frame(frame)
    results = model(frame, conf=confidence)[0]
    detections = sv.Detections.from_ultralytics(results)

    box_corner_annotator, label_annotator = load_style_annotator()
    labels = [f'{model.names[class_id]} {confidence:0.2f}'
              for class_id, confidence in zip(detections.class_id, detections.confidence)]

    annotated_image = box_corner_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image


# Visualization image using SuperVision by Roboflow
def image_visualization_supervision(image, weight_type, confidence):
    t_image = save_to_tempfile(image)
    image = cv2.imread(t_image)
    model = load_model(weight_type)

    annotated_image = display_frame(image, model, confidence)
    st.image(annotated_image, channels='BGR')


# Visualization video using SuperVision by Roboflow
def video_visualization_supervision(video, weight_type, confidence):
    t_video = save_to_tempfile(video)
    cap = cv2.VideoCapture(t_video)
    model = load_model(weight_type)

    st_frame = st.empty()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            annotated_frame = display_frame(frame, model, confidence)
            st_frame.image(annotated_frame, channels='BGR')
        else:
            cap.release()
            break
