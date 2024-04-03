import streamlit as st
from path_setup import WEIGHTS_YOLOV9, WEIGHTS_YOLOV8, WEIGHTS_YOLOV5
from utils import (video_visualization, image_visualization, image_visualization_supervision,
                   video_visualization_supervision, list_files_in_directory)


def main():
    st.set_page_config(page_title='Face detection')
    st.title('Face detection of anime characters')

    # Store video and image uploaded by the user
    uploader_video = None
    uploader_image = None

    with st.sidebar:
        st.header('Settings')

        # Multimedia selection
        multimedia = st.selectbox('Choose Image or Video', ['Video', 'Image'])

        if multimedia == 'Video':
            uploader_video = st.file_uploader('Please upload', type=['mp4', 'mpeg4'])
        if multimedia == 'Image':
            uploader_image = st.file_uploader('Please upload', type=['jpeg', 'jpg', 'png'])

        # Model and weight selection
        model = st.selectbox('Select model', ['Yolov9', 'Yolov8', 'Yolov5'])

        if model == 'Yolov9':
            weight_type = st.selectbox('Select weight', list_files_in_directory(WEIGHTS_YOLOV9))
        if model == 'Yolov8':
            weight_type = st.selectbox('Select weight', list_files_in_directory(WEIGHTS_YOLOV8))
        if model == 'Yolov5':
            weight_type = st.selectbox('Select weight', list_files_in_directory(WEIGHTS_YOLOV5))

        # Confidence selection
        confidence = st.slider('Select confidence', 0, 100, 60) / 100

        # On or Off using library SuperVision by Roboflow
        on_supervision = st.toggle('Use of SuperVision by Roboflow')

    # Processing of uploaded docs with SuperVision option
    if st.sidebar.button('Process'):
        if uploader_video is not None:
            if on_supervision:
                video_visualization_supervision(uploader_video, weight_type, confidence)
            else:
                video_visualization(uploader_video, weight_type, confidence)

        if uploader_image is not None:
            if on_supervision:
                image_visualization_supervision(uploader_image, weight_type, confidence)
            else:
                image_visualization(uploader_image, weight_type, confidence)


if __name__ == '__main__':
    main()
