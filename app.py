import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings
import tensorflow as tf
import numpy as np
import av
from utils import visualize_boxes_and_labels_on_image_array

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.title('Face Mask Detection')


@st.cache
def load_model():
    detect_fn = tf.saved_model.load('my_model_mobnet/saved_model')

    return detect_fn


detect_fn = load_model()


class MaskDetector(VideoProcessorBase):
    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        self.category_index = {1: {'id': 1, "name": 'with_mask'}, 2: {'id': 2, 'name': 'without_mask'},
                                3: {'id': 3, 'name': 'mask_weared_incorrect'}}
        self.num_boxes = 1

    def gen_pred(self, image):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        visualize_boxes_and_labels_on_image_array(
            image,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=self.num_boxes,
            min_score_thresh=self.confidence_threshold,
            agnostic_mode=False)

        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        image = self.gen_pred(image)

        return av.VideoFrame.from_ndarray(image, format="bgr24")



webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    video_processor_factory=MaskDetector,
    async_processing=True,
)

confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
num_boxes = st.slider('Number of boxes', 1, 5, 1)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
    webrtc_ctx.video_processor.num_boxes = num_boxes





