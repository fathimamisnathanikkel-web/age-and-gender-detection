import hashlib
import io
from pathlib import Path
import sys

import numpy as np
import cv2
import streamlit as st
from PIL import Image

app_dir = Path(__file__).resolve().parent
module_dirs = tuple(
    dict.fromkeys(
        (
            app_dir,
            app_dir / "Age Gender Detector",
            Path.cwd(),
            Path.cwd() / "Age Gender Detector",
        )
    )
)
for module_dir in module_dirs:
    if (module_dir / "ml_features.py").is_file() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

# Set page configuration
st.set_page_config(
    page_title="Image, Traffic & Sign Detection",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
   .main-header{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
   }

   .sub-header{
        font-size : 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .result-text{
        font-size: 1.5rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .image-container{
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }

    .app-footer{
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }

    .stButton>button{
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }

    .stButton>button:hover{
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

AGE_RANGES = ("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
FACE_CONFIDENCE_THRESHOLD = 0.7


@st.cache_resource(show_spinner=False)
def load_traffic_detector():
    from traffic_detection import load_vehicle_detector

    return load_vehicle_detector()


@st.cache_resource
def load_face_age_models():
    model_subdirectory = "Age-and-Gender-Detection--main"
    app_dir = Path(__file__).resolve().parent
    model_dirs = tuple(
        dict.fromkeys(
            (
                app_dir / model_subdirectory,
                app_dir / "Age Gender Detector" / model_subdirectory,
                Path.cwd() / model_subdirectory,
                Path.cwd() / "Age Gender Detector" / model_subdirectory,
            )
        )
    )
    model_files = (
        "opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb",
        "age_deploy.prototxt",
        "age_net.caffemodel",
    )
    model_dir = next(
        (
            directory
            for directory in model_dirs
            if all((directory / filename).is_file() for filename in model_files)
        ),
        model_dirs[0],
    )
    face_config = model_dir / "opencv_face_detector.pbtxt"
    face_weights = model_dir / "opencv_face_detector_uint8.pb"
    age_config = model_dir / "age_deploy.prototxt"
    age_weights = model_dir / "age_net.caffemodel"

    model_paths = (
        face_config,
        face_weights,
        age_config,
        age_weights,
    )
    missing_files = [path.name for path in model_paths if not path.is_file()]
    if missing_files:
        searched_dirs = ", ".join(str(directory) for directory in model_dirs)
        st.error(
            f"Missing model files. Searched {searched_dirs}; missing in {model_dir}: "
            f"{', '.join(missing_files)}"
        )
        return None

    try:
        face_net = cv2.dnn.readNet(str(face_weights), str(face_config))
        age_net = cv2.dnn.readNet(str(age_weights), str(age_config))
    except cv2.error as exc:
        st.error(
            "Could not load the bundled Caffe models. They require OpenCV 4.x; "
            f"the current OpenCV version is {cv2.__version__}. Details: {exc}"
        )
        return None

    return face_net, age_net


def estimate_hair_length(frame, face_box):
    x1, y1, x2, y2 = face_box
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width <= 0 or face_height <= 0:
        return "Unclear"

    frame_height, frame_width = frame.shape[:2]
    sample_y1 = max(0, y1 - int(face_height * 0.15))
    jaw_y = min(frame_height, y2)
    sample_y2 = min(frame_height, y2 + int(face_height * 0.9))
    jaw_offset = jaw_y - sample_y1
    side_width = max(1, int(face_width * 0.55))
    side_bounds = (
        (max(0, x1 - side_width), x1),
        (x2, min(frame_width, x2 + side_width)),
    )
    side_runs = []
    upper_runs = []
    for side_x1, side_x2 in side_bounds:
        region = frame[sample_y1:sample_y2, side_x1:side_x2]
        if region.size == 0 or region.shape[1] < max(4, int(face_width * 0.12)):
            return "Unclear"

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127)) > 0
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        edges = cv2.Canny(gray, 35, 100) > 0
        hair_pixels = ((value < 175) | (saturation > 45)) & ~skin & edges
        row_coverage = np.mean(hair_pixels, axis=1)
        if len(row_coverage) < 3:
            return "Unclear"

        smoothed_coverage = np.convolve(
            row_coverage, np.ones(3, dtype=np.float32) / 3, mode="same"
        )
        upper_coverage = smoothed_coverage[:jaw_offset]
        lower_coverage = smoothed_coverage[jaw_offset:]
        if upper_coverage.size == 0 or lower_coverage.size == 0:
            return "Unclear"

        upper_rows = upper_coverage >= 0.04
        padded_upper = np.concatenate(([False], upper_rows, [False]))
        upper_transitions = np.diff(padded_upper.astype(np.int8))
        upper_starts = np.flatnonzero(upper_transitions == 1)
        upper_ends = np.flatnonzero(upper_transitions == -1)
        upper_run = (
            int(np.max(upper_ends - upper_starts)) if upper_starts.size else 0
        )
        upper_runs.append(upper_run / face_height)

        supported_rows = lower_coverage >= 0.04
        padded = np.concatenate(([False], supported_rows, [False]))
        transitions = np.diff(padded.astype(np.int8))
        run_starts = np.flatnonzero(transitions == 1)
        run_ends = np.flatnonzero(transitions == -1)
        longest_run = int(np.max(run_ends - run_starts)) if run_starts.size else 0
        side_runs.append(longest_run / face_height)

    if len(side_runs) != 2:
        return "Unclear"

    left_run, right_run = side_runs
    if left_run >= 0.35 and right_run >= 0.35:
        return "Long"
    if left_run >= 0.12 and right_run >= 0.12:
        return "Medium"
    if max(left_run, right_run) < 0.12 and min(upper_runs) >= 0.1:
        return "Short"
    return "Unclear"


def predict_age(image, face_net, age_net):
    rgb_image = np.asarray(image.convert("RGB"))
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    frame_height, frame_width = frame.shape[:2]
    detector_blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104, 117, 123), swapRB=False, crop=False
    )
    face_net.setInput(detector_blob)
    detections = face_net.forward()

    face_crops = []
    invalid_face_detected = False
    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2])
        if confidence < FACE_CONFIDENCE_THRESHOLD:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = map(
            float, detections[0, 0, index, 3:7]
        )
        if (
            not np.isfinite((x1_norm, y1_norm, x2_norm, y2_norm)).all()
            or x1_norm < 0
            or y1_norm < 0
            or x2_norm > 1
            or y2_norm > 1
            or x2_norm <= x1_norm
            or y2_norm <= y1_norm
        ):
            invalid_face_detected = True
            continue

        x1 = int(x1_norm * frame_width)
        y1 = int(y1_norm * frame_height)
        x2 = min(frame_width, int(x2_norm * frame_width))
        y2 = min(frame_height, int(y2_norm * frame_height))
        if x2 <= x1 or y2 <= y1:
            invalid_face_detected = True
            continue

        face = frame[
            max(0, y1 - 20):min(frame_height, y2 + 20),
            max(0, x1 - 20):min(frame_width, x2 + 20),
        ]
        if face.size:
            face_crops.append((face, (x1, y1, x2, y2), False))

    if (
        not face_crops
        and invalid_face_detected
        and max(frame_height, frame_width) <= 512
        and 0.8 <= frame_width / frame_height <= 1.25
    ):
        face_crops.append((frame, None, True))

    results = []
    for face, face_box, used_full_image in face_crops:
        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )
        age_net.setInput(face_blob)
        age_scores = age_net.forward()[0]
        if age_scores.size != len(AGE_RANGES):
            continue

        score_total = float(np.sum(age_scores))
        if not np.isfinite(score_total) or score_total <= 0:
            continue
        normalized_scores = age_scores / score_total
        age_index = int(np.argmax(normalized_scores))
        predicted_age_range = AGE_RANGES[age_index]
        age_score = float(normalized_scores[age_index])

        hair_length = estimate_hair_length(
            frame, face_box
        ) if face_box is not None else "Unclear"
        results.append(
            (
                predicted_age_range,
                age_score,
                hair_length,
                used_full_image,
                face_box,
            )
        )

    return results


@st.fragment(run_every=1)
def render_sign_sample_table(sign_session, letters):
    counts = sign_session.sample_counts()
    with sign_session.lock:
        sample_labels = tuple(sign_session.labels)

    with st.container(border=True):
        st.subheader("Collected sign samples")
        st.caption(
            "Each row is one landmark sample collected for the selected sign."
        )
        if not sample_labels:
            st.info("No sign samples have been collected yet.")
            return

        st.dataframe(
            [
                {"Letter": letter, "Samples": counts.get(letter, 0)}
                for letter in letters
                if counts.get(letter, 0)
            ],
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            [
                {"Sample": sample_number, "Letter": letter}
                for sample_number, letter in enumerate(sample_labels, start=1)
            ],
            hide_index=True,
            width="stretch",
            height=320,
        )


def main():
    st.title("Image, Traffic & Sign Detection")
    st.caption(
        "Face-based age results are broad estimates from a pretrained model, not exact ages."
    )

    with st.spinner("Loading face and age models..."):
        models = load_face_age_models()

    if models is None:
        st.warning(
            "Image-based age and hair analysis is unavailable. "
            "You can still record a visit using voluntarily provided information."
        )
        face_net = age_net = None
    else:
        face_net, age_net = models

    st.header("Image analysis")
    st.caption(
        "Gender, sex, and nationality are not inferred from images. "
        "Analysis runs automatically after uploading an image."
    )
    uploaded_image = st.file_uploader(
        "Upload one or more images",
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=True,
    )
    st.caption(
        "The image model estimates a broad age band, not an exact age. Its eight "
        "output bands are 0-2, 4-6, 8-12, 15-20, "
        "25-32, 38-43, 48-53, and 60-100. The model cannot distinguish exact ages "
        "within a band or estimate anyone over 100. Its score is not a calibrated "
        "probability and the model can misclassify faces. Use "
        "a voluntarily provided age when an exact record is required."
    )
    st.caption(
        "Possible-senior flags reuse the existing 60–100 face-age band and are not "
        "a separate or medically validated age assessment."
    )
    if uploaded_image:
        cached_analyses = st.session_state.setdefault("image_analyses_v8", {})
        for uploaded_file in uploaded_image:
            image_bytes = uploaded_file.getvalue()
            image_id = hashlib.sha256(image_bytes).hexdigest()
            try:
                with Image.open(io.BytesIO(image_bytes)) as opened_image:
                    image = opened_image.convert("RGB")
                st.image(
                    image,
                    caption=f"Uploaded image: {uploaded_file.name}",
                    width="stretch",
                )
            except (OSError, Image.UnidentifiedImageError) as exc:
                st.error(f"Could not open {uploaded_file.name}: {exc}")
                continue

            if face_net is None or age_net is None:
                st.info("Image analysis is unavailable until the face and age models load.")
                continue

            if image_id not in cached_analyses:
                try:
                    cached_analyses[image_id] = predict_age(image, face_net, age_net)
                except cv2.error as exc:
                    st.error(f"Could not analyze {uploaded_file.name}: {exc}")
                    continue

            predictions = cached_analyses[image_id]
            st.subheader(f"Estimates for {uploaded_file.name}")
            if predictions:
                for face_index, (
                    age_group,
                    age_score,
                    hair_length,
                    used_full_image,
                    face_box,
                ) in enumerate(
                    predictions, start=1
                ):
                    crop_note = (
                        " (estimated from the uploaded face crop)"
                        if used_full_image
                        else ""
                    )
                    st.write(
                        f"Face {face_index}: estimated age band {age_group} years "
                        f"(uncalibrated model score {age_score:.0%}){crop_note}; "
                        f"hair-length estimate: {hair_length}"
                    )
                    if age_group == "60-100":
                        st.info(
                            f"Face {face_index}: possible senior (the model's broad "
                            "60–100 age band; verify age directly)."
                        )
            else:
                st.info("No face was detected in this image.")

            hair_result_key = f"segmented_hair_{image_id}"
            if st.button(
                "Run long-hair identification model",
                key=f"identify_long_hair_{image_id}",
            ):
                usable_faces = [
                    (face_index, prediction[4])
                    for face_index, prediction in enumerate(predictions, start=1)
                    if prediction[4] is not None
                ]
                if not usable_faces:
                    st.info(
                        "A valid face box is needed. Try a clear image showing "
                        "the face and hair below the jaw."
                    )
                else:
                    try:
                        from ml_features import (
                            identify_long_hair,
                            load_hair_segmentation_model,
                        )

                        with st.spinner(
                            "Loading hair segmentation model and analyzing image..."
                        ):
                            torch, hair_processor, hair_model = (
                                load_hair_segmentation_model()
                            )
                            hair_results = [
                                (
                                    face_index,
                                    identify_long_hair(
                                        image,
                                        face_box,
                                        torch,
                                        hair_processor,
                                        hair_model,
                                    ),
                                )
                                for face_index, face_box in usable_faces
                            ]
                        st.session_state[hair_result_key] = hair_results
                    except (
                        ImportError,
                        OSError,
                        RuntimeError,
                        ValueError,
                        ConnectionError,
                        TimeoutError,
                    ) as exc:
                        st.error(f"Could not identify long hair: {exc}")

            if hair_result_key in st.session_state:
                for face_index, hair_result in st.session_state[hair_result_key]:
                    st.write(
                        f"Face {face_index}: segmentation-based result — "
                        f"**{hair_result}**."
                    )

    st.header("Voice age and emotion analysis")
    st.caption(
        "Experimental estimates from speech audio only; they are not a person's "
        "verified age or emotional state. Voice models are English-trained and "
        "may perform poorly on other languages, accents, or noisy recordings. "
        "Audio is analyzed locally in memory and is not uploaded or saved by this feature."
    )
    st.caption(
        "The voice-age and hair-segmentation models are restricted to non-commercial "
        "research/evaluation use by their published licenses. Upload a clear 1–30 "
        "second speech clip. WAV works directly; MP3 and M4A require PyAV to load "
        "successfully. First use downloads pretrained models."
    )
    voice_file = st.file_uploader(
        "Upload a voice recording",
        type=("wav", "mp3", "m4a"),
        accept_multiple_files=False,
        key="voice_analysis_upload",
    )
    if voice_file is not None:
        voice_bytes = voice_file.getvalue()
        voice_id = hashlib.sha256(voice_bytes).hexdigest()
        voice_result_key = f"voice_analysis_{voice_id}"
        if st.button("Analyze voice age and emotion", key="analyze_voice"):
            try:
                from ml_features import (
                    analyze_voice_age,
                    analyze_voice_emotion,
                    decode_audio,
                    load_voice_age_model,
                    load_voice_emotion_model,
                )

                with st.spinner(
                    "Loading voice models and analyzing audio. "
                    "First-time model downloads may take several minutes..."
                ):
                    waveform = decode_audio(voice_bytes)
                    age_torch, age_processor, age_model = load_voice_age_model()
                    emotion_torch, emotion_extractor, emotion_model = (
                        load_voice_emotion_model()
                    )
                    voice_age = analyze_voice_age(
                        waveform,
                        age_torch,
                        age_processor,
                        age_model,
                    )
                    emotion, emotion_score = analyze_voice_emotion(
                        waveform,
                        emotion_torch,
                        emotion_extractor,
                        emotion_model,
                    )
                st.session_state[voice_result_key] = (voice_age, emotion, emotion_score)
            except (
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
                ConnectionError,
                TimeoutError,
            ) as exc:
                st.error(f"Could not analyze this voice recording: {exc}")

        if voice_result_key in st.session_state:
            voice_age, emotion, emotion_score = st.session_state[voice_result_key]
            st.write(f"Approximate voice-based age: **about {voice_age} years**.")
            st.write(
                f"Speech emotion cue: **{emotion}** "
                f"(uncalibrated model score {emotion_score:.0%})."
            )

    st.caption(
        "Nationality is not inferred from a face or voice. If it is needed, enter "
        "it voluntarily here; it remains in this Streamlit session only."
    )
    st.text_input(
        "Nationality (optional, self-reported)",
        key="self_reported_nationality",
    )

    st.header("Traffic: car colors and people count")
    st.caption(
        "Blue cars are outlined in red; other cars are outlined in blue. "
        "Model confidence is shown on each detection and is not a guarantee of accuracy. "
        "The first inference downloads the YOLO model if it is not already cached."
    )
    traffic_images = st.file_uploader(
        "Upload traffic image(s)",
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=True,
        key="traffic_images",
    )
    if traffic_images:
        try:
            from traffic_detection import detect_traffic

            detector = load_traffic_detector()
        except (
            RuntimeError,
            OSError,
            ImportError,
            ConnectionError,
            TimeoutError,
        ) as exc:
            st.warning(f"Traffic detection is unavailable: {exc}")
        else:
            for traffic_file in traffic_images:
                try:
                    with Image.open(io.BytesIO(traffic_file.getvalue())) as traffic_image:
                        traffic_image = traffic_image.convert("RGB")
                    annotated, car_count, person_count, detections = detect_traffic(
                        traffic_image, detector
                    )
                    st.subheader(f"Traffic analysis: {traffic_file.name}")
                    st.image(
                        annotated,
                        caption="Detected blue cars: red boxes; other cars: blue boxes.",
                        width="stretch",
                    )
                    st.metric("Cars detected", car_count)
                    st.metric("People detected", person_count)
                    if detections:
                        st.dataframe(
                            [
                                {
                                    "Object": detection["type"].title(),
                                    "Color": detection.get("color", "—"),
                                    "Model confidence": (
                                        f"{detection['confidence']:.0%}"
                                    ),
                                }
                                for detection in detections
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("No cars or people detected in this image.")
                except (OSError, ValueError, cv2.error, RuntimeError) as exc:
                    st.error(f"Could not analyze {traffic_file.name}: {exc}")

    st.header("ASL static alphabet recognition")
    try:
        from sign_language import (
            SAMPLE_PATH,
            STATIC_ASL_LETTERS,
            SignLanguageSession,
            SignVideoProcessor,
            classify_image,
            is_recognition_time,
        )
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError as exc:
        st.warning(
            "Sign recognition needs the optional dependencies. Install them with "
            "`pip install -r requirements-sign-language.txt` in a Python 3.12 "
            f"environment. Details: {exc}"
        )
    else:
        try:
            sign_session = st.session_state.get("sign_language_session")
            if sign_session is None:
                sign_session = SignLanguageSession()
                st.session_state["sign_language_session"] = sign_session
        except (OSError, ValueError) as exc:
            st.error(f"Could not load saved sign samples/model: {exc}")
            sign_session = None

        if sign_session is not None:
            st.caption(
                "Starter support covers static ASL alphabet letters A-I and K-Y. "
                "J and Z require movement and are not included. Webcam samples and a "
                "local model are saved in this project folder."
            )
            counts = sign_session.sample_counts()
            collected_total = sum(counts.values())
            st.write(f"Collected training samples: **{collected_total}**")
            selected_letter = st.selectbox(
                "Letter to collect",
                STATIC_ASL_LETTERS,
                key="sign_training_label",
            )
            with sign_session.lock:
                sign_session.selected_label = selected_letter
                sign_session.collecting = st.toggle(
                    "Collect samples from webcam",
                    key="sign_collecting",
                    help="Hold the selected sign steady in the webcam view.",
                )

            webrtc_streamer(
                key="sign_language_webcam",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: SignVideoProcessor(sign_session),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            render_sign_sample_table(sign_session, STATIC_ASL_LETTERS)
            updated_counts = sign_session.sample_counts()

            save_col, train_col, delete_col = st.columns(3)
            if save_col.button("Save collected samples"):
                try:
                    sign_session.save_samples()
                    st.success(f"Training samples saved to {SAMPLE_PATH.name}.")
                except OSError as exc:
                    st.error(f"Could not save sign samples: {exc}")

            if train_col.button("Train sign recognizer"):
                try:
                    accuracy = sign_session.train()
                    st.success(
                        f"Model trained. Held-out sample accuracy: {accuracy:.0%}. "
                        "Accuracy on new signers may differ."
                    )
                except (OSError, ValueError) as exc:
                    st.error(f"Could not train sign recognizer: {exc}")

            selected_sample_count = updated_counts.get(selected_letter, 0)
            confirm_delete_key = "confirm_delete_sign_samples"
            if delete_col.button(
                f"Delete {selected_letter} samples",
                disabled=selected_sample_count == 0,
            ):
                st.session_state[confirm_delete_key] = selected_letter

            if st.session_state.get(confirm_delete_key) == selected_letter:
                st.warning(
                    f"Delete all {selected_sample_count} collected samples for "
                    f"sign {selected_letter}? This also removes the trained model."
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    f"Confirm delete {selected_letter}",
                    type="primary",
                ):
                    try:
                        deleted_count = sign_session.delete_samples(selected_letter)
                    except OSError as exc:
                        st.error(f"Could not delete sign samples: {exc}")
                    else:
                        st.toast(
                            f"Deleted {deleted_count} samples for {selected_letter}."
                        )
                        del st.session_state[confirm_delete_key]
                        st.rerun()
                if cancel_col.button("Cancel deletion"):
                    del st.session_state[confirm_delete_key]
                    st.rerun()

            delete_all_key = "confirm_delete_all_sign_samples"
            delete_all_col, _ = st.columns(2)
            if delete_all_col.button(
                "Delete all sign samples",
                disabled=collected_total == 0,
            ):
                st.session_state[delete_all_key] = True

            if st.session_state.get(delete_all_key):
                st.warning(
                    f"Delete all {collected_total} collected sign samples? "
                    "This also removes the trained model and cannot be undone."
                )
                confirm_all_col, cancel_all_col = st.columns(2)
                if confirm_all_col.button(
                    "Confirm delete all samples",
                    type="primary",
                ):
                    try:
                        deleted_count = sign_session.delete_all_samples()
                    except OSError as exc:
                        st.error(f"Could not delete all sign samples: {exc}")
                    else:
                        st.toast(f"Deleted all {deleted_count} sign samples.")
                        del st.session_state[delete_all_key]
                        st.rerun()
                if cancel_all_col.button("Cancel delete all"):
                    del st.session_state[delete_all_key]
                    st.rerun()

            st.subheader("Recognize a sign")
            if is_recognition_time():
                st.success("Recognition is available now (6 PM–10 PM local time).")
                sign_image_file = st.file_uploader(
                    "Upload an image of one hand making a static sign",
                    type=("jpg", "jpeg", "png"),
                    accept_multiple_files=False,
                    key="sign_image_upload",
                )
                if sign_image_file is not None:
                    try:
                        with Image.open(sign_image_file) as sign_image:
                            sign_image = sign_image.convert("RGB")
                        st.image(
                            sign_image,
                            caption="Sign image",
                            width="stretch",
                        )
                        if sign_session.model is None:
                            st.info("Train the recognizer before predicting a letter.")
                        else:
                            prediction = classify_image(sign_image, sign_session.model)
                            if prediction is None:
                                st.info("No hand detected. Try a clearer image.")
                            else:
                                predicted_letter, confidence = prediction
                                st.write(
                                    f"Predicted static sign: **{predicted_letter}** "
                                    f"({confidence:.0%} confidence)"
                                )
                    except (OSError, ValueError, cv2.error) as exc:
                        st.error(f"Could not recognize this sign image: {exc}")
            else:
                st.info(
                    "Sign recognition is available daily from 6 PM to 10 PM local time. "
                    "You can collect webcam samples and train the model outside those hours."
                )

if __name__ == "__main__":
    main()