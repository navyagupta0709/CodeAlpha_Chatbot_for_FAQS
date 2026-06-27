import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
from collections import defaultdict

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI — Object Detection & Tracking",
    page_icon="👁️",
    layout="wide"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080b14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

.hero {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.hero h1 .acc { color: #f97316; }
.hero p { color: #64748b; font-size: 0.87rem; margin-top: 0.3rem; }
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.1);
    color: #f97316;
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 20px;
    padding: 0.18rem 0.75rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 0.7rem;
}

/* Panel */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.panel-title {
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.9rem;
}

/* Metric cards */
.metrics-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card {
    flex: 1; min-width: 100px;
    background: rgba(249,115,22,0.07);
    border: 1px solid rgba(249,115,22,0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f97316;
    line-height: 1;
}
.metric-lbl { color: #64748b; font-size: 0.72rem; margin-top: 0.2rem; }

/* Object tags */
.obj-tag {
    display: inline-block;
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.2);
    color: #fed7aa;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    margin: 0.15rem;
}

/* Progress legend */
.legend-item { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* Info box */
.info-box {
    background: rgba(14,165,233,0.07);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #7dd3fc;
    font-size: 0.82rem;
    line-height: 1.5;
}

.footer { text-align:center; color:#1e293b; font-size:0.72rem; margin-top:2rem; padding-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">YOLOV8 · REAL-TIME TRACKING · OPENCV</div>
    <h1>Vision<span class="acc">AI</span></h1>
    <p>Real-time object detection & tracking · Upload a video or image · Powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_size):
    return YOLO(f"yolov8{model_size}.pt")

# ─── Layout: Left Controls | Right Output ─────────────────────────────────────
col_ctrl, col_out = st.columns([1, 2.5])

with col_ctrl:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙️ Settings</div>', unsafe_allow_html=True)

    model_size = st.select_slider(
        "Model Size",
        options=["n", "s", "m"],
        value="n",
        help="n=Nano(fast), s=Small(balanced), m=Medium(accurate)"
    )
    size_label = {"n": "Nano — Fastest", "s": "Small — Balanced", "m": "Medium — Accurate"}
    st.caption(f"Selected: {size_label[model_size]}")

    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.5, 0.05)
    enable_tracking = st.toggle("Enable Object Tracking", value=True)
    show_labels = st.toggle("Show Labels", value=True)
    show_conf = st.toggle("Show Confidence", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📂 Input Source</div>', unsafe_allow_html=True)
    input_mode = st.radio("", ["🖼️ Image", "🎬 Video"], horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        ℹ️ <strong>YOLOv8</strong> detects 80 COCO classes: people, vehicles, animals, everyday objects and more.<br><br>
        Tracking assigns persistent IDs to objects across video frames using ByteTrack.
    </div>
    """, unsafe_allow_html=True)

with col_out:
    model = load_model(model_size)

    # ─── COCO color palette ───────────────────────────────────────────────────
    np.random.seed(42)
    COLORS = np.random.randint(50, 230, size=(80, 3), dtype=np.uint8)

    def process_image(img_array, conf, iou):
        results = model(img_array, conf=conf, iou=iou, verbose=False)[0]
        annotated = img_array.copy()
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = tuple(int(c) for c in COLORS[cls_id % 80])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                display_text = f"{label}"
                if show_conf:
                    display_text += f" {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, display_text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detections.append({"label": label, "confidence": confidence,
                                "bbox": (x1, y1, x2 - x1, y2 - y1)})

        return annotated, detections

    def process_video(video_path, conf, iou, tracking):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0, text="Processing video...")
        frame_count = 0
        all_labels = defaultdict(int)
        track_ids_seen = set()
        processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            if tracking:
                results = model.track(frame, conf=conf, iou=iou, persist=True, verbose=False)[0]
            else:
                results = model(frame, conf=conf, iou=iou, verbose=False)[0]

            processing_times.append(time.time() - t0)
            annotated = frame.copy()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = tuple(int(c) for c in COLORS[cls_id % 80])
                track_id = None

                if tracking and box.id is not None:
                    track_id = int(box.id[0])
                    track_ids_seen.add(track_id)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                all_labels[label] += 1

                if show_labels:
                    text = label
                    if show_conf:
                        text += f" {confidence:.0%}"
                    if track_id is not None:
                        text += f" #{track_id}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(annotated)
            frame_count += 1
            progress.progress(frame_count / max(total_frames, 1),
                              text=f"Processing frame {frame_count}/{total_frames}...")

        cap.release()
        out.release()
        avg_fps = 1 / np.mean(processing_times) if processing_times else 0

        return out_path, dict(all_labels), len(track_ids_seen), avg_fps, frame_count

    # ─── Image Mode ───────────────────────────────────────────────────────────
    if "🖼️ Image" in input_mode:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

        if uploaded_img:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with st.spinner("Running YOLOv8 detection..."):
                annotated_rgb, detections = process_image(img_rgb, conf_threshold, iou_threshold)

            st.image(annotated_rgb, caption="Detected Objects", use_container_width=True)

            if detections:
                label_counts = defaultdict(int)
                for d in detections:
                    label_counts[d["label"]] += 1

                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-card">
                        <div class="metric-val">{len(detections)}</div>
                        <div class="metric-lbl">Objects Found</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{len(label_counts)}</div>
                        <div class="metric-lbl">Unique Classes</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{max(d['confidence'] for d in detections):.0%}</div>
                        <div class="metric-lbl">Best Confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                tags_html = "".join(
                    f'<span class="obj-tag">{lbl} ×{cnt}</span>'
                    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
                )
                st.markdown(f'<div style="margin-bottom:1rem">{tags_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No objects detected. Try lowering the confidence threshold.")

    # ─── Video Mode ───────────────────────────────────────────────────────────
    else:
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_vid:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()

            st.video(tfile.name)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀 Run Detection & Tracking", type="primary", use_container_width=True):
                out_path, label_counts, unique_ids, avg_fps, total_frames = process_video(
                    tfile.name, conf_threshold, iou_threshold, enable_tracking
                )

                st.success("✅ Processing complete!")

                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-card">
                        <div class="metric-val">{sum(label_counts.values())}</div>
                        <div class="metric-lbl">Total Detections</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{len(label_counts)}</div>
                        <div class="metric-lbl">Unique Classes</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{unique_ids}</div>
                        <div class="metric-lbl">Tracked IDs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{avg_fps:.1f}</div>
                        <div class="metric-lbl">FPS (avg)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{total_frames}</div>
                        <div class="metric-lbl">Frames</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                tags_html = "".join(
                    f'<span class="obj-tag">{lbl} ×{cnt}</span>'
                    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
                )
                st.markdown(f'<div style="margin-bottom:1rem">{tags_html}</div>', unsafe_allow_html=True)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=f,
                        file_name="detected_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

                os.unlink(tfile.name)
                os.unlink(out_path)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by Navya · CodeAlpha AI Internship · Task 4 — Object Detection & Tracking · YOLOv8 + OpenCV
</div>
""", unsafe_allow_html=True)
