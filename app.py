import streamlit as st
import numpy as np
import json
import os
import tempfile
import plotly.graph_objects as go
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from src.preprocessing.audio_loader import load_audio_pipeline, load_audio, normalize_audio, remove_silence
from src.preprocessing.spectrogram import create_spectrogram
from species_lookup import get_species_info, common_name


# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Acoustic Specimen Index",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Field Notebook / Specimen Ledger
#
#   Background   #0a0e0f   near-black, slightly green-grey
#   Panel        #11161a   card surfaces
#   Hairline     #232b2c   borders, dividers
#   Paper text   #e8e6e1   primary text (warm off-white)
#   Muted text   #8d978f   secondary text
#   Sage         #7a9b8e   primary accent (foliage)
#   Amber        #c9a876   secondary accent (specimen tag / rank)
#
#   Display face : "Source Serif 4"  — specimen card titles
#   Mono face    : "JetBrains Mono"  — codes, metrics, ledger rows
#   Body face    : system sans
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:        #0a0e0f;
    --panel:     #11161a;
    --panel-alt: #14191d;
    --hairline:  #232b2c;
    --paper:     #e8e6e1;
    --muted:     #8d978f;
    --sage:      #7a9b8e;
    --sage-dim:  #4f6760;
    --amber:     #c9a876;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--paper);
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding: 3rem 3rem 3rem 3rem;
    max-width: 1180px;
}
::selection { background: var(--sage-dim); }

/* ---------------------------------------------------------- */
/* HEADER / MASTHEAD                                            */
/* ---------------------------------------------------------- */
.masthead {
    border-bottom: 1px solid var(--hairline);
    padding-top: 0.8rem;
    padding-bottom: 1.6rem;
    margin-bottom: 2.2rem;
    animation: fadeIn 0.5s ease-out;
}
.masthead-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.1;
    letter-spacing: -0.01em;
    color: var(--paper);
    margin-bottom: 0.5rem;
}
.masthead-sub {
    font-size: 0.98rem;
    color: var(--muted);
    max-width: 560px;
    line-height: 1.5;
    margin-bottom: 1.3rem;
}
.masthead-stats {
    display: flex;
    gap: 2.2rem;
    flex-wrap: wrap;
}
.stat-block { display: flex; flex-direction: column; gap: 0.15rem; }
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--sage);
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ---------------------------------------------------------- */
/* SECTION LABELS                                               */
/* ---------------------------------------------------------- */
.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-eyebrow::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--hairline);
}

/* ---------------------------------------------------------- */
/* UPLOAD PANEL                                                 */
/* ---------------------------------------------------------- */
.upload-shell {
    border: 1px dashed var(--hairline);
    border-radius: 10px;
    padding: 1.4rem 1.5rem 0.6rem 1.5rem;
    background: var(--panel);
    transition: border-color 0.25s ease, background 0.25s ease;
    margin-bottom: 1rem;
}
.upload-shell:hover {
    border-color: var(--sage-dim);
    background: var(--panel-alt);
}
div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* ---------------------------------------------------------- */
/* GENERIC PANEL                                                */
/* ---------------------------------------------------------- */
.panel {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    animation: fadeUp 0.4s ease-out both;
}

/* ---------------------------------------------------------- */
/* SPECIMEN CARD — primary prediction                           */
/* ---------------------------------------------------------- */
.specimen-card {
    background: linear-gradient(165deg, var(--panel) 0%, var(--panel-alt) 100%);
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--sage);
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.1rem;
    animation: fadeUp 0.45s ease-out both;
}
.specimen-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--amber);
    border: 1px solid var(--amber);
    border-radius: 99px;
    padding: 0.18rem 0.65rem;
    margin-bottom: 0.9rem;
}
.specimen-name {
    font-family: 'Source Serif 4', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--paper);
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.specimen-scientific {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted);
    margin-bottom: 1.1rem;
}
.specimen-meta {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.meta-item .meta-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.meta-item .meta-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: var(--paper);
    font-weight: 500;
}

/* confidence tick-ruler */
.ruler-wrap { margin-top: 0.4rem; }
.ruler-track {
    position: relative;
    height: 6px;
    background: var(--hairline);
    border-radius: 3px;
    overflow: hidden;
}
.ruler-fill {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: linear-gradient(90deg, var(--sage-dim), var(--sage));
    border-radius: 3px;
    transition: width 0.9s cubic-bezier(0.22, 1, 0.36, 1);
}
.ruler-ticks {
    display: flex;
    justify-content: space-between;
    margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}

/* status chip */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    border: 1px solid currentColor;
}
.status-high   { color: var(--sage); }
.status-medium { color: var(--amber); }
.status-low    { color: #c97a6a; }
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
}

/* ---------------------------------------------------------- */
/* LEDGER (Top-5)                                               */
/* ---------------------------------------------------------- */
.ledger-row {
    display: grid;
    grid-template-columns: 2.4rem 1fr 4.5rem;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--hairline);
}
.ledger-row:last-child { border-bottom: none; }
.ledger-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
}
.ledger-rank.r1 { color: var(--sage); font-weight: 600; }
.ledger-main .ledger-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--paper);
    margin-bottom: 0.35rem;
}
.ledger-main .ledger-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-left: 0.4rem;
}
.ledger-bar-track {
    height: 4px;
    background: var(--hairline);
    border-radius: 2px;
    overflow: hidden;
}
.ledger-bar-fill {
    height: 100%;
    background: var(--sage-dim);
    border-radius: 2px;
}
.ledger-row.r1 .ledger-bar-fill { background: var(--sage); }
.ledger-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--paper);
    text-align: right;
    font-weight: 500;
}
.ledger-row.r1 .ledger-pct { color: var(--sage); }

/* ---------------------------------------------------------- */
/* EMPTY STATE                                                  */
/* ---------------------------------------------------------- */
.empty-state {
    text-align: center;
    padding: 3.2rem 1rem;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.6;
}
.empty-state .empty-icon {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
    opacity: 0.6;
}

/* ---------------------------------------------------------- */
/* AUDIO PLAYER WRAP                                            */
/* ---------------------------------------------------------- */
.audio-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
}
.stAudio audio { width: 100%; }

/* ---------------------------------------------------------- */
/* TOGGLE / FOOTER                                              */
/* ---------------------------------------------------------- */
.footer-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--hairline);
}

/* ---------------------------------------------------------- */
/* ANIMATIONS                                                   */
/* ---------------------------------------------------------- */
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    .specimen-card, .panel, .masthead, .ruler-fill { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_application_assets():
    with open("src/model/bird_class_map.json", "r") as f:
        label_map = json.load(f)

    index_to_species = {int(v): k for k, v in label_map.items()}
    num_classes = len(label_map)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 313, 3)
    )
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(base_model.input, outputs)
    model.load_weights("src/model/bird_efficientnet_weights.weights.h5")

    return model, index_to_species


model, index_to_species = load_application_assets()


# ════════════════════════════════════════════════════════════════
# PREPROCESSING (unchanged from validated pipeline)
# ════════════════════════════════════════════════════════════════
def spec_to_tensor(spec):
    min_val, max_val = spec.min(), spec.max()
    if max_val - min_val > 0:
        spec = ((spec - min_val) / (max_val - min_val)) * 255.0
    else:
        spec = np.zeros_like(spec)
    spec = np.expand_dims(spec, axis=-1)
    spec = np.repeat(spec, 3, axis=-1)
    return np.expand_dims(spec.astype(np.float32), axis=0)


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".wav"
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def process_field_audio_single(file_path):
    audio, sr = load_audio_pipeline(file_path, sr=32000, top_db=45, duration=5)
    spec = create_spectrogram(audio, sr)
    return spec_to_tensor(spec)


def process_field_audio_multi(file_path, sr=32000, duration=5, top_db=45, max_windows=12):
    audio, _ = load_audio(file_path, sr=sr)
    audio = normalize_audio(audio)
    audio = remove_silence(audio, top_db=top_db)

    window_len = sr * duration
    tensors = []

    if len(audio) <= window_len:
        padded = np.pad(audio, (0, window_len - len(audio)))
        tensors.append(spec_to_tensor(create_spectrogram(padded, sr)))
    else:
        for start in range(0, len(audio), window_len):
            clip = audio[start:start + window_len]
            if len(clip) < window_len:
                clip = np.pad(clip, (0, window_len - len(clip)))
            tensors.append(spec_to_tensor(create_spectrogram(clip, sr)))
            if len(tensors) >= max_windows:
                break
    return tensors


def predict_multi_window(model, tensors):
    preds = [model.predict(t, verbose=0)[0] for t in tensors]
    return np.mean(preds, axis=0)


# ════════════════════════════════════════════════════════════════
# HELPERS — confidence status
# ════════════════════════════════════════════════════════════════
def confidence_status(pct):
    if pct >= 60:
        return "status-high", "Strong match"
    elif pct >= 30:
        return "status-medium", "Probable match"
    else:
        return "status-low", "Weak signal"


# ════════════════════════════════════════════════════════════════
# MASTHEAD
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Field Audio Classifier · Specimen 001</div>
    <div class="masthead-title">Acoustic Specimen Index</div>
    <div class="masthead-sub">
        Upload a field recording to identify the vocalizing species.
        Audio is segmented, converted to a log-mel spectrogram, and
        classified by an EfficientNetB0 transfer-learning model
        trained on bioacoustic recordings.
    </div>
    <div class="masthead-stats">
        <div class="stat-block">
            <span class="stat-value">52</span>
            <span class="stat-label">Species Indexed</span>
        </div>
        <div class="stat-block">
            <span class="stat-value">70.4%</span>
            <span class="stat-label">Validation Accuracy</span>
        </div>
        <div class="stat-block">
            <span class="stat-value">128 × 313</span>
            <span class="stat-label">Spectrogram Bins</span>
        </div>
        <div class="stat-block">
            <span class="stat-value">EfficientNetB0</span>
            <span class="stat-label">Model Backbone</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown('<div class="section-eyebrow">Upload Recording</div>', unsafe_allow_html=True)

    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "Drop a recording — .ogg, .mp3, or .wav",
        type=["ogg", "mp3", "wav"],
        label_visibility="visible",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    use_multi_window = st.toggle(
        "Scan full recording (multi-window)",
        value=True,
        help="Slides 5-second windows across the entire recording "
             "and averages predictions. Switch off to use only the "
             "single highest-energy segment."
    )

    if audio_file is not None:
        st.markdown('<div class="section-eyebrow" style="margin-top:1.4rem">Playback</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        audio_file.seek(0)
        st.audio(audio_file)
        size_kb = len(audio_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="audio-meta">
            <span>{audio_file.name}</span>
            <span>{size_kb:.1f} KB · 32 kHz target</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-eyebrow">Identification Result</div>', unsafe_allow_html=True)

    if audio_file is None:
        st.markdown("""
        <div class="panel">
            <div class="empty-state">
                <div class="empty-icon">🪶</div>
                No recording loaded.<br>
                Upload an audio file to run identification.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing waveform and extracting acoustic features…"):
            try:
                temp_path = save_uploaded_file(audio_file)
                try:
                    if use_multi_window:
                        tensors = process_field_audio_multi(temp_path)
                        raw_predictions = predict_multi_window(model, tensors)
                        window_note = f"{len(tensors)} window(s) scanned"
                    else:
                        tensor_input = process_field_audio_single(temp_path)
                        raw_predictions = model.predict(tensor_input, verbose=0)[0]
                        window_note = "single 5s segment"
                finally:
                    os.remove(temp_path)

                top_idx = int(np.argmax(raw_predictions))
                confidence = float(raw_predictions[top_idx]) * 100
                info = get_species_info(index_to_species[top_idx])

                status_class, status_label = confidence_status(confidence)

                # ── Specimen card ─────────────────────────────
                st.markdown(f"""
                <div class="specimen-card">
                    <div class="specimen-tag">Rank 1 · {window_note}</div>
                    <div class="specimen-name">{info['common']}</div>
                    <div class="specimen-scientific">{info['scientific']}</div>
                    <div class="specimen-meta">
                        <div class="meta-item">
                            <div class="meta-label">BirdCLEF Code</div>
                            <div class="meta-value">{info['code']}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Confidence</div>
                            <div class="meta-value">{confidence:.1f}%</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Status</div>
                            <div class="meta-value">
                                <span class="status-chip {status_class}">
                                    <span class="status-dot"></span>{status_label}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="ruler-wrap">
                        <div class="ruler-track">
                            <div class="ruler-fill" style="width:{min(confidence,100):.1f}%"></div>
                        </div>
                        <div class="ruler-ticks">
                            <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if confidence < 20:
                    st.markdown("""
                    <div class="panel" style="border-left:3px solid #c97a6a;">
                        <span style="color:#c97a6a; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">
                            ⚠ Low confidence — recording may contain overlapping calls,
                            background noise, or no clear vocalization.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Top-5 ledger ──────────────────────────────
                st.markdown('<div class="section-eyebrow" style="margin-top:1.4rem">Ranked Candidates</div>', unsafe_allow_html=True)

                top5_idx = np.argsort(raw_predictions)[::-1][:5]
                rows_html = '<div class="panel">'
                for rank, idx in enumerate(top5_idx, start=1):
                    pct = float(raw_predictions[idx]) * 100
                    sp = get_species_info(index_to_species[int(idx)])
                    rank_class = "r1" if rank == 1 else ""
                    rows_html += f"""
                    <div class="ledger-row {rank_class}">
                        <div class="ledger-rank {rank_class}">{rank:02d}</div>
                        <div class="ledger-main">
                            <div class="ledger-name">{sp['common']}<span class="ledger-code">{sp['code']}</span></div>
                            <div class="ledger-bar-track">
                                <div class="ledger-bar-fill" style="width:{min(pct,100):.1f}%"></div>
                            </div>
                        </div>
                        <div class="ledger-pct">{pct:.1f}%</div>
                    </div>
                    """
                rows_html += '</div>'
                st.markdown(rows_html, unsafe_allow_html=True)

                # ── Probability chart ─────────────────────────
                st.markdown('<div class="section-eyebrow" style="margin-top:1.4rem">Probability Distribution</div>', unsafe_allow_html=True)

                top5_names = [common_name(index_to_species[int(i)]) for i in top5_idx]
                top5_pcts = [float(raw_predictions[i]) * 100 for i in top5_idx]

                fig = go.Figure(go.Bar(
                    x=top5_pcts[::-1],
                    y=top5_names[::-1],
                    orientation='h',
                    marker=dict(
                        color=["#4f6760"] * 4 + ["#7a9b8e"],
                        line=dict(width=0),
                    ),
                    hovertemplate='%{y}<br><b>%{x:.2f}%</b><extra></extra>',
                ))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=220,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="JetBrains Mono, monospace", color="#8d978f", size=11),
                    xaxis=dict(
                        range=[0, 100],
                        showgrid=True,
                        gridcolor='#232b2c',
                        zeroline=False,
                        title=None,
                    ),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            except Exception as e:
                st.markdown(f"""
                <div class="panel" style="border-left:3px solid #c97a6a;">
                    <div class="section-eyebrow" style="margin-bottom:0.4rem;">Processing Error</div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--muted);">
                        {str(e)}
                    </span>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-note">
    MODEL: EfficientNetB0 (fine-tuned) · INPUT: 128×313×3 log-mel ·
    SAMPLE RATE: 32 kHz · SILENCE GATE: 45 dB
</div>
""", unsafe_allow_html=True)
