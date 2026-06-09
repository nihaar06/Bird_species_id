import streamlit as st
import numpy as np
import librosa
import json
import os
import plotly.graph_objects as go
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# --- SET CONFIGURATION (Professional Theme & Wide Engine) ---
st.set_page_config(
    page_title="Bioacoustic Species ID System",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Sleek Dark/Light Adaptive Design with Micro-Animations) ---
st.markdown("""
    <style>
    /* Smooth transition rendering across all elements */
    * {
        transition: background-color 0.4s ease, transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Sleek gradient background premium panel header */
    .header-panel {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(22, 36, 71, 0.15);
        animation: slideDown 0.6s ease-out;
    }
    
    /* Interactive Card hover micro-animations */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        border-left: 5px solid #e43f5a;
        margin-bottom: 1rem;
    }
    
    html[data-theme="dark"] .metric-card {
        background: #1b1e23;
        border-left: 5px solid #00b4d8;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }

    @keyframes slideDown {
        0% { opacity: 0; transform: translateY(-15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- STEP 1: CACHED RESOURCE INITIALIZATION ---
@st.cache_resource
def load_prediction_assets():
    # 1. Parse your structured class mapping dictionary
    with open("src/model/bird_class_map.json", "r") as f:
        label_map = json.load(f)
        
    # Invert the mapping to decode class indices instantly back to true species codes
    index_to_species = {v: k for k, v in label_map.items()}
    
    # 2. Reconstruct the deep EfficientNetB0 structural framework shell
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(128, 313, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(len(label_map), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 3. Load the weights checkpoint into your network framework
    model.load_weights("src/model/bird_efficientnet_weights.weights.h5")
    
    return model, index_to_species

# Securely instantiate assets
model, index_to_species = load_prediction_assets()

# --- STEP 2: AUDIO PROCESSING PIPELINE (Matches your optimized training math) ---
def preprocess_audio_input(file_path, sr=32000, duration=5, top_db=45):
    # 1. Native waveform ingestion & loading
    audio, _ = librosa.load(file_path, sr=sr)
    
    # 2. Peak amplitude rescaling normalization
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
        
    # 3. Silence Trimming with your safe 45dB threshold to look for distant calls
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    
    # 4. Instant Vectorized Sliding Window Energy focus selection
    target_length = sr * duration
    if len(audio) > target_length:
        frame_length, hop_length = 2048, 512
        num_frames = (len(audio) - frame_length) // hop_length + 1
        
        # Build views to eliminate slow iterative loops
        shape = (num_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        audio_windows = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        energy = np.sum(audio_windows**2, axis=1)
        
        if len(energy) > 0:
            max_energy_idx = np.argmax(energy)
            center_sample = max_energy_idx * hop_length
            start = max(0, center_sample - (target_length // 2))
            end = start + target_length
            if end > len(audio):
                end = len(audio)
                start = max(0, end - target_length)
            audio = audio[start:end]
        else:
            audio = audio[:target_length]
    else:
        # Pad with trailing zero signals if the audio clip is too short
        audio = np.pad(audio, (0, target_length - len(audio)))
        
    # 5. Compute Log-Mel Spectrogram representation matrix
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 6. Corrective [0, 255] Min-Max scaling to avoid mass dying ReLUs
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val - min_val > 0:
        log_mel = ((log_mel - min_val) / (max_val - min_val)) * 255.0
    else:
        log_mel = np.zeros_like(log_mel)
        
    # 7. Triplicate shape dimensions into 3 channels for EfficientNet model parameters
    log_mel = np.expand_dims(log_mel, axis=-1)
    log_mel = np.repeat(log_mel, 3, axis=-1)
    
    # Return batch array wrapper shape (1, 128, 313, 3)
    return np.expand_dims(log_mel, axis=0)

# --- STEP 3: INTERACTIVE USER INTERFACE ---
st.markdown("""
    <div class="header-panel">
        <h1 style='margin:0; font-weight:700;'>🦅 Bioacoustic Species Identification System</h1>
        <p style='margin-top:0.5rem; margin-bottom:0; opacity:0.85;'>
            Deploying a 237-layer compound-scaled EfficientNetB0 architecture to track wild animal vocalizations.
        </p>
    </div>
""", unsafe_allow_html=True)

# Organize core screen layout into functional split columns
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("📁 Audio Ingestion Port")
    st.info("Supported formats: .ogg, .mp3, .wav. Files will be dynamically resampled to 32kHz and analyzed.")
    
    uploaded_file = st.file_uploader("Upload recording sample here", type=["ogg", "mp3", "wav"])
    
    if uploaded_file is not None:
        st.write("### Audio Preview")
        # Native web-audio playback wrapper node
        st.audio(uploaded_file, format='audio/ogg')

with right_col:
    st.subheader("📊 Analytics & Inference Engine")
    
    if uploaded_file is not None:
        with st.spinner("Processing bioacoustic signal matrix..."):
            try:
                # 1. Execute full audio preprocessing pipeline directly from memory byte stream
                processed_spectrogram = preprocess_audio_input(uploaded_file)
                
                # 2. Run model classification forward pass
                preds = model.predict(processed_spectrogram)[0]
                top_idx = np.argmax(preds)
                top_confidence = preds[top_idx] * 100
                top_species_code = index_to_species[top_idx]
                
                # 3. Render modern UI prediction report metric card container
                st.markdown(f"""
                    <div class="metric-card">
                        <span style="text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.85rem; opacity: 0.7; font-weight: 600;">
                            Primary Target Classification Match
                        </span>
                        <h2 style="margin: 0.5rem 0 0.2rem 0; font-size: 2.5rem; font-weight: 800;">{top_species_code}</h2>
                        <span style="font-size: 1.25rem; font-weight: 700; color: #00b4d8;">{top_confidence:.2f}% Confidence Score</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # 4. Generate beautifully styled interactive Plotly distribution chart
                st.write("#### Top Candidate Species Probabilities")
                
                # Sort the arrays to grab the top 5 match possibilities
                top_5_indices = np.argsort(preds)[::-1][:5][::-1] # Reverse twice to sort highest-at-top visually
                top_5_scores = [preds[i] * 100 for i in top_5_indices]
                top_5_names = [index_to_species[i] for i in top_5_indices]
                
                # Construct custom plotly layout frame block
                fig = go.Figure(go.Bar(
                    x=top_5_scores,
                    y=top_5_names,
                    orientation='h',
                    marker=dict(
                        color=top_5_scores,
                        colorscale='Blues',
                        line=dict(color='rgba(25s, 255, 255, 0)', width=1)
                    ),
                    hovertemplate='Species Code: %{y}<br>Confidence: %{x:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=260,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="Confidence (%)", showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
                    yaxis=dict(automargin=True)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            except Exception as e:
                st.error(f"Execution tracking exception failure notice: {str(e)}")
    else:
        st.write("Waiting for audio ingestion sequence to run prediction operations.")
