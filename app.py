import streamlit as st
import numpy as np
import librosa
import json
import os
import plotly.graph_objects as go
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# --- DESIGN SETTINGS (Sleek Theme & Wide Mode) ---
st.set_page_config(
    page_title="Bioacoustic Species ID Portal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN STYLING INTERFACE (Custom Transitions & Micro-Animations) ---
st.markdown("""
    <style>
    /* Global layout adjustments with smooth variable transitions */
    * {
        transition: background-color 0.3s ease, transform 0.2s ease-out;
    }
    
    /* Premium Title Card Jumbotron */
    .hero-container {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        animation: fadeInDown 0.6s ease-out;
    }
    
    /* Interactive Result Panels with subtle floating hover elevations */
    .metric-panel {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border-top: 5px solid #2b6cb0;
        animation: fadeInUp 0.5s ease-out;
    }
    
    html[data-theme="dark"] .metric-panel {
        background: #1a202c;
        border-top: 5px solid #63b3ed;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-panel:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    }
    
    /* Micro Animations Keyframes */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- STEP 1: LIFECYCLE MEMORY MANAGEMENT (Cached Network Loader) ---
@st.cache_resource
def load_application_assets():
    # 1. Parse your attached label map configuration file
    with open("src/model/bird_class_map.json", "r") as f:
        label_map = json.load(f)
        
    # Invert to map neural output indices back to actual string species names
    index_to_species = {v: k for k, v in label_map.items()}
    
    # 2. Reconstruct the deep compound-scaled EfficientNetB0 backbone structure
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(128, 313, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(len(label_map), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 3. Pull weights out of your lightweight checkpoint binary
    model.load_weights("src/model/bird_efficientnet_weights.weights.h5")
    
    return model, index_to_species

# Fetch compilation assets safely
model, index_to_species = load_application_assets()

# --- STEP 2: FIELD PRODUCTION PROCESSING PIPELINE (100% Signal Match) ---
def process_field_audio(file_path, sr=32000, duration=5, top_db=45):
    # Load raw sound clip signals natively
    audio, _ = librosa.load(file_path, sr=sr)
    
    # Rescale peak waveform variance amplitudes
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
        
    # Apply your optimized 45dB silence dropping cut
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    
    # Apply your lightning-fast Vectorized Local Energy focus window locator
    target_length = sr * duration
    if len(audio) > target_length:
        frame_length, hop_length = 2048, 512
        num_frames = (len(audio) - frame_length) // hop_length + 1
        
        # Stride matrix view mappings to replace slow python processing loops
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
        # Pad shorter sound cues safely with silent trailing zeroes
        audio = np.pad(audio, (0, target_length - len(audio)))
        
    # Generate the identical 128-bin Mel Spectrogram representation matrix
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Corrective Min-Max scaling mapped right to [0, 255] for EfficientNet expectations
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val - min_val > 0:
        log_mel = ((log_mel - min_val) / (max_val - min_val)) * 255.0
    else:
        log_mel = np.zeros_like(log_mel)
        
    # Triplicate matrix planes into 3 matching data channels (RGB format emulation)
    log_mel = np.expand_dims(log_mel, axis=-1)
    log_mel = np.repeat(log_mel, 3, axis=-1)
    
    # Return batch array wrapper shape (1, 128, 313, 3)
    return np.expand_dims(log_mel, axis=0)

# --- STEP 3: SCREEN INTERFACE LAYOUT GRAPHICS ---
st.markdown("""
    <div class="hero-container">
        <h1 style='margin:0; font-weight:800; font-size:2.8rem; letter-spacing:-1px;'>🦅 Bioacoustic Species Identification System</h1>
        <p style='margin-top:0.6rem; margin-bottom:0; font-size:1.1rem; opacity:0.85; font-weight:400;'>
            Deep Acoustic Pattern Analysis Engine leveraging a 70.56% Accuracy Pre-trained EfficientNetB0 Core.
        </p>
    </div>
""", unsafe_allow_html=True)

# Instantiate wide visual layouts
col_panel, col_charts = st.columns([1, 1.2], gap="large")

with col_panel:
    st.markdown("### 📥 Audio Sample Upload")
    st.caption("Drop field audio tracking recordings (.ogg, .mp3, or .wav) to isolate energy windows and run classification passes.")
    
    audio_file = st.file_uploader("", type=["ogg", "mp3", "wav"], label_visibility="collapsed")
    
    if audio_file is not None:
        st.markdown("### 🎧 Sound Wave Player")
        st.audio(audio_file, format='audio/ogg')

with col_charts:
    st.markdown("### 📊 Classification Output Analytics")
    
    if audio_file is not None:
        # Utilize the modern sleek spinner indicator 
        with st.spinner("Decoding audio tensors and evaluating frequency metrics..."):
            try:
                # 1. Execute full processing matrix pipeline directly out of uploaded memory stream
                tensor_input = process_field_audio(audio_file)
                
                # 2. Run model feedforward inference
                raw_predictions = model.predict(tensor_input)[0]
                top_class_idx = np.argmax(raw_predictions)
                target_confidence = raw_predictions[top_class_idx] * 100
                target_species = index_to_species[top_class_idx]
                
                # 3. Output beautiful premium CSS metric panel
                st.markdown(f"""
                    <div class="metric-panel">
                        <span style="text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.8rem; opacity: 0.6; font-weight: 700; color: #4a5568;">
                            Primary Species Code Identified
                        </span>
                        <h2 style="margin: 0.4rem 0 0.2rem 0; font-size: 3rem; font-weight: 900; letter-spacing: -0.5px;">
                            {target_species}
                        </h2>
                        <span style="font-size: 1.3rem; font-weight: 700; color: #3182ce;">
                            {target_confidence:.2f}% Match Probability
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.write("#### Top 5 Candidate Class Likelihoods")
                
                # Sort arrays cleanly to grab top 5 match possibilities
                top_5_idxs = np.argsort(raw_predictions)[::-1][:5][::-1]
                top_5_probabilities = [raw_predictions[i] * 100 for i in top_5_idxs]
                top_5_species_labels = [index_to_species[i] for i in top_5_idxs]
                
                # 4. Generate high-contrast horizontal probability tracking chart via Plotly
                fig = go.Figure(go.Bar(
                    x=top_5_probabilities,
                    y=top_5_species_labels,
                    orientation='h',
                    marker=dict(
                        color=top_5_probabilities,
                        colorscale='Blues',
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    hovertemplate='Species: %{y}<br>Probability: %{x:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=5, b=5),
                    height=240,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="Confidence Margin (%)", showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
                    yaxis=dict(automargin=True)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            except Exception as e:
                st.error(f"Runtime extraction processing collapse: {str(e)}")
    else:
        st.info("System engine standing by. Upload a bioacoustic file to initialize deep learning analysis.")
