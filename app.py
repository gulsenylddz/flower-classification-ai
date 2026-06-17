import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Flora · AI",
    page_icon="🌸",
    layout="wide"
)


def inject_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_css("""
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #080c12 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #080c12 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] strong {
    color: #8b9ab0 !important;
}

.block-container {
    padding-top: 2.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1400px !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px dashed rgba(236,72,153,0.3) !important;
    border-radius: 18px !important;
    padding: 28px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(236,72,153,0.6) !important;
}
[data-testid="stFileUploader"] section > div > span {
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] section > div > small {
    color: #4b5563 !important;
}
[data-testid="stFileUploader"] button {
    background-color: rgba(236,72,153,0.12) !important;
    border: 1px solid rgba(236,72,153,0.3) !important;
    color: #f9a8d4 !important;
    border-radius: 10px !important;
}

[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

[data-testid="stSpinner"] * { color: #f9a8d4 !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
""")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sidebar_logo():
    st.markdown(
        '<p style="font-family:\'Playfair Display\',serif;font-size:26px;'
        'color:#f9a8d4;letter-spacing:-0.5px;margin-bottom:2px">Flora · AI</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:11px;text-transform:uppercase;letter-spacing:3px;'
        'color:#374151;margin-top:0">Botanical Intelligence</p>',
        unsafe_allow_html=True,
    )


def sidebar_stat(label, value):
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border-radius:10px;'
        f'padding:10px 14px;margin:6px 0;border:1px solid rgba(255,255,255,0.05)">'
        f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;'
        f'color:#4b5563">{label}</div>'
        f'<div style="font-size:18px;font-weight:500;color:#e2e8f0;margin-top:2px">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;'
        f'color:#4b5563;margin-bottom:12px;margin-top:8px;display:flex;align-items:center;gap:10px">'
        f'{text}'
        f'<div style="flex:1;height:1px;background:rgba(255,255,255,0.05)"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Pollen Risk Engine ────────────────────────────────────────────────────────
# Base biological pollen production data (0-100 scale)
# Sources: allergy research literature on airborne pollen counts per species
_pollen_base = {
    "astilbe":           15,   # insect-pollinated, sticky pollen
    "bellflower":        40,
    "black_eyed_susan":  65,
    "calendula":         45,
    "california_poppy":  50,
    "carnation":         20,   # hybrid cultivars shed little
    "common_daisy":      70,
    "coreopsis":         55,
    "dandelion":         92,   # wind-dispersed, prolific
    "iris":              25,   # heavy pollen, not airborne
    "rose":              18,   # heavy waxy pollen
    "sunflower":         68,
    "tulip":             38,
    "water_lily":        12,   # aquatic, minimal airborne exposure
}

# Modifiers that nudge the score based on contextual factors
_season_modifier = {
    "spring":  +8,
    "summer":  +5,
    "autumn":  -5,
    "winter": -12,
}


def compute_pollen_score(
    class_name: str,
    confidence: float,
    season: str = "summer",
) -> dict:
    """
    Returns a dict with:
        score       : 0-100 float
        tier        : "Low" | "Medium" | "High" | "Very High"
        color       : hex
        bg          : rgba string
        border      : rgba string
        title       : display title
        body        : advisory text
        breakdown   : list of (label, value, delta) tuples for the UI gauge
    """
    base = _pollen_base.get(class_name, 50)
    season_mod = _season_modifier.get(season, 0)

    # Confidence amplifies certainty — higher confidence → score closer to true base
    # Low confidence → regress toward the population mean (50)
    conf_weight = confidence / 100.0
    regressed = base * conf_weight + 50 * (1 - conf_weight)
    score = float(np.clip(regressed + season_mod, 0, 100))

    # Tier thresholds
    if score < 25:
        tier, color, bg, border = (
            "Low",
            "#4ade80",
            "rgba(74,222,128,0.07)",
            "rgba(74,222,128,0.2)",
        )
        title = "Low Pollen Risk"
        body = (
            "This flower produces minimal airborne pollen and poses little to no "
            "allergy risk for most people. Safe for most environments."
        )
    elif score < 50:
        tier, color, bg, border = (
            "Medium",
            "#fbbf24",
            "rgba(251,191,36,0.07)",
            "rgba(251,191,36,0.2)",
        )
        title = "Moderate Pollen Risk"
        body = (
            "May trigger mild symptoms in sensitive individuals. "
            "Consider limiting prolonged close exposure, especially during peak bloom."
        )
    elif score < 75:
        tier, color, bg, border = (
            "High",
            "#f87171",
            "rgba(248,113,113,0.07)",
            "rgba(248,113,113,0.2)",
        )
        title = "High Pollen Risk"
        body = (
            "A significant allergy trigger. People with hay fever or asthma "
            "should take precautions and consider antihistamines before exposure."
        )
    else:
        tier, color, bg, border = (
            "Very High",
            "#ef4444",
            "rgba(239,68,68,0.07)",
            "rgba(239,68,68,0.2)",
        )
        title = "Very High Pollen Risk"
        body = (
            "Releases exceptionally high volumes of airborne pollen. "
            "Allergy sufferers should avoid close contact and keep windows closed on windy days."
        )

    breakdown = [
        ("Species baseline", int(base), None),
        ("Season modifier", season_mod, season_mod >= 0),
        ("Confidence weight", round(conf_weight, 2), None),
        ("Final score", round(score, 1), None),
    ]

    return dict(
        score=score,
        tier=tier,
        color=color,
        bg=bg,
        border=border,
        title=title,
        body=body,
        breakdown=breakdown,
    )


# ── Pollen Gauge Widget ───────────────────────────────────────────────────────
def render_pollen_gauge(risk: dict):
    score     = risk["score"]
    color     = risk["color"]
    tier      = risk["tier"]
    title_str = risk["title"]

    # Arc SVG gauge (semicircle, 0-100)
    # Semicircle path: cx=110,cy=110 r=80 → circumference of semi = π*r ≈ 251.3
    SEMI_CIRC = 251.3
    filled    = (score / 100) * SEMI_CIRC

    # Pick track gradient stops
    gradient_stops = (
        '<stop offset="0%"   stop-color="#4ade80"/>'
        '<stop offset="33%"  stop-color="#fbbf24"/>'
        '<stop offset="66%"  stop-color="#f87171"/>'
        '<stop offset="100%" stop-color="#ef4444"/>'
    )

    # Breakdown rows HTML
    breakdown_rows = ""
    for label, val, positive in risk["breakdown"]:
        if positive is True:
            indicator = '<span style="color:#4ade80;font-size:10px">▲</span>'
        elif positive is False:
            indicator = '<span style="color:#4ade80;font-size:10px">▼</span>'
        else:
            indicator = ""
        breakdown_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)">'
            f'<span style="font-size:11px;color:#4b5563">{label}</span>'
            f'<span style="font-size:12px;color:#94a3b8">{val} {indicator}</span>'
            f'</div>'
        )

    gauge_html = f"""
<div style="background:linear-gradient(145deg,#0f1623,#131c2b);
            border:1px solid rgba(255,255,255,0.07);
            border-radius:18px;padding:20px 22px;margin-bottom:18px">

  <!-- SVG arc gauge -->
  <div style="display:flex;justify-content:center;margin-bottom:4px">
    <svg viewBox="0 0 220 130" width="220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          {gradient_stops}
        </linearGradient>
      </defs>
      <!-- Track -->
      <path d="M 30 110 A 80 80 0 0 1 190 110"
            fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="14"
            stroke-linecap="round"/>
      <!-- Fill — stroke-dasharray trick on a semi arc -->
      <path d="M 30 110 A 80 80 0 0 1 190 110"
            fill="none" stroke="url(#arcGrad)" stroke-width="14"
            stroke-linecap="round"
            stroke-dasharray="{filled:.1f} {SEMI_CIRC:.1f}"
            stroke-dashoffset="0"/>
      <!-- Center label -->
      <text x="110" y="98" text-anchor="middle"
            font-family="'DM Sans',sans-serif"
            font-size="26" font-weight="500" fill="{color}">{score:.0f}</text>
      <text x="110" y="114" text-anchor="middle"
            font-family="'DM Sans',sans-serif"
            font-size="10" fill="#4b5563" letter-spacing="1">/ 100</text>
      <!-- Tick labels -->
      <text x="26"  y="126" text-anchor="middle" font-size="9" fill="#374151" font-family="'DM Sans',sans-serif">0</text>
      <text x="194" y="126" text-anchor="middle" font-size="9" fill="#374151" font-family="'DM Sans',sans-serif">100</text>
    </svg>
  </div>

  <!-- Tier badge -->
  <div style="text-align:center;margin-bottom:16px">
    <span style="display:inline-block;background:{risk['bg']};border:1px solid {risk['border']};
                 color:{color};padding:4px 16px;border-radius:100px;font-size:12px;font-weight:500">
      {tier}
    </span>
  </div>

  <!-- Score breakdown -->
  <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;
              color:#4b5563;margin-bottom:8px">Score breakdown</div>
  {breakdown_rows}
</div>
"""
    st.markdown(gauge_html, unsafe_allow_html=True)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: int,
) -> np.ndarray:
    """
    Returns a [0,1] float32 heatmap (H×W) using Grad-CAM on the last conv layer.
    Works with any Keras model that has at least one Conv2D layer.
    """
    # Find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv is None:
        return None

    # Build grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)            # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # (c,)

    conv_outputs = conv_outputs[0]                        # (h, w, c)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on the original PIL image.
    Returns a PIL Image.
    """
    w, h = original_img.size
    heat = cv2.resize(heatmap, (w, h))
    cmap = cm.get_cmap(colormap)
    colored = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)
    orig_np = np.array(original_img.resize((w, h))).astype(np.uint8)
    blended = (orig_np * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(blended)


# ── Model & Data ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/flower_model.h5")


model = load_model()

class_names = [
    "astilbe", "bellflower", "black_eyed_susan", "calendula",
    "california_poppy", "carnation", "common_daisy", "coreopsis",
    "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily",
]

rank_icons = ["①", "②", "③", "④", "⑤"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_logo()
    st.markdown("---")
    sidebar_stat("Species Library", "14 Flowers")
    sidebar_stat("Input Resolution", "224 × 224")
    sidebar_stat("Pollen Tiers", "4 Levels")

    st.markdown("---")

    # Season selector influences pollen score
    st.markdown(
        '<p style="font-size:11px;text-transform:uppercase;letter-spacing:2px;'
        'color:#4b5563;margin-bottom:6px">Current Season</p>',
        unsafe_allow_html=True,
    )
    season = st.selectbox(
        "season",
        ["spring", "summer", "autumn", "winter"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown(
        '<p style="font-size:11px;color:#374151;margin-top:12px">'
        "Upload a photo · Get instant species ID · Dynamic pollen risk + AI heatmap"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="margin-bottom:40px">'

    '<div style="display:inline-flex;align-items:center;gap:8px;font-size:11px;'
    'text-transform:uppercase;letter-spacing:3px;color:#f9a8d4;'
    'background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.2);'
    'padding:5px 14px;border-radius:100px;margin-bottom:18px">'
    '<span style="width:6px;height:6px;background:#ec4899;border-radius:50%;'
    'display:inline-block"></span>'
    "Deep Learning &middot; Botanical Analysis"
    "</div>"

    '<div style="font-family:\'Playfair Display\',serif;font-size:58px;font-weight:700;'
    'color:#f1f5f9;line-height:1.05;letter-spacing:-2px;margin-bottom:12px">'
    "Identify any<br>"
    '<span style="font-style:italic;background:linear-gradient(135deg,#f9a8d4,#c084fc);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
    "flower instantly"
    "</span>"
    "</div>"

    '<div style="font-size:15px;color:#64748b;font-weight:300;max-width:520px">'
    "Upload a photo and let AI classify the species, show what it sees via a Grad-CAM "
    "heatmap, and compute a dynamic pollen allergy score."
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader(
        "Drop your flower image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
    if not uploaded_file:
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;height:320px;text-align:center;gap:14px">'
            '<div style="font-size:52px;filter:grayscale(1) brightness(0.3)">🌸</div>'
            '<div style="font-size:14px;color:#1e293b;font-style:italic">'
            "Your analysis will appear here"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing botanical features…"):
            img       = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            prediction   = model.predict(img_array)
            pred_index   = int(np.argmax(prediction))
            predicted_class = class_names[pred_index]
            confidence   = float(np.max(prediction) * 100)

            # Dynamic pollen score
            risk = compute_pollen_score(predicted_class, confidence, season)

            # Grad-CAM
            heatmap = make_gradcam_heatmap(model, img_array, pred_index)

        display_name = predicted_class.replace("_", " ").title()
        risk_color   = risk["color"]
        conf_pct     = int(confidence)

        # ── Result card ──────────────────────────────────────────────────────
        card_style = (
            "background:linear-gradient(145deg,#0f1623,#131c2b);"
            "border:1px solid rgba(255,255,255,0.07);"
            "border-radius:22px;padding:28px;margin-bottom:18px;position:relative"
        )
        badge_style = (
            "display:inline-block;font-size:10px;text-transform:uppercase;"
            "letter-spacing:2.5px;color:#f9a8d4;"
            "background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.18);"
            "padding:4px 12px;border-radius:100px;margin-bottom:18px"
        )
        stat_box = (
            "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);"
            "border-radius:14px;padding:14px 16px"
        )
        stat_lbl = "font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#4b5563;margin-bottom:4px"
        stat_val = "font-size:26px;font-weight:500;color:#f1f5f9;letter-spacing:-0.5px"
        bar_wrap = "background:rgba(255,255,255,0.05);border-radius:100px;height:5px;overflow:hidden;margin-top:8px"
        bar_fill = f"height:100%;border-radius:100px;background:linear-gradient(90deg,#ec4899,#c084fc);width:{conf_pct}%"

        st.markdown(
            f'<div style="{card_style}">'
            f'<div style="{badge_style}">&#10022; AI Result</div>'
            f'<div style="font-family:\'Playfair Display\',serif;font-size:42px;font-weight:700;'
            f'color:#f1f5f9;line-height:1;letter-spacing:-1px;margin-bottom:6px">{display_name}</div>'
            f'<div style="font-size:11px;color:#4b5563;text-transform:uppercase;'
            f'letter-spacing:2px;margin-bottom:24px">Identified species</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
            f'<div style="{stat_box}">'
            f'<div style="{stat_lbl}">Confidence</div>'
            f'<div style="{stat_val}">{confidence:.1f}<span style="font-size:13px;color:#64748b">%</span></div>'
            f'<div style="{bar_wrap}"><div style="{bar_fill}"></div></div>'
            f'</div>'
            f'<div style="{stat_box}">'
            f'<div style="{stat_lbl}">Pollen Score</div>'
            f'<div style="font-size:20px;font-weight:600;color:{risk_color};margin-top:6px">'
            f'{risk["score"]:.0f}<span style="font-size:12px;color:#4b5563"> / 100</span></div>'
            f'<div style="font-size:11px;color:{risk_color};margin-top:4px">{risk["tier"]}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Top 5 ────────────────────────────────────────────────────────────
        section_header("Top Predictions")

        top_indices = prediction[0].argsort()[-5:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_scores  = [float(prediction[0][i] * 100) for i in top_indices]

        panel_style = (
            "background:linear-gradient(145deg,#0f1623,#131c2b);"
            "border:1px solid rgba(255,255,255,0.07);"
            "border-radius:18px;padding:18px 22px;margin-bottom:18px"
        )
        rows = ""
        for i in range(len(top_classes)):
            nm = top_classes[i].replace("_", " ")
            sc = top_scores[i]
            bw = int(min(100, sc))
            rows += (
                '<div style="display:flex;align-items:center;gap:12px;padding:9px 0;'
                'border-bottom:1px solid rgba(255,255,255,0.04)">'
                f'<span style="font-size:11px;color:#374151;width:20px;text-align:center">{rank_icons[i]}</span>'
                f'<span style="font-size:13px;color:#cbd5e1;flex:1;text-transform:capitalize">{nm}</span>'
                f'<div style="width:80px;height:4px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden">'
                f'<div style="height:100%;width:{bw}%;background:linear-gradient(90deg,#ec4899,#c084fc);border-radius:100px"></div>'
                f'</div>'
                f'<span style="font-size:12px;color:#64748b;width:46px;text-align:right">{sc:.1f}%</span>'
                "</div>"
            )

        st.markdown(f'<div style="{panel_style}">{rows}</div>', unsafe_allow_html=True)

        # ── Dynamic Pollen Gauge ──────────────────────────────────────────────
        section_header("Pollen Risk Score")
        render_pollen_gauge(risk)

        # ── Allergy advisory ─────────────────────────────────────────────────
        section_header("Allergy Advisory")

        st.markdown(
            f'<div style="background:{risk["bg"]};border:1px solid {risk["border"]};'
            f'border-radius:14px;padding:16px 20px;font-size:13px;'
            f'color:#94a3b8;line-height:1.7;margin-bottom:14px">'
            f'<strong style="color:{risk_color}">{risk["title"]}</strong><br>{risk["body"]}'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Grad-CAM section (full width below) ──────────────────────────────────────
if uploaded_file and heatmap is not None:
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    section_header("Grad-CAM · What the Model Sees")

    cam_col1, cam_col2, cam_col3 = st.columns([1, 1, 1], gap="large")

    with cam_col1:
        st.markdown(
            '<p style="font-size:11px;text-transform:uppercase;letter-spacing:2px;'
            'color:#4b5563;margin-bottom:8px">Original</p>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with cam_col2:
        st.markdown(
            '<p style="font-size:11px;text-transform:uppercase;letter-spacing:2px;'
            'color:#4b5563;margin-bottom:8px">Activation Map</p>',
            unsafe_allow_html=True,
        )
        # Raw heatmap coloured with inferno palette
        heat_img = cv2.resize(heatmap, (image.width, image.height))
        cmap     = cm.get_cmap("inferno")
        heat_rgb = (cmap(heat_img)[:, :, :3] * 255).astype(np.uint8)
        st.image(Image.fromarray(heat_rgb), use_container_width=True)

    with cam_col3:
        st.markdown(
            '<p style="font-size:11px;text-transform:uppercase;letter-spacing:2px;'
            'color:#4b5563;margin-bottom:8px">Overlay</p>',
            unsafe_allow_html=True,
        )
        overlay = overlay_heatmap(image, heatmap, alpha=0.5, colormap="inferno")
        st.image(overlay, use_container_width=True)

    # Explanation card
    st.markdown(
        '<div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.18);'
        'border-radius:14px;padding:16px 20px;font-size:13px;color:#94a3b8;line-height:1.7;margin-top:6px">'
        '<strong style="color:#e2e8f0">How to read the heatmap</strong><br>'
        'Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the image '
        'most strongly influenced the prediction. <span style="color:#ef4444">Red/yellow</span> regions '
        'are the most decisive features — typically petal shape, color distribution, and centre structure. '
        '<span style="color:#4b5563">Dark/purple</span> areas contributed little to the classification.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── AI Analysis Note ──────────────────────────────────────────────────
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    section_header("AI Analysis Note")
    risk_color_note = risk["color"] if uploaded_file else "#f9a8d4"
    display_name_note = predicted_class.replace("_", " ").title() if uploaded_file else ""
    st.markdown(
        f'<div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.18);'
        f'border-radius:14px;padding:16px 20px;font-size:13px;color:#94a3b8;line-height:1.7">'
        f'The model identified this flower as <strong style="color:#e2e8f0">{display_name_note}</strong> with '
        f'<strong style="color:#e2e8f0">{confidence:.1f}% confidence</strong> based on petal shape, '
        f'color distribution, and structural patterns. The dynamic pollen risk score of '
        f'<strong style="color:{risk_color_note}">{risk["score"]:.0f}/100 ({risk["tier"]})</strong> '
        f'accounts for species biology and current season (<em>{season}</em>).'
        f'</div>',
        unsafe_allow_html=True,
    )
