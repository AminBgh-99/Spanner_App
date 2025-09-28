import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO
import time
import os

# Optional für Kamera
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False

# ============================= 
# Modell laden 
# ============================= 
@st.cache_resource
def lade_yolo():
    return YOLO("C:/Users/bough/spanner_gui/optimierter_finaler_best.pt")

@st.cache_resource
def lade_gradcam_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model_yolo = lade_yolo()
model_cam = lade_gradcam_model()



# Seitenlayout & Stil
# =============================
st.set_page_config(page_title="KI-Spannerprüfung", layout="wide")

st.markdown("""
<style>
    .stButton > button {
        height: 3em;
        width: 100%;
        font-size: 1.1em;
    }
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        border: 3px solid black;
        padding: 0.5em 1em;
        margin: auto;
        margin-bottom: 2em;
        width: fit-content;
        background-color: #f0f0f0;
    }
    .emoji-title {
        font-size: 2.2em;
        text-align: center;
    }
    .center-text {
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
    }
    .result-box {
        background-color: #d4edda;
        padding: 1em;
        border-radius: 10px;
        font-size: 1.2em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-container"><div class="emoji-title">🧠 KI-gestützte Spanner-Prüfung</div></div>', unsafe_allow_html=True)

# ============================= 
# Spanner-Auswahl 
# ============================= 
st.markdown("<div class='center-text'>📦 Erwarteter Spanner wählen</div>", unsafe_allow_html=True)

# Initialisiere den Key in session_state, falls er noch nicht existiert
if 'selected_spanner' not in st.session_state:
    st.session_state.selected_spanner = None  # Initialisiere es mit einem Standardwert

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("46 (klein)"):
        st.session_state.selected_spanner = "46"
with col2:
    if st.button("77 (mittel)"):
        st.session_state.selected_spanner = "77"
with col3:
    if st.button("125 (groß)"):
        st.session_state.selected_spanner = "125"

if st.session_state.selected_spanner:
    st.success(f"✅ Gewählt: Spanner {st.session_state.selected_spanner}")

# ============================= 
# Bildquelle: Upload oder Kamera 
# ============================= 
st.markdown("<br><div class='center-text'>🔍 Bildquelle wählen</div><br>", unsafe_allow_html=True)
wahl = st.radio("Wählen Sie die Bildquelle", ["📁 Bild vom Computer hochladen", "📷 Livebild mit Basler-Kamera aufnehmen"])

bildquelle = None
if wahl == "📁 Bild vom Computer hochladen":
    uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        bildquelle = Image.open(uploaded_file).convert("RGB")
elif wahl == "📷 Livebild mit Basler-Kamera aufnehmen":
    if PYLON_AVAILABLE:
        if st.button("📸 Bild aufnehmen"):
            try:
                # Kamera öffnen und Bild aufnehmen
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                camera.Open()
                camera.StartGrabbing()
                grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                img = grab.Array
                camera.StopGrabbing()
                camera.Close()

                # Farbkonvertierung von Bayer in RGB (Verwende BG2RGB statt GR2RGB)
                bildquelle = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB))

                # Bild in Streamlit anzeigen
                st.session_state.livebild = bildquelle
                st.image(bildquelle, caption="📸 Livebild", use_container_width=True)
            except Exception as e:
                st.error(f"Kamerazugriff fehlgeschlagen: {e}")
    else:
        st.warning("Pylon SDK nicht installiert – Kameraaufnahme nicht verfügbar.")

if 'livebild' in st.session_state and bildquelle is None:
    bildquelle = st.session_state.livebild

# ============================= 
# Grad-CAM über ResNet50 (für Crop) 
# ============================= 
def generate_gradcam_on_crop(crop_img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(crop_img).unsqueeze(0)

    final_conv = model_cam.layer4
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = final_conv.register_forward_hook(forward_hook)
    handle_bwd = final_conv.register_backward_hook(backward_hook)

    output = model_cam(input_tensor)
    pred_class = output.argmax().item()
    model_cam.zero_grad()
    output[0, pred_class].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.numpy()
    cam = cv2.resize(cam, (crop_img.width, crop_img.height))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(crop_img), 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay)

# ============================= 
# Analyse 
# ============================= 
if bildquelle:
    image_array = np.array(bildquelle)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        bildquelle.save(tmp_file.name)
        results = model_yolo(tmp_file.name)

    if results[0].boxes:  # Überprüfen, ob Ergebnisse vorhanden sind
        erkannter_spanner = results[0].names[int(results[0].boxes.cls[0].item())]  # Richtig eingerückt
        erkannter_spanner = erkannter_spanner or "Unbekannt"  # Sicherstellen, dass es kein None ist
        ist_korrekt_positioniert = "i.O" in erkannter_spanner or "i.o" in erkannter_spanner

        st.markdown("<br><div class='center-text'>📌 Dein aktuelles Prüfbild</div><br>", unsafe_allow_html=True)
        st.image(results[0].plot(), caption="📁 Analysebild mit Bounding Boxen", use_container_width=True)

        st.markdown("<br><div class='center-text'>🧠 KI hat das Bild analysiert</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='center-text'>🔖 Erkanntes Objekt: {erkannter_spanner}</div><br>", unsafe_allow_html=True)
        
        # Hier der Vergleich, ob der richtige Spanner erkannt wurde
        gewählter_spanner = st.session_state.selected_spanner
        spanner_stimmt = gewählter_spanner in erkannter_spanner

        if spanner_stimmt and ist_korrekt_positioniert:
            st.markdown('<div class="result-box">Alles passt! Der Spanner wurde korrekt erkannt und positioniert.</div>', unsafe_allow_html=True)
        elif spanner_stimmt and not ist_korrekt_positioniert:
            st.markdown('<div class="result-box" style="background-color:#f8d7da;">Spanner erkannt, aber die Position ist nicht korrekt.</div>', unsafe_allow_html=True)
        elif not spanner_stimmt and ist_korrekt_positioniert:
            st.markdown(f'<div class="result-box" style="background-color:#fff3cd;">Erwartet war Spanner {gewählter_spanner}, aber erkannt wurde {erkannter_spanner}. Dieser ist korrekt positioniert, aber nicht der erwartete Typ.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box" style="background-color:#f8d7da;">Der eingesetzte Spanner stimmt nicht mit der Auswahl überein.</div>', unsafe_allow_html=True)
    else:
        st.warning("❗ Kein Objekt erkannt.")
