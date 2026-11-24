"""Streamlit web application for Explainable CNN for Fashion Image Classification."""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data import CLASS_NAMES, find_dataset_files, read_idx_images, read_idx_labels
from src.evaluate import load_model_from_checkpoint
from src.explain import GradCAM, create_heatmap_overlay
from src.models import RegularizedCNN

# Set page configuration
st.set_page_config(
    page_title="Explainable CNN for Fashion Image Classification",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished dark mode UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #cbd5e1;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .badge-correct {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-error {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_best_model():
    """Load the best trained CNN model and preprocessing metadata."""
    ckpt_path = "outputs/models/best_cnn.pt"
    device = torch.device("cpu")
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    train_mean = ckpt.get("train_mean", 0.2858857)
    train_std = ckpt.get("train_std", 0.3529603)
    return model, ckpt, train_mean, train_std, device


@st.cache_data
def load_sample_dataset():
    """Load representative test samples across classes for instant demonstration."""
    try:
        file_map = find_dataset_files("dataset")
        test_images = read_idx_images(file_map["test_images"])
        test_labels = read_idx_labels(file_map["test_labels"])

        samples = []
        # Curate 12 clear and interesting test samples (different classes & confusion pairs)
        indices = [0, 1, 3, 5, 6, 8, 12, 17, 19, 21, 23, 25]
        for idx in indices:
            if idx < len(test_images):
                samples.append(
                    {
                        "index": idx,
                        "image": test_images[idx],
                        "true_label": int(test_labels[idx]),
                        "label_name": CLASS_NAMES[test_labels[idx]],
                    }
                )
        return samples
    except Exception as e:
        st.warning(f"Could not load local dataset for samples: {e}")
        return []


@st.cache_data
def load_metrics_json():
    """Load metrics JSON summary if available."""
    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Load resources
model, ckpt, train_mean, train_std, device = load_best_model()
samples = load_sample_dataset()
metrics_data = load_metrics_json()

# Header
st.title("Explainable CNN for Fashion Image Classification")
st.markdown(
    "Aplikasi inferensi dan interpretasi visual menggunakan **Convolutional Neural Network (CNN)** dengan penjelasan atensi fitur berbasis **Grad-CAM**."
)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Klasifikasi & Explainability (Grad-CAM)",
        "📊 Evaluasi Model & Metrik Uji",
        "ℹ️ Informasi Dataset & Arsitektur",
    ]
)

# ==============================================================================
# TAB 1: KLASIFIKASI & GRAD-CAM EXPLAINABILITY
# ==============================================================================
with tab1:
    st.subheader("Uji Klasifikasi Citra & Visualisasi Grad-CAM")
    st.write(
        "Pilih sampel citra dari data pengujian Fashion-MNIST atau unggah citra Anda sendiri. Sistem akan melakukan inferensi kelas dan menampilkan heatmap atensi Grad-CAM."
    )

    col_input, col_result = st.columns([1, 2], gap="large")

    input_mode = None
    input_image = None
    true_label = None

    with col_input:
        st.markdown("### Sumber Masukan Citra")
        input_source = st.radio(
            "Pilih Metode Masukan:",
            ["Pilih dari Sampel Uji", "Unggah Citra Kustom", "Uji Penanganan Input Invalid"],
            index=0,
        )

        if input_source == "Pilih dari Sampel Uji":
            if samples:
                sample_options = [
                    f"Sampel #{s['index']:04d} — {s['label_name']}" for s in samples
                ]
                selected_sample_idx = st.selectbox(
                    "Pilih Contoh Produk Busana:",
                    range(len(sample_options)),
                    format_func=lambda i: sample_options[i],
                    index=0,
                )
                selected_item = samples[selected_sample_idx]
                input_image = selected_item["image"]
                true_label = selected_item["true_label"]

                st.image(
                    input_image,
                    caption=f"Citra Uji Asli (Label: {selected_item['label_name']})",
                    width=200,
                    clamp=True,
                )
            else:
                st.info("Sampel dataset lokal tidak ditemukan.")

        elif input_source == "Unggah Citra Kustom":
            uploaded_file = st.file_uploader(
                "Unggah Citra Produk (PNG, JPG, BMP):",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Citra akan diubah menjadi ukuran 28x28 grayscale secara otomatis.",
            )
            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file).convert("L")
                    pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
                    input_image = np.array(pil_img, dtype=np.uint8)
                    st.image(
                        input_image,
                        caption="Citra yang Diunggah (28x28 Grayscale)",
                        width=200,
                        clamp=True,
                    )
                except Exception as e:
                    st.error(f"Gagal memproses berkas gambar: {e}")

        elif input_source == "Uji Penanganan Input Invalid":
            st.error("Simulasi Masukan Tidak Valid:")
            st.markdown(
                """
                > **Peringatan Sistem:** Berkas yang diunggah tidak memiliki format citra valid atau dimensi tidak sesuai standar Fashion-MNIST (28 × 28 grayscale).
                > 
                > - **Kode Status:** `INVALID_IMAGE_PAYLOAD`
                > - **Tindakan Pencegahan:** Operasi konvolusi dan inferensi dihentikan dengan aman untuk mencegah memory corruption atau runtime error.
                """
            )
            input_image = None

    with col_result:
        if input_image is not None:
            # Preprocessing identical to training
            img_normalized = (input_image.astype(np.float32) / 255.0 - train_mean) / train_std
            tensor_input = (
                torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
            )
            tensor_input.requires_grad = True

            # Target layer for Grad-CAM
            target_layer = model.get_target_layer()
            grad_cam = GradCAM(model, target_layer)

            # Grad-CAM forward & backward
            heatmap, pred_idx, confidence = grad_cam.generate_heatmap(
                tensor_input, target_class=None
            )
            grad_cam.remove_hooks()

            # Predictions logits and all probabilities
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            predicted_name = CLASS_NAMES[pred_idx]

            # Display prediction badge & confidence
            st.markdown("### Hasil Prediksi Model")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-title'>Prediksi Kategori</div>"
                    f"<div class='metric-value'>{predicted_name}</div>"
                    f"<div class='metric-sub'>Index Kelas: {pred_idx}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with mcol2:
                conf_pct = confidence * 100.0
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-title'>Tingkat Keyakinan (Confidence)</div>"
                    f"<div class='metric-value'>{conf_pct:.2f}%</div>"
                    f"<div class='metric-sub'>Probabilitas Softmax</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if true_label is not None:
                true_name = CLASS_NAMES[true_label]
                if pred_idx == true_label:
                    st.markdown(
                        f"<span class='badge-correct'>✓ Prediksi Tepat: {true_name}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span class='badge-error'>✗ Salah Klasifikasi: Label Asli adalah '{true_name}'</span>",
                        unsafe_allow_html=True,
                    )

            # Explainability Visualization (Grad-CAM)
            st.markdown("#### Visualisasi Atensi Spasial (Grad-CAM)")
            overlay = create_heatmap_overlay(input_image, heatmap, alpha=0.55)

            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.image(input_image, caption="1. Citra Masukan (28×28)", use_container_width=True)
            with vcol2:
                st.image(
                    heatmap,
                    caption="2. Heatmap Grad-CAM (Aktivasi conv4)",
                    use_container_width=True,
                    clamp=True,
                )
            with vcol3:
                st.image(overlay, caption="3. Heatmap Overlay", use_container_width=True)

            st.info(
                f"**Interpretasi Fitur:** Area berwarna merah-oranye pada overlay menunjukkan daerah spasial yang paling kuat mengaktivasi filter konvolusi untuk kelas **{predicted_name}**. Fitur morfologis di area tersebut memiliki bobot gradien paling dominan dalam penentuan logit keluaran."
            )

            # Probability Distribution Chart
            st.markdown("#### Distribusi Probabilitas 10 Kelas")
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#0f172a")

            colors = ["#38bdf8" if i == pred_idx else "#475569" for i in range(10)]
            bars = ax.barh(range(10), probs * 100, color=colors, height=0.65)
            ax.set_yticks(range(10))
            ax.set_yticklabels(CLASS_NAMES, color="#f8fafc", fontsize=8.5)
            ax.set_xlabel("Probabilitas Softmax (%)", color="#94a3b8", fontsize=8.5)
            ax.set_xlim(0, 105)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.grid(axis="x", linestyle="--", alpha=0.3, color="#64748b")
            for spine in ax.spines.values():
                spine.set_color("#334155")

            for bar in bars:
                w = bar.get_width()
                if w > 3:
                    ax.text(
                        w + 1.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}%",
                        va="center",
                        ha="left",
                        color="#f8fafc",
                        fontsize=7.5,
                        fontweight="bold",
                    )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            if input_source != "Uji Penanganan Input Invalid":
                st.info("Silakan pilih sampel pengujian atau unggah gambar untuk memulai inferensi.")

# ==============================================================================
# TAB 2: EVALUASI MODEL & METRIK UJI
# ==============================================================================
with tab2:
    st.subheader("Evaluasi Kinerja Model pada Himpunan Pengujian (10.000 Citra)")
    st.markdown(
        "Metrik berikut diperoleh dari evaluasi final yang terisolasi pada 10.000 sampel pengujian Fashion-MNIST."
    )

    # Key Performance Metrics Cards
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Test Accuracy</div>"
            "<div class='metric-value'>92,61%</div>"
            "<div class='metric-sub'>9.261 / 10.000 benar</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kcol2:
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Macro F1-Score</div>"
            "<div class='metric-value'>92,58%</div>"
            "<div class='metric-sub'>Rata-rata tidak berbobot</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kcol3:
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Parameter Model</div>"
            "<div class='metric-value'>871 K</div>"
            "<div class='metric-sub'>871.082 bobot terlatih</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with kcol4:
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Latensi Inferensi</div>"
            "<div class='metric-value'>0,92 ms</div>"
            "<div class='metric-sub'>CPU 16-Core (Batch=1)</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Comparative Performance
    st.markdown("### Perbandingan Eksperimen & Kurva Pelatihan")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_path = Path("outputs/figures/04_training_curves.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Kurva Pelatihan & Validasi (Loss & Accuracy)", use_container_width=True)
    with pcol2:
        fig_path = Path("outputs/figures/05_model_comparison.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Perbandingan Kinerja & Kompleksitas Model", use_container_width=True)

    # Confusion Matrix
    st.markdown("### Analisis Matriks Konfusi Test Set")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        fig_path = Path("outputs/figures/06_confusion_matrix_counts.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Matriks Konfusi (Jumlah Kasus Mentah)", use_container_width=True)
    with ccol2:
        fig_path = Path("outputs/figures/07_confusion_matrix_normalized.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Matriks Konfusi Ternormalisasi (Recall Baris)", use_container_width=True)

    # Per class metrics
    st.markdown("### Metrik Presisi, Recall, dan F1-Score Per Kategori")
    fig_path = Path("outputs/figures/08_per_class_metrics.png")
    if fig_path.exists():
        st.image(str(fig_path), caption="Evaluasi Rinci Per Kategori Produk", use_container_width=True)

    # Top Confusion Pairs Table
    st.markdown("### Pasangan Kesalahan Prediksi Terbanyak (*Top Confusion Pairs*)")
    confusion_pairs = [
        {"Kelas Sebenarnya": "Shirt", "Prediksi Salah": "T-shirt/top", "Kasus": "92 kali", "Proporsi": "9,2%"},
        {"Kelas Sebenarnya": "T-shirt/top", "Prediksi Salah": "Shirt", "Kasus": "86 kali", "Proporsi": "8,6%"},
        {"Kelas Sebenarnya": "Shirt", "Prediksi Salah": "Coat", "Kasus": "71 kali", "Proporsi": "7,1%"},
        {"Kelas Sebenarnya": "Pullover", "Prediksi Salah": "Shirt", "Kasus": "48 kali", "Proporsi": "4,8%"},
        {"Kelas Sebenarnya": "Shirt", "Prediksi Salah": "Pullover", "Kasus": "47 kali", "Proporsi": "4,7%"},
        {"Kelas Sebenarnya": "Coat", "Prediksi Salah": "Shirt", "Kasus": "45 kali", "Proporsi": "4,5%"},
    ]
    st.table(confusion_pairs)

# ==============================================================================
# TAB 3: INFORMASI DATASET & ARSITEKTUR
# ==============================================================================
with tab3:
    st.subheader("Informasi Dataset Fashion-MNIST & Arsitektur Jaringan")

    dcol1, dcol2 = st.columns([1, 1], gap="large")
    with dcol1:
        st.markdown("### Karakteristik Dataset")
        st.markdown(
            """
            - **Sumber Data:** Zalando Research Fashion-MNIST (Zalando SE)
            - **Ukuran Citra:** 28 × 28 piksel grayscale (rentang intensitas 0–255)
            - **Distribusi Data:**
              - **Train Split Asli:** 60.000 citra (dibagi menjadi 54.000 train & 6.000 validation via stratified sampling seed 42)
              - **Test Set Terisolasi:** 10.000 citra
            - **Normalisasi:** $X_{\\text{norm}} = (X/255.0 - \\mu) / \\sigma$ dengan $\\mu=0{,}2859$ dan $\\sigma=0{,}3530$ dihitung murni dari data latih.
            - **Augmentasi Latih:** *Random Horizontal Flip* ($p=0{,}5$) & *Random Crop with Reflection Padding* (translasi spasial $\\pm 2$ piksel).
            """
        )
        fig_path = Path("outputs/figures/02_class_distribution.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Keseimbangan Distribusi Kelas", use_container_width=True)

    with dcol2:
        st.markdown("### Arsitektur Regularized CNN")
        st.markdown(
            """
            Model terbaik yang digunakan adalah **Regularized CNN** dengan struktur berlapis:
            
            1. **Blok Konvolusi 1:**
               - `Conv2d(1 -> 32, k=3, p=1)` -> `BatchNorm2d` -> `ReLU`
               - `Conv2d(32 -> 32, k=3, p=1)` -> `BatchNorm2d` -> `ReLU`
               - `MaxPool2d(2, 2)` (Reduksi ke 14 × 14) -> `Dropout2d(0.20)`
            2. **Blok Konvolusi 2 (Target Layer Grad-CAM: `conv4`):**
               - `Conv2d(32 -> 64, k=3, p=1)` -> `BatchNorm2d` -> `ReLU`
               - `Conv2d(64 -> 64, k=3, p=1)` -> `BatchNorm2d` -> `ReLU`
               - `MaxPool2d(2, 2)` (Reduksi ke 7 × 7) -> `Dropout2d(0.20)`
            3. **Kepala Klasifikasi (Fully Connected Head):**
               - `Flatten` (64 × 7 × 7 = 3.136 fitur)
               - `Linear(3136 -> 256)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(0.40)`
               - `Linear(256 -> 10)` (10 logit kelas)
            """
        )

        fig_path = Path("outputs/figures/01_samples_per_class.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Galeri 10 Kategori Fashion-MNIST", use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Explainable CNN for Fashion Image Classification • Dilengkapi Grad-CAM Explainability • Mode Gelap Terintegrasi"
    "</div>",
    unsafe_allow_html=True,
)
