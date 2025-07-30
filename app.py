import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import base64

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Daun Herbal",
    initial_sidebar_state = 'auto'
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    font-family:sans-serif;
    }

    [data-testid="stHeader"] {
    background: rgba(0,0,0,0);
    }   
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('coolbackgrounds-gradient-kale.png')


# ----- Setup Model -----
@st.cache_resource()
def load_model(mode='Skenario Terang'):
    if mode == 'Skenario Gelap':
        return tf.keras.models.load_model('model_gelap_val.keras')
    else:
        return tf.keras.models.load_model('model_terang_val.keras')

class_names = [
    "binahong", "kari", "katuk", "kelor", "kemangi",
    "ketumbar", "kumis kucing", "salam", "seledri", "sirih"
]

# ----- Sidebar -----
st.sidebar.title("🌿 Daftar Kelas Daun")
leaf_classes = [
    "Daun Binahong", "Daun Kari", "Daun Katuk", "Daun Kelor",
    "Daun Kemangi", "Daun Ketumbar", "Daun Kumis kucing",
    "Daun Salam", "Daun Seledri", "Daun Sirih"
]
for leaf in leaf_classes:
    st.sidebar.write(f"• {leaf}")

# ----- Judul Halaman -----
st.markdown('<h2 ">Sistem klasifikasi daun herbal</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: justify">
        Ini adalah sistem klasifikasi daun herbal menggunakan deep learning yaitu Convolutional Neural Network 
        dengan arsitektur ResNet-50. Terdapat 10 jenis daun herbal yang dapat diklasifikasikan sistem ini, 
        bisa dilihat pada daftar di samping kiri. Mohon gunakan gambar daun herbal yang jelas, jangan gambar dengan background yang ramai!
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
# ----- Input Gambar -----
col1, col2= st.columns(2)

with col1:
    st.header("Input Gambar")
    # Pilihan model
    model_choice = st.selectbox(
        "Pilih model sesuai kondisi foto:",
        ["Skenario Terang", "Skenario Gelap"]
    )
    # Widget upload file
    uploaded_file = st.file_uploader(
        "Upload gambar daun di sini:",
        type=["jpg", "jpeg", "png"]
    )
    #ambang batas
    threshold = 0.8
    # Tombol prediksi
    predict_button = st.button(
        "🔍 Lakukan Prediksi",
        key="predict_button",
        use_container_width=True
    )
    # Memuat model berdasarkan pilihan
    with st.spinner('Model sedang dimuat...'):
        model = load_model(model_choice)

with col2:
    st.header("Hasil Analisis")
    if uploaded_file is not None:
        # Menampilkan gambar yang diupload di kolom kedua
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diupload", width=200)

        # Logika prediksi hanya berjalan jika tombol ditekan
        if predict_button:
            with st.spinner("Mengklasifikasikan..."):
                # Preprocessing gambar
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0 # Normalisasi
                img_batch = np.expand_dims(img_array, axis=0)

                # Prediksi
                prediction = model.predict(img_batch)
                confidence = np.max(prediction)
                pred_class = class_names[np.argmax(prediction)]

                if confidence >= threshold:
                    # Jika probabilitas di atas threshold, tampilkan hasil
                    st.markdown(f"### Prediksi: Daun **{pred_class}** (Probabilitas: **{confidence:.4f}**)")
                else:
                    # Jika di bawah threshold, tampilkan peringatan
                    st.warning(f"⚠️ **Model Tidak Yakin**")
                    st.markdown(f"Prediksi teratas adalah **{pred_class}**, namun probabilitasnya (`{confidence:.4f}`) di bawah ambang batas yang telah ditentukan (`{threshold}`).")
    else:
        st.info("Silakan upload gambar dan klik tombol 'Lakukan Prediksi' di kolom sebelah kiri.")
