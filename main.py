import tensorflow as tf
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import base64
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps

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
    color:White;
    }

    [data-testid="stHeader"] {
    background: rgba(0,0,0,0);
    }   

    
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('coolbackgrounds-gradient-salt.png')

st.title("Sistem Klasifikasi Daun Herbal")
st.write("Ini adalah sistem klasifikasi daun herbal menggunakan deep learning yaitu Convolutional Neural Network, terdapat 10 jenis daun herbal yang dapat diklasifikasikan sistem ini yaitu: ")
st.markdown('''
        1. Daun salam
        2. Daun kari
        3. Daun kemangi
        4. Daun kumis kucing
        5. Daun kelor
        6. Daun katuk
        7. Daun ketumbar
        8. Daun seledri
        9. Daun binahong
        10. Daun sirih
       ''')
st.write("Pilih model sesuai dengan skenario gambar lalu Upload gambar daun herbal untuk mengetahui jenisnya.")

model_choice = st.selectbox("Pilih model:", ["Skenario Terang", "Skenario Gelap"])

#load model, set cache to prevent reloading
@st.cache_resource()
def load_model(mode='terang'):
    if mode == 'gelap':
        return tf.keras.models.load_model('model_gelap_test_val.keras')
    else:
        return tf.keras.models.load_model('model_terang_val.keras')

# Load model berdasarkan pilihan
with st.spinner('Model lagi Loading..'):
    mode = 'gelap' if model_choice == 'Model Gelap' else 'terang'
    model = load_model(mode)

class_names=[ 'binahong',
              'kari', 
              'katuk',
              'kelor',
              'kemangi',
              'ketumbar',
              'kumis kucing',
              'salam',
              'seledri',
              'sirih']

tab1, tab2 = st.tabs(["📷 Prediksi Gambar", "📊 Evaluasi & Confusion Matrix"])

#TAB 1: PREDIKSI GAMBAR
with tab1:
    uploaded_file = st.file_uploader("Upload Gambar Daun", type=["jpg", "png", "jpeg"])
    threshold = 3
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", width=200)
        if st.button("🔍 Lakukan Prediksi", key="predict_button"):
            with st.spinner("Mengklasifikasikan..."):
    # Preprocess gambar
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
    # Predict
                prediction = model.predict(img_batch)
                confidence = np.argmax(prediction)
                pred_class = class_names[np.argmax(prediction)]
                if confidence <= threshold:
                    st.success(f"✅ Prediksi: Daun **{pred_class}**")
                    st.write(f"📊 Probabilitas tertinggi: `{confidence:.4f}`")
                else:
                    st.warning(
                    f"⚠️ Model tidak yakin dalam memprediksi.\n"
                    f"Probabilitas tertinggi: `{confidence:.4f}` di bawah threshold `{threshold}`"
                )

#TAB 2: EVALUASI & CMATRIX
with tab2:
    st.markdown("### Evaluasi Model dari Dataset Test")
    test_dir = st.text_input("Masukkan path ke folder test set (berisi subfolder per kelas):")

    if st.button("Evaluasi Model"):
        if not os.path.exists(test_dir) or not os.listdir(test_dir):
            st.error("Folder test tidak ditemukan. Pastikan path benar dan tidak kosong.")
        else:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            test_gen = ImageDataGenerator(rescale=1./255)
            test_data = test_gen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=16,
                class_mode='categorical',
                shuffle=False
        )
        # Evaluasi metrik
            loss, accuracy = model.evaluate(test_data, verbose=0)
            st.success(f"✅ Akurasi: `{accuracy:.4f}` | ❌ Loss: `{loss:.4f}`")

        # Prediksi batch
            y_true = test_data.classes
            predictions = model.predict(test_data)
            y_pred = np.argmax(predictions, axis=1)

        # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

         # Classification Report
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()
            st.markdown("### 📋 Classification Report")
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)