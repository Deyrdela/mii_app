import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Классификатор изображений", layout="wide")

# Стили
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
        color: white;
        border-radius: 30px;
        padding: 12px 28px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
    }
    .pred-box {
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .cat-pred { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .dog-pred { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }
    .rock-pred { background: linear-gradient(135deg, #434343, #000000); color: white; }
    .paper-pred { background: linear-gradient(135deg, #2193b0, #6dd5ed); color: white; }
    .scissors-pred { background: linear-gradient(135deg, #ee9ca7, #ffdde1); color: white; }
    .info-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Классификатор изображений</div>', unsafe_allow_html=True)
st.markdown("---")

# вкладки
tab1, tab2 = st.tabs(["🐱 КОШКИ и СОБАКИ 🐶", "✊ КАМЕНЬ, НОЖНИЦЫ, БУМАГА ✋✂️"])

# вкладка 1: Кошки и собаки
with tab1:
    st.markdown("### Определяет, кто на фото: кошка или собака 🐾")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Загрузите изображение")
        uploaded_file_1 = st.file_uploader("", type=["jpg", "png", "jpeg"], key="upload_1", label_visibility="collapsed")
        
        st.markdown("#### Выберите модель")
        selected_model_1 = st.radio(
            "",
            ["Исходная CNN", "CNN + SE"],
            horizontal=True,
            key="model_1"
        )
        model_name_1 = "Исходная CNN" if "Исходная" in selected_model_1 else "CNN + SE"
        
        # Подсказка о размере
        st.markdown('<div class="info-box">Модель обучена на изображениях 32×32 пикселя<br>Ваше фото автоматически сжимается до нужного размера</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Предпросмотр")
        if uploaded_file_1 is not None:
            image_1 = Image.open(uploaded_file_1)
            st.image(image_1, caption="Ваше фото", width=250)
            st.caption(f"Оригинальный размер: {image_1.size[0]}×{image_1.size[1]} пикселей")
        else:
            st.info("👈 Загрузите фото")
    
    if uploaded_file_1 is not None:
        if st.button("Распознать", key="btn_1", use_container_width=True):
            with st.spinner("Нейросеть анализирует..."):
                time.sleep(0.3)
                
                if model_name_1 == "Исходная CNN":
                    model = tf.keras.models.load_model("cnn_model.keras")
                else:
                    model = tf.keras.models.load_model("se_model.keras")
                
                img = image_1.resize((32, 32))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array, verbose=0)[0][0]
            
            if pred > 0.5:
                st.markdown(f'<div class="pred-box dog-pred">СОБАКА 🐶 {pred*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box cat-pred">КОШКА 🐱  {(1-pred)*100:.1f}%</div>', unsafe_allow_html=True)
            
            st.progress(float(pred if pred > 0.5 else 1 - pred))
            st.caption(f"Модель: {model_name_1}")

# вкладка 2: Камень, Ножницы, Бумага
with tab2:
    st.markdown("### Распознаёт жест руки: камень ✊, ножницы ✂️ или бумага ✋")
    
    rps_classes = ["КАМЕНЬ ✊", "НОЖНИЦЫ ✂️", "БУМАГА ✋"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Загрузите изображение")
        uploaded_file_2 = st.file_uploader("", type=["jpg", "png", "jpeg"], key="upload_2", label_visibility="collapsed")
        
        st.markdown("#### Выберите модель")
        selected_model_2 = st.radio(
            "",
            ["MobileNetV2 (RPS)"],
            horizontal=True,
            key="model_2"
        )
        
        # Подсказка о размере
        st.markdown('<div class="info-box">Модель обучена на изображениях 100×100 пикселей<br>Ваше фото автоматически сжимается до нужного размера</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Предпросмотр")
        if uploaded_file_2 is not None:
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption="Ваше фото", width=250)
            st.caption(f"Оригинальный размер: {image_2.size[0]}×{image_2.size[1]} пикселей")
        else:
            st.info("👈 Загрузите фото жеста руки")
    
    if uploaded_file_2 is not None:
        if st.button("Распознать жест", key="btn_2", use_container_width=True):
            with st.spinner("Нейросеть анализирует жест..."):
                time.sleep(0.3)
                
                model_rps = tf.keras.models.load_model("mobilenetv2_rps.keras")
                
                img = image_2.resize((100, 100))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred_probs = model_rps.predict(img_array, verbose=0)[0]
                pred_class = np.argmax(pred_probs)
                confidence = pred_probs[pred_class] * 100
                
                result_class = rps_classes[pred_class]
                
                if pred_class == 0:
                    st.markdown(f'<div class="pred-box rock-pred"> {result_class} {confidence:.1f}%</div>', unsafe_allow_html=True)
                elif pred_class == 1:
                    st.markdown(f'<div class="pred-box scissors-pred"> {result_class} {confidence:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="pred-box paper-pred"> {result_class} {confidence:.1f}%</div>', unsafe_allow_html=True)
                
                st.progress(float(confidence / 100))
                st.caption(f"Модель: {selected_model_2}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>✨ Нейросети обучены на разных датасетах ✨</p>", unsafe_allow_html=True)