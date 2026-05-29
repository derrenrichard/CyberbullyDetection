import streamlit as st
import joblib
import re
import numpy as np

# 1. Load the saved model and vectorizer (Sesuai nama file di gambar)
@st.cache_resource # Gunakan cache agar model tidak di-load ulang setiap kali ada interaksi
def load_models():
    model = joblib.load('rf_cyberbullying_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# 2. Fungsi Pembersihan Teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 3. Fungsi untuk Highlight Kata berdasarkan TF-IDF
def highlight_important_words(original_text, cleaned_text, vectorizer, vectorized_input, top_n=3):
    # Ambil daftar semua kata (vocabulary) dari vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Ambil array bobot/skor TF-IDF dari input pengguna
    tfidf_scores = vectorized_input.toarray()[0]
    
    # Cari indeks kata dengan skor tertinggi (Top N)
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    
    # Ekstrak kata-kata penting yang skornya lebih dari 0
    important_words = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
    
    # Lakukan proses highlight pada teks asli pengguna menggunakan Regex
    highlighted_text = original_text
    for word in important_words:
        # Regex (?i) untuk case-insensitive, \b untuk batasan kata (word boundary)
        pattern = re.compile(rf'(?i)(\b{word}\b)')
        # Ganti kata dengan versi yang dibungkus tag HTML <mark>
        highlighted_text = pattern.sub(r'<mark style="background-color: #ffcccc; color: #cc0000; padding: 0 4px; border-radius: 4px;">\1</mark>', highlighted_text)
        
    return highlighted_text, important_words

# ==========================================
# USER INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")
st.title("🛡️ Advanced Cyberbullying Detection")
st.write("Aplikasi ini tidak hanya mendeteksi cyberbullying, tetapi juga mengkategorikannya dan menyorot kata-kata yang memicu deteksi.")

user_input = st.text_area("Masukkan teks di sini:", height=150)

if st.button("Analisis Teks"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner('Menganalisis teks...'):
            # A. Proses input
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # B. Prediksi Model
            prediction = model.predict(vectorized_input)[0]
            
            # C. Tampilkan Hasil
            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            if prediction == "not_cyberbullying":
                st.success("✅ **Aman!** Teks ini terdeteksi TIDAK mengandung unsur cyberbullying.")
            else:
                # Menampilkan Kategori Cyberbullying
                st.error(f"⚠️ **TERDETEKSI CYBERBULLYING!**")
                st.warning(f"**Kategori:** {prediction.replace('_', ' ').title()}")
                
                # D. Proses dan Tampilkan Highlight Kata
                highlighted_result, trigger_words = highlight_important_words(
                    original_text=user_input,
                    cleaned_text=cleaned_input,
                    vectorizer=vectorizer,
                    vectorized_input=vectorized_input,
                    top_n=4 # Anda bisa mengubah jumlah maksimal kata yang di-highlight di sini
                )
                
                st.markdown("### Konteks Teks:")
                st.write("Kata-kata yang disorot merah adalah kata dengan bobot tertinggi yang mempengaruhi keputusan model:")
                
                # Menggunakan unsafe_allow_html=True agar tag <mark> bisa dirender oleh Streamlit
                st.markdown(f"> {highlighted_result}", unsafe_allow_html=True)
                
                # Menampilkan list kata pemicu di bawahnya
                if trigger_words:
                    st.info(f"**Kata pemicu utama:** {', '.join(trigger_words)}")
