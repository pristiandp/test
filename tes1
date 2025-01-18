import streamlit as st

# Judul aplikasi
st.title("Aplikasi Sederhana dengan Streamlit")

# Input teks
name = st.text_input("Masukkan nama Anda:", "")

# Slider untuk memilih angka
age = st.slider("Berapa usia Anda?", min_value=0, max_value=100, value=25)

# Tombol untuk submit
if st.button("Submit"):
    if name:
        st.success(f"Halo {name}, usia Anda adalah {age} tahun!")
    else:
        st.warning("Mohon masukkan nama Anda terlebih dahulu.")

# Menampilkan gambar (opsional)
st.image("https://via.placeholder.com/300x150", caption="Contoh Gambar")
