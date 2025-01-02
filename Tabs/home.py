import streamlit as st
from web_functions import load_data
from Tabs import home, predict, visualise  # Pastikan file/folder Tabs tersedia

def app(df, x, y):
    # Custom HTML for title with purple color, centered alignment, larger background image, blur effect, and button inside the image
    st.markdown(
        """
        <style>
            .title-container {
                position: relative;
                text-align: center;
                color: white;
                background-image: url("https://agrokomplekskita.com/wp-content/uploads/2018/01/BUNGA-IRIS.jpg");
                padding: 100px;
                font-size: 3rem;
                background-size: cover;
                background-position: left;
                border-radius: 10px;
            }
            .title-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: inherit;
                filter: blur(2px); /* Apply blur to the background only */
                z-index: 0; /* Ensure the blurred background is behind text and button */
                border-radius: inherit; /* Make sure the background has rounded corners */
            }
            .title-container h1 {
                position: relative;
                z-index: 1; /* Make sure text is above the blurred background */
            }
            .start-button {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                padding: 10px 20px;
                font-size: 1.2rem;
                background-color: #800080;
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                z-index: 1; /* Keep button above the blurred background */
            }
        </style>
        <div class="title-container">
            <h1>Aplikasi Prediksi Jenis Tanaman Iris</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Informasi Tentang Bunga Iris (justified text alignment)
    st.markdown(
        """
        <p style='text-align: justify;'>
        Bunga iris adalah salah satu jenis tanaman berbunga yang terkenal karena keindahan bentuk dan warna bunganya. 
        Nama 'iris' berasal dari bahasa Yunani yang berarti 'pelangi', sesuai dengan ragam warna bunga ini. 
        Tiga jenis iris yang paling umum adalah Iris-setosa, Iris-versicolor, dan Iris-virginica. 
        Setiap spesies memiliki karakteristik unik yang membuatnya menarik untuk dipelajari.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # Menambahkan Gambar Iris Berjejeran dan memastikan gambar tetap berjejer horizontal dan rata tengah
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://www.re-natur.de/media/catalog/product/cache/e42b3e3e6979353f5fe1d4a80fd6f01b/4/0/40271_iris_setosa.jpg", caption="Iris Setosa", width=150)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg", caption="Iris Versicolor", width=150)
    with col3:
        st.image("https://www.agrecol.com/assets/images/store%20images/irivir%206001.jpg", caption="Iris Virginica", width=150)

    # Menampilkan Data
    st.write("""Dataset yang digunakan dalam prediksi bunga iris :""")
    df.drop('Id', axis=1, inplace=True)
    st.write(df)