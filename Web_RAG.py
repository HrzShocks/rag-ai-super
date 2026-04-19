import streamlit as st
import PyPDF2
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai

st.set_page_config(page_title="RAG AI Super", page_icon="🧠")
st.title("🧠 Mesin RAG: Penakluk Dokumen Raksasa")
st.write("Upload PDF tebal lu! Otak AI ngga bakal meledak karena kita pakai Vector Database!")

KUNCI_RAHASIA = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = KUNCI_RAHASIA

@st.cache_resource
def proses_dokumen_ke_vector(file_uploads):
    teks_full=""
    for file in file_uploads:
        mesin_pembaca = PyPDF2.PdfReader(file)
        for halaman in mesin_pembaca.pages:
            if halaman.extract_text():
                teks_full += halaman.extract_text() + "\n"

    pemotong = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    kepingan_teks = pemotong.split_text(teks_full)

    mesin_embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    brankas_vector = Chroma.from_texts(kepingan_teks, mesin_embedding)

    return brankas_vector

file_uploads = st.file_uploader("Upload PDF Tebal / Banyak PDF lu!", type="pdf", accept_multiple_files=True)

if file_uploads:
    try:
        with st.spinner("Membangun brankas Vector dan Mengubah Teks ke Matematika...(Tunggu King!)"):
            brankas_vector = proses_dokumen_ke_vector(file_uploads)
        st.success("✅ Brankas Vector Selesai Dibangun! Dokumen siap diinterogasi!")
    except Exception as e:
        st.cache_resource.clear()
        st.error("🚨 Waduh King! Mesin Google lagi ngos-ngosan atau kuota gratisan limit.")
        st.warning(f"Pesan Error Asli:{e}")
        st.info("💡 Solusi: Tunggu 1 menit, terus pencet F5 (Refresh) dan coba upload PDF yang halamannya lebih sedikit ya!")
        st.stop()

    client = genai.Client(api_key=KUNCI_RAHASIA)

    if "pesan_rag" not in st.session_state:
        st.session_state.pesan_rag = []

    for pesan in st.session_state.pesan_rag:
        with st.chat_message(pesan["role"]):
            st.markdown(pesan["teks"])
    
    prompt = st.chat_input("Tanya isi dokumen raksasa ini...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.pesan_rag.append({"role": "user", "teks": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Mencari kordinat paragraf & Mengingat Obrolan..."):

                try:
                    dokumen_relevan = brankas_vector.similarity_search(prompt, k=3)
                    teks_contekan = ""
                    for doc in dokumen_relevan:
                        teks_contekan += doc.page_content + "\n\n"

                    history_obrolan = ""
                    for pesan_lama in st.session_state.pesan_rag[-5:-1]: 
                        siapa = "USER" if pesan_lama["role"] == "user" else "AI"
                        history_obrolan += f"{siapa}: {pesan_lama['teks']}\n"

                    prompt_gabungan = f"""
                    Kamu adalah Asisten AI Pintar. 
                    Jawab pertanyaan baru dari user HANYA berdasarkan 'Contekan Dokumen' di bawah ini.
                    Gunakan 'History Obrolan' untuk memahami konteks jika user menggunakan kata ganti (seperti 'itu', 'dia', 'yang tadi').

                    --- HISTORY OBROLAN SEBELUMNYA ---
                    {history_obrolan}

                    --- CONTEKAN DOKUMEN MULAI ---
                    {teks_contekan}
                    --- CONTEKAN DOKUMEN SELESAI ---

                    Pertanyaan Baru User: {prompt}
                    """

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_gabungan
                    )

                    jawaban = response.text
                    st.markdown(jawaban)
                    st.session_state.pesan_rag.append({"role": "assistant", "teks": jawaban})

                except Exception as e:
                    pesan_gagal = "🙏 Maaf Bos, koneksi ke otak AI terputus atau limit. Tunggu beberapa detik dan coba nanya lagi ya!"
                    st.error(pesan_gagal)
                    st.session_state.pesan_rag.append({"role": "assistant", "teks": pesan_gagal})

else:
    st.info("Upload dokumen dulu king buat ngebangun Vector Database!")
