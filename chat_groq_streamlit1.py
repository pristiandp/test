import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
from langchain_groq import ChatGroq

# Set the page layout to 'wide' to make better use of space
st.set_page_config(layout="wide")

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = ChatGroq(
    groq_api_key="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",
    model_name='gemma2-9b-it'
)

# Initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)

# Define the chat prompts for both risk explanation and mitigation
prompt_risk = ChatPromptTemplate.from_messages(
    [
        ("system", "Anda adalah asisten yang membantu menjelaskan risiko pekerjaan berdasarkan deskripsi pekerjaan dan lokasi yang diberikan secara sangat singkat dan to the point. dalam bentuk point-point. misalnya: 1. Paparan Bahan Kimia 2.Ketinggian 3.Kebakaran dan seterusnya"),
        ("human", "Gunakan deskripsi {description} dan lokasi {location} untuk menjelaskan risiko. Gunakan hanya {context} untuk menjawab pertanyaan.")
    ]
)

prompt_mitigation = ChatPromptTemplate.from_messages(
    [
        ("system", "Anda adalah asisten yang membantu menjelaskan mitigasi risiko berdasarkan risiko yang telah dijelaskan secara sangat singkat dan to the point.  misalnya: 1. Paparan Bahan Kimia: Penggunaan APD (masker, sarung tangan, pelindung mata) 2. Ketinggian: Penggunaan peralatan keselamatan di ketinggian (harness, scaffolding yang aman) 3. Kebakaran: sediakan alat pemadam api"),
        ("human", "Berikan mitigasi risiko berdasarkan point-point dari {risk_description} dan konteks {context}.")
    ]
)

# Define the retrieval chains for both risk explanation and mitigation
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain_risk = create_stuff_documents_chain(llm, prompt_risk)
retrieval_chain_risk = create_retrieval_chain(retriever, combine_docs_chain_risk)

combine_docs_chain_mitigation = create_stuff_documents_chain(llm, prompt_mitigation)
retrieval_chain_mitigation = create_retrieval_chain(retriever, combine_docs_chain_mitigation)

# Streamlit App
st.title("Doctor JSA")
st.write("""**Doctor JSA** membantu Anda melakukan analisis risiko pekerjaan dan mitigasi risiko secara otomatis.""")

# Create two columns for input and output
col1, col2 = st.columns(2, gap="small")

# Input fields for the user input (left column)
with col1:
    st.header("Input")
    # Combine Deskripsi and Lokasi Pekerjaan into one input field
    job_info = st.text_input("Deskripsi dan Lokasi Pekerjaan", placeholder="Masukkan deskripsi pekerjaan dan lokasi di sini...")

    # Handle submission button
    submit_button = st.button("Kirim")
    
    if submit_button and job_info.strip():
        with st.spinner("Mengambil risiko dan mitigasi..."):
            # Separate job description and location from the combined input
            description, location = job_info.split(",", 1) if "," in job_info else (job_info, "Tidak ada lokasi spesifik")
            
            # Retrieve the context from the vector store based on the query
            context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents("analisis risiko pekerjaan")])

            # Input data for risk explanation
            input_data_risk = {
                "input": "analisis risiko pekerjaan",
                "description": description.strip(),
                "location": location.strip(),
                "context": context
            }

            # Get the risk explanation
            result_risk = retrieval_chain_risk.invoke(input_data_risk)
            risk_description = result_risk["answer"]

            # Input data for risk mitigation
            input_data_mitigation = {
                "input": "mitigasi risiko pekerjaan",
                "risk_description": risk_description,
                "context": context
            }

            # Get the risk mitigation explanation
            result_mitigation = retrieval_chain_mitigation.invoke(input_data_mitigation)
            mitigation_description = result_mitigation["answer"]

            st.success("Risiko dan Mitigasi berhasil diambil!")

# Output fields for the results (right column)
with col2:
    st.header("Hasil")
    if submit_button:
        st.subheader("Penjelasan Risiko Pekerjaan")
        st.text_area("Risiko Pekerjaan", value=risk_description, height=200)
        
        st.subheader("Mitigasi Risiko Pekerjaan")
        st.text_area("Mitigasi Risiko", value=mitigation_description, height=200)
    else:
        st.write("Silakan isi deskripsi pekerjaan dan lokasi pekerjaan sebelum mengirim.")

# Optional: Inject some custom CSS to reduce margins, font size, and avoid title cut-off
st.markdown(""" 
    <style>
        .main { padding-left: 10px; padding-right: 10px; }
        .block-container { padding-top: 2rem; padding-bottom: 1rem; }
        .css-1d391kg { margin-left: 0px; margin-right: 0px; }
        .css-12ttj6v { margin-left: 0px; margin-right: 0px; }
        h1 { margin-top: 0px; padding-top: 0px; font-size: 20px; }
        h2 { font-size: 18px; }
        p, li, .css-1d391kg, .css-12ttj6v { font-size: 14px; }
        .stTextInput input { font-size: 14px; }
        .stButton button { font-size: 14px; }
        .stTextArea textarea { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)
