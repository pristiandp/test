import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
from langchain_groq import ChatGroq
from huggingface_hub import login

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = ChatGroq(
    groq_api_key="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",
    model_name='llama-3.2-3b-preview',
    temperature=0.1,
#gemma2-9b-it
#llama-3.3-70b-specdec
)


# Initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the question based only the data provided in bahasa indonesia. You will give risk and risk control of work deskription."),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question. To the point. Dalam bentuk point-point disesuaikan dengan peraturan menteri ketenagakerjaan. example: \n 1. *Deskripsi Pekerjaan*: Jelaskan secara singkat dan jelas dalam bentuk paragraf bukan point-point. 2. *Alat yang Digunakan*: misalnya: tangga, gerinda, dll Dalam bentuk point-point.. 3. *Identifikasi Bahaya*: misalnya: kebakaran, listrik, jatuh, dll Dalam bentuk point-point. 4. *Risiko Bahaya*: Apa dampak yang mungkin terjadi jika bahaya tersebut terjadi? Dalam bentuk point-point. 5. *Pengendalian Risiko*: Apa saja langkah-langkah yang akan diambil untuk mengendalikan risiko? Misalnya: menggunakan APD, penguncian peralatan, pelatihan keselamatan, dll Dalam bentuk point-point. 6. *Penggunaan APD*: APD yang diperlukan misalnya: helm, sepatu keselamatan, sarung tangan, masker, dll Dalam bentuk point-point.")
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Function to handle the question
def get_answer(query):
    if not query.strip():
        return "Masukkan pertanyaan yang valid."
    try:
        result = retrieval_chain.invoke({"input": query})
        answer = result["answer"]
        return f"**Deskripsi Pekerjaan:** {query}\n\n**Analisa Risiko:** {answer}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(label="Deskripsi Pekerjaan", placeholder="Tulis deskripsi pekerjaan dan lokasi pekerjaan disini..."),
    outputs=gr.Markdown(label="Analisa Risiko"),
    title="Job Safety Analysis (JSA) Virtual Assistant - UP Muara Karang",
    description="Berikan Deskripsi dan Lokasi Pekerjaan"
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)

# Login dengan API token
login("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")