import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
from langchain_groq import ChatGroq

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

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
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
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
        return f"**Your Question:** {query}\n\n**Answer:** {answer}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(label="Your Question", placeholder="Type your question here..."),
    outputs=gr.Markdown(label="Answer"),
    title="Safety Procedure Assistant - UP Muara Karang",
    description="Berikan Deskripsi dan Lokasi Pekerjaan"
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
