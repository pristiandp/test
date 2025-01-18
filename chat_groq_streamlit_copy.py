import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
from langchain_groq import ChatGroq

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = ChatGroq(
    groq_api_key="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",
    model_name='llama-3.3-70b-specdec'
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
        ("system", "You are a helpful assistant. Answer the question based only the data provided in bahasa indonesia."),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit App
st.set_page_config(page_title="Question and Answer System", page_icon=":robot_face:", layout="wide")

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("Adjust the settings below:")

# Main content
st.title("Question and Answer System")
st.subheader("Ask a question and get an answer based on the provided context.")
st.markdown("""
    This application uses a language model to retrieve answers based on the context provided.
    Simply type your question in the input box below and press Submit.
""")

# Input field for the question
query = st.text_input("Your Question:", placeholder="Type your question here...")

# Handle submission
if st.button("Submit") and query.strip():
    with st.spinner("Retrieving answer..."):
        result = retrieval_chain.invoke({"input": query})
        answer = result["answer"]
        st.success("Answer retrieved!")
        st.markdown(f"**Your Question:** {query}")
        st.markdown(f"**Answer:** {answer}")
else:
    st.write("Type your question above and press Submit.")
