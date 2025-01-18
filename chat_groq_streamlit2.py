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

# Define the chat prompts
risk_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a safety expert. Based on the provided context, identify potential risks."),
        ("human", "Given the context: {context}, what are the potential risks? Use knowledge from the database or general expertise.")
    ]
)

mitigation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a safety mitigation expert. Suggest mitigations for the identified risks."),
        ("human", "Based on the identified risks: {risks}, suggest appropriate mitigation measures.")
    ]
)

# Define the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(llm, risk_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit App
st.set_page_config(page_title="Doctor JSA", layout="wide")
st.title("Doctor JSA")
st.write("\n**Analyze job safety risks and suggest mitigations**")

# Input fields for job description and location
col1, col2 = st.columns(2)
with col1:
    description = st.text_area("Job Description:", placeholder="Describe the job details here...", height=150)
    location = st.text_input("Location:", placeholder="Enter the job location...")

# Handle submission
if st.button("Analyze") and description.strip() and location.strip():
    context = f"Job Description: {description}\nLocation: {location}"
    with st.spinner("Analyzing risks and mitigations..."):
        # Generate risks
        risk_result = retrieval_chain.invoke({"context": context})
        risks = risk_result["answer"]
        
        # Generate mitigations
        mitigation_chain = create_stuff_documents_chain(llm, mitigation_prompt)
        mitigation_result = mitigation_chain.invoke({"risks": risks})
        mitigations = mitigation_result["answer"]
        
    # Display outputs
    with col2:
        st.markdown("**Risks Identified:**")
        st.write(f"<p style='font-size:12px;'>{risks}</p>", unsafe_allow_html=True)

        st.markdown("**Mitigation Suggestions:**")
        st.write(f"<p style='font-size:12px;'>{mitigations}</p>", unsafe_allow_html=True)
else:
    with col2:
        st.markdown("**Risks Identified:**")
        st.write("<p style='font-size:12px;'>No data provided yet.</p>", unsafe_allow_html=True)

        st.markdown("**Mitigation Suggestions:**")
        st.write("<p style='font-size:12px;'>No data provided yet.</p>", unsafe_allow_html=True)