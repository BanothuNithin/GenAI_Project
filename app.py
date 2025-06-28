import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Load .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for valid API key
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "GOOGLE_API_KEY":
    st.error("❌ GOOGLE_API_KEY not found or invalid in your .env file.")
    st.stop()

# Streamlit UI
st.title("News Research Tool 📈")
st.markdown('By Nithin-GenAI 👨‍💻 ', unsafe_allow_html=True)
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_google.pkl"
main_placeholder = st.empty()

# Google Generative AI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=512,
)

# Process URLs
if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅") 
    docs = splitter.split_documents(data)
    main_placeholder.text("📄 Text split complete!")

    # Embeddings with Google API
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore.serialize_to_bytes()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2) 

    # Save index
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)
    main_placeholder.success("✅ Vector index created and saved.")

# Ask a question
query = st.text_input("❓ Ask a question based on the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("❌ No FAISS index found. Please process URLs first.")
    else:
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
        vectorstore = FAISS.deserialize_from_bytes(
            embeddings=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            ),
            serialized=pkl,
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain.invoke({"question": query})

        st.header("📤 Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("🔗 Sources")
            for src in result["sources"].split("\n"):
                st.write(src)
