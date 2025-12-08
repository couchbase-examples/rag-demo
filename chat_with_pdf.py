import tempfile
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_couchbase.cache import CouchbaseCache
import time
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta


def parse_bool(value: str):
    """Parse boolean values from environment variables"""
    return value.lower() in ("yes", "true", "t", "1")


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"
        )
        st.stop()


def save_to_vector_store(uploaded_file, vector_store):
    """Chunk the PDF & store it in Couchbase Vector Store"""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )

        doc_pages = text_splitter.split_documents(docs)

        vector_store.add_documents(doc_pages)
        st.info(f"PDF loaded into vector store in {len(doc_pages)} documents")


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    _embedding,
    index_name,
):
    """Return the Couchbase vector store"""
    vector_store = CouchbaseSearchVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        index_name=index_name,
    )
    return vector_store


@st.cache_resource(show_spinner="Connecting to Cache")
def get_cache(_cluster, db_bucket, db_scope, cache_collection):
    """Return the Couchbase cache"""
    cache = CouchbaseCache(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=cache_collection,
    )
    return cache


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    cluster = Cluster(connection_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


def stream_string(s, chunk_size=10):
    """Stream a string with a delay to simulate streaming"""
    for i in range(0, len(s), chunk_size):
        yield s[i : i + chunk_size]
        time.sleep(0.02)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Chat with your PDF using Langchain, Couchbase & OpenAI",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
    else:
        # Authorization
        if "auth" not in st.session_state:
            st.session_state.auth = False

        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        # Authentication
        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")

    if st.session_state.auth:
        # Load environment variables
        DB_CONN_STR = os.getenv("DB_CONN_STR")
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_BUCKET = os.getenv("DB_BUCKET")
        DB_SCOPE = os.getenv("DB_SCOPE")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        INDEX_NAME = os.getenv("INDEX_NAME")
        CACHE_COLLECTION = os.getenv("CACHE_COLLECTION")

        # Ensure that all environment variables are set
        required_env_vars = [
            "OPENAI_API_KEY",
            "DB_CONN_STR",
            "DB_USERNAME",
            "DB_PASSWORD",
            "DB_BUCKET",
            "DB_SCOPE",
            "DB_COLLECTION",
            "INDEX_NAME",
            "CACHE_COLLECTION",
        ]
        for var in required_env_vars:
            check_environment_variable(var)

        # Use OpenAI Embeddings
        embedding = OpenAIEmbeddings()

        # Connect to Couchbase Vector Store
        cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)

        vector_store = get_vector_store(
            cluster,
            DB_BUCKET,
            DB_SCOPE,
            DB_COLLECTION,
            embedding,
            INDEX_NAME,
        )

        # Use couchbase vector store as a retriever for RAG
        retriever = vector_store.as_retriever()

        # Set the LLM cache
        cache = get_cache(cluster, DB_BUCKET, DB_SCOPE, CACHE_COLLECTION)
        set_llm_cache(cache)

        # Build the prompt for the RAG
        template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        # Use OpenAI GPT 4 as the LLM for the RAG
        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

        # RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Pure OpenAI output without RAG
        template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

        Question: {question}"""

        prompt_without_rag = ChatPromptTemplate.from_template(template_without_rag)

        llm_without_rag = ChatOpenAI(model="gpt-4-1106-preview", streaming=True)

        chain_without_rag = (
            {"question": RunnablePassthrough()}
            | prompt_without_rag
            | llm_without_rag
            | StrOutputParser()
        )

        # Frontend
        couchbase_logo = (
            "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"
        )

        st.title("Chat with PDF (Search Vector Store)")
        st.markdown(
            "Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* while 🤖️ are generated by pure *LLM (ChatGPT)*"
        )

        with st.sidebar:
            st.header("Upload your PDF")
            with st.form("upload pdf"):
                uploaded_file = st.file_uploader(
                    "Choose a PDF.",
                    help="The document will be deleted after one hour of inactivity (TTL).",
                    type="pdf",
                )
                submitted = st.form_submit_button("Upload")
                if submitted:
                    # store the PDF in the vector store after chunking
                    save_to_vector_store(uploaded_file, vector_store)

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you will get two answers: 
                * one using RAG ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
                * one using pure LLM - OpenAI (🤖️). 
                """
            )

            st.markdown(
                "For RAG, we are using [Langchain](https://langchain.com/), [Couchbase Vector Search using Search Service](https://couchbase.com/) & [OpenAI](https://openai.com/). We fetch parts of the PDF relevant to the question using Vector Search using the Search (FTS) Service and add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store."
            )

            # View Code
            if st.checkbox("View Code"):
                st.write(
                    "View the code here: [Github](https://github.com/couchbase-examples/rag-demo/blob/main/chat_with_pdf.py)"
                )

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                    "avatar": "🤖️",
                }
            )

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        # React to user input
        if question := st.chat_input("Ask a question based on the PDF"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": "👤"}
            )

            # Add placeholder for streaming the response
            with st.chat_message("assistant", avatar=couchbase_logo):
                # Get the response from the RAG & stream it
                # In order to cache the response, we need to invoke the chain and cache the response locally as OpenAI does not support it yet
                # Ref: https://github.com/langchain-ai/langchain/issues/9762

                rag_response = chain.invoke(question)

                st.write_stream(stream_string(rag_response))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    "avatar": couchbase_logo,
                }
            )

            # Get the response from the pure LLM & stream it
            pure_llm_response = chain_without_rag.invoke(question)

            # Add placeholder for streaming the response
            with st.chat_message("ai", avatar="🤖️"):
                st.write_stream(stream_string(pure_llm_response))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": pure_llm_response,
                    "avatar": "🤖️",
                }
            )
