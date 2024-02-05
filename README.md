## RAG Demo using Couchbase, Streamlit, Langchain, and OpenAI

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.

For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

For RAG, we are using Langchain, Couchbase Vector Search & OpenAI. We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.

### How to Run

- Install dependencies

  `pip install -r requirements.txt`

- Set the environment secrets
  Copy the secrets.example.toml file in `streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment

  ```
  OPENAI_API_KEY = "<open_ai_api_key"
  DB_CONN_STR = "<connection_string_for_couchbase_cluster"
  DB_USERNAME = "<username_for_couchbase_cluster>"
  DB_PASSWORD = "<password_for_couchbase_cluster>"
  DB_BUCKET = "<name_of_bucket_to_store_documents"
  DB_SCOPE = "<name_of_scope_to_store_documents"
  DB_COLLECTION = "<name_of_collection_to_store_documents>"
  INDEX_NAME = "<name_of_fts_index_with_vector_support>"
  LOGIN_PASSWORD = "<password to access the streamlit app"
  ```

- Run the application

  `streamlit run chat_with_pdf.py`
