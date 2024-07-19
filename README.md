## RAG Demo using Couchbase, Streamlit, LangChain, and OpenAI

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

The demo also caches the LLM responses using [CouchbaseCache](https://python.langchain.com/v0.2/docs/integrations/llm_caching/#couchbase-cache) to avoid repeated calls to the LLMs saving time and cost. You need to specify just the collection (in the same scope and bucket for simplicity) to cache the LLM responses.

For the full tutorial, please visit [Developer Portal](https://developer.couchbase.com/tutorial-python-langchain-pdf-chat).

> Note that you need Couchbase Server 7.6 or higher for Vector Search.

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.

For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

For RAG, we are using LangChain, Couchbase Vector Search & OpenAI. We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.

All LLM responses are cached in the collection specified. If the same exact question is asked again, the results are fetched from the Cache instead of calling the LLM.

> Note: The streaming of Cached responses is purely for visual experience as OpenAI integration cannot stream responses from the Cache due to a known [issue](https://github.com/langchain-ai/langchain/issues/9762).

### How to Run

- #### Install dependencies

  `pip install -r requirements.txt`

- #### Set the environment secrets

  Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment

  ```
  OPENAI_API_KEY = "<open_ai_api_key>"
  DB_CONN_STR = "<connection_string_for_couchbase_cluster>"
  DB_USERNAME = "<username_for_couchbase_cluster>"
  DB_PASSWORD = "<password_for_couchbase_cluster>"
  DB_BUCKET = "<name_of_bucket_to_store_documents>"
  DB_SCOPE = "<name_of_scope_to_store_documents>"
  DB_COLLECTION = "<name_of_collection_to_store_documents>"
  CACHE_COLLECTION = "<name_of_collection_to_cache_llm_responses>"
  INDEX_NAME = "<name_of_fts_index_with_vector_support>"
  AUTH_ENABLED = "True/False" # enables authentication for the streamlit app using LOGIN_PASSWORD
  LOGIN_PASSWORD = "<password to access the streamlit app>"
  ```

- #### Create the Search Index on Full Text Service

  We need to create the Search Index on the Full Text Service in Couchbase. For this demo, you can import the following index using the instructions.

  - [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

    - Copy the index definition to a new file index.json
    - Import the file in Capella using the instructions in the documentation.
    - Click on Create Index to create the index.

  - [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

    - Click on Search -> Add Index -> Import
    - Copy the following Index definition in the Import screen
    - Click on Create Index to create the index.

  #### Index Definition

  Here, we are creating the index `pdf_search` on the documents in the `docs` collection within the `shared` scope in the bucket `pdf-docs`. The Vector field is set to `embeddings` with 1536 dimensions and the text field set to `text`. We are also indexing and storing all the fields under `metadata` in the document as a dynamic mapping to account for varying document structures. The similarity metric is set to `dot_product`. If there is a change in these parameters, please adapt the index accordingly.

  ```
  {
    "name": "pdf_search",
    "type": "fulltext-index",
    "params": {
        "doc_config": {
            "docid_prefix_delim": "",
            "docid_regexp": "",
            "mode": "scope.collection.type_field",
            "type_field": "type"
        },
        "mapping": {
            "default_analyzer": "standard",
            "default_datetime_parser": "dateTimeOptional",
            "default_field": "_all",
            "default_mapping": {
                "dynamic": true,
                "enabled": false
            },
            "default_type": "_default",
            "docvalues_dynamic": false,
            "index_dynamic": true,
            "store_dynamic": false,
            "type_field": "_type",
            "types": {
                "shared.docs": {
                    "dynamic": true,
                    "enabled": true,
                    "properties": {
                        "embedding": {
                            "enabled": true,
                            "dynamic": false,
                            "fields": [
                                {
                                    "dims": 1536,
                                    "index": true,
                                    "name": "embedding",
                                    "similarity": "dot_product",
                                    "type": "vector",
                                    "vector_index_optimized_for": "recall"
                                }
                            ]
                        },
                        "text": {
                            "enabled": true,
                            "dynamic": false,
                            "fields": [
                                {
                                    "index": true,
                                    "name": "text",
                                    "store": true,
                                    "type": "text"
                                }
                            ]
                        }
                    }
                }
            }
        },
        "store": {
            "indexType": "scorch",
            "segmentVersion": 16
        }
    },
    "sourceType": "gocbcore",
    "sourceName": "pdf-docs",
    "sourceParams": {},
    "planParams": {
        "maxPartitionsPerPIndex": 64,
        "indexPartitions": 16,
        "numReplicas": 0
    }
  }
  ```

- #### Run the application

  `streamlit run chat_with_pdf.py`
