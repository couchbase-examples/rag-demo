## RAG Demo using Couchbase, Streamlit, LangChain, and OpenAI

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

The demo also caches the LLM responses using [CouchbaseCache](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#module-langchain_couchbase.cache ) to avoid repeated calls to the LLMs saving time and cost. You need to specify just the collection (in the same scope and bucket for simplicity) to cache the LLM responses.

## Two Vector Search Implementations

This demo provides two implementations showcasing different Couchbase vector search approaches:

1. **CouchbaseQueryVectorStore** (`chat_with_pdf_query.py`) - Using Couchbase Vector Search (Hyperscale/Composite Vector Indexes) with the Query and Indexing Services.
2. **CouchbaseSearchVectorStore** (`chat_with_pdf.py`) - Using Couchbase Search (formerly known as Full Text Search) Service.


### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.

For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

For RAG, we are using LangChain, Couchbase Vector Search & OpenAI. We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.

All LLM responses are cached in the collection specified. If the same exact question is asked again, the results are fetched from the Cache instead of calling the LLM.

> Note: The streaming of Cached responses is purely for visual experience as OpenAI integration cannot stream responses from the Cache due to a known [issue](https://github.com/langchain-ai/langchain/issues/9762).



## Setup Instructions

### Install dependencies

  `pip install -r requirements.txt`

### Set the environment secrets

Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment.

**For Couchbase Vector Search - Hyperscale/Composite (`chat_with_pdf_query.py`):**
```toml
OPENAI_API_KEY = "<open_ai_api_key>"
DB_CONN_STR = "<connection_string_for_couchbase_cluster>"
DB_USERNAME = "<username_for_couchbase_cluster>"
DB_PASSWORD = "<password_for_couchbase_cluster>"
DB_BUCKET = "<name_of_bucket_to_store_documents>"
DB_SCOPE = "<name_of_scope_to_store_documents>"
DB_COLLECTION = "<name_of_collection_to_store_documents>"
CACHE_COLLECTION = "<name_of_collection_to_cache_llm_responses>"
AUTH_ENABLED = "False"
LOGIN_PASSWORD = "<password_to_access_the_streamlit_app>"
```

**For Couchbase Search (`chat_with_pdf.py`):**
```toml
OPENAI_API_KEY = "<open_ai_api_key>"
DB_CONN_STR = "<connection_string_for_couchbase_cluster>"
DB_USERNAME = "<username_for_couchbase_cluster>"
DB_PASSWORD = "<password_for_couchbase_cluster>"
DB_BUCKET = "<name_of_bucket_to_store_documents>"
DB_SCOPE = "<name_of_scope_to_store_documents>"
DB_COLLECTION = "<name_of_collection_to_store_documents>"
CACHE_COLLECTION = "<name_of_collection_to_cache_llm_responses>"
INDEX_NAME = "<name_of_search_index_with_vector_support>"
AUTH_ENABLED = "False"
LOGIN_PASSWORD = "<password_to_access_the_streamlit_app>"
```

> **Note:** Couchbase Vector Search approach does not require `INDEX_NAME` parameter.



## Approach 1: Couchbase Vector Search (Hyperscale/Composite)

For the full tutorial on Couchbase Vector Search approach, please visit [Developer Portal - Couchbase Vector Search](https://developer.couchbase.com/tutorial-python-langchain-pdf-chat-query).

### Prerequisites
- Couchbase Server 8.0+ or Couchbase Capella

This approach uses `CouchbaseQueryVectorStore` which leverages Couchbase's Hyperscale and Composite Vector Indexes (built on Global Secondary Index infrastructure). The vector search is performed using SQL++ queries with cosine similarity distance metric.

### Understanding Vector Index Types

Couchbase offers different types of vector indexes for Couchbase Vector Search:

**Hyperscale Vector Indexes (BHIVE)**
- Best for pure vector searches - content discovery, recommendations, semantic search
- High performance with low memory footprint - designed to scale to billions of vectors
- Optimized for concurrent operations - supports simultaneous searches and inserts
- Use when: You primarily perform vector-only queries without complex scalar filtering
- Ideal for: Large-scale semantic search, recommendation systems, content discovery

**Composite Vector Indexes**
- Best for filtered vector searches - combines vector search with scalar value filtering
- Efficient pre-filtering - scalar attributes reduce the vector comparison scope
- Use when: Your queries combine vector similarity with scalar filters that eliminate large portions of data
- Ideal for: Compliance-based filtering, user-specific searches, time-bounded queries

**Choosing the Right Index Type**
- Start with Hyperscale Vector Index for pure vector searches and large datasets
- Use Composite Vector Index when scalar filters significantly reduce your search space
- Consider your dataset size: Hyperscale scales to billions, Composite works well for tens of millions to billions

For more details, see the [Couchbase Vector Index documentation](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html).

### Index Configuration (Optional)

While the application works without creating indexes manually, you can optionally create a vector index for better performance.

> **Important:** The vector index should be created **after** ingesting the documents (uploading PDFs).

**Using LangChain:**

You can create the index programmatically after uploading your PDFs:

```python
from langchain_couchbase.vectorstores import IndexType

# Create a vector index on the collection
vector_store.create_index(
    index_name="idx_vector",
    dimension=1536,
    similarity="cosine",
    index_type=IndexType.BHIVE,  # or IndexType.COMPOSITE
    index_description="IVF,SQ8"
)
```

For more details on the `create_index()` method, see the [LangChain Couchbase API documentation](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#langchain_couchbase.vectorstores.query_vector_store.CouchbaseQueryVectorStore.create_index).

**Understanding Index Configuration Parameters:**

The `description` parameter controls how Couchbase optimizes vector storage and search performance:

**Format:** `'IVF[<centroids>],{PQ|SQ}<settings>'`

**Centroids (IVF - Inverted File):**
- Controls how the dataset is subdivided for faster searches
- More centroids = faster search, slower training
- Fewer centroids = slower search, faster training
- If omitted (like `IVF,SQ8`), Couchbase auto-selects based on dataset size

**Quantization Options:**
- **SQ (Scalar Quantization)**: `SQ4`, `SQ6`, `SQ8` (4, 6, or 8 bits per dimension)
- **PQ (Product Quantization)**: `PQ<subquantizers>x<bits>` (e.g., `PQ32x8`)
- Higher values = better accuracy, larger index size

**Common Examples:**
- `IVF,SQ8` - Auto centroids, 8-bit scalar quantization (good default)
- `IVF1000,SQ6` - 1000 centroids, 6-bit scalar quantization
- `IVF,PQ32x8` - Auto centroids, 32 subquantizers with 8 bits

For detailed configuration options, see the [Quantization & Centroid Settings](https://docs.couchbase.com/server/current/vector-index/hyperscale-vector-index.html#algo_settings).

> **Note:** In Couchbase Vector Search, the distance represents the vector distance between the query and document embeddings. Lower distance indicates higher similarity, while higher distance indicates lower similarity. This demo uses cosine similarity for measuring document relevance.

### Run the Couchbase Vector Search application

```bash
streamlit run chat_with_pdf_query.py
```


## Approach 2: Couchbase Search

For the full tutorial on Couchbase Search approach, please visit [Developer Portal - Couchbase Search](https://developer.couchbase.com/tutorial-python-langchain-pdf-chat).

### Prerequisites
- Couchbase Server 7.6+ or Couchbase Capella

### Create the Search Index

We need to create the Search Index in Couchbase. For this demo, you can import the following index using the instructions.

- [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

  - Copy the index definition to a new file index.json
  - Import the file in Capella using the instructions in the documentation.
  - Click on Create Index to create the index.

- [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

  - Click on Search -> Add Index -> Import
  - Copy the following Index definition in the Import screen
  - Click on Create Index to create the index.

#### Index Definition

Here, we are creating the index `pdf_search` on the documents in the `docs` collection within the `shared` scope in the bucket `pdf-docs`. The Vector field is set to `embedding` with 1536 dimensions and the text field set to `text`. We are also indexing and storing all the fields under `metadata` in the document as a dynamic mapping to account for varying document structures. The similarity metric is set to `dot_product`. If there is a change in these parameters, please adapt the index accordingly.

```json
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

### Run the Couchbase Search application

```bash
streamlit run chat_with_pdf.py
```