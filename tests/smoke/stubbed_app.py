"""Run a demo entrypoint with Couchbase and OpenAI replaced by in-memory fakes.

Used only by the smoke tests, so the full UI (PDF upload, retrieval, chat) can be
exercised in CI without a Couchbase cluster or OpenAI credentials:

    streamlit run tests/smoke/stubbed_app.py -- chat_with_pdf_query.py

Streamlit re-executes this script on every interaction, so the fakes are
re-applied before each run of the target app.
"""

import runpy
import sys
from typing import Any

import couchbase.cluster
import langchain_couchbase
import langchain_couchbase.cache
import langchain_couchbase.vectorstores
import langchain_openai
from langchain_core.caches import InMemoryCache
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from support import FakeChatModel, FakeCluster


def fake_vector_store(**kwargs: Any) -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding=kwargs["embedding"])


couchbase.cluster.Cluster = FakeCluster
langchain_couchbase.CouchbaseQueryVectorStore = fake_vector_store
langchain_couchbase.vectorstores.CouchbaseSearchVectorStore = fake_vector_store
langchain_couchbase.cache.CouchbaseCache = lambda **kwargs: InMemoryCache()
langchain_openai.OpenAIEmbeddings = lambda **kwargs: DeterministicFakeEmbedding(size=64)
langchain_openai.ChatOpenAI = lambda **kwargs: FakeChatModel()

runpy.run_path(sys.argv[1], run_name="__main__")
