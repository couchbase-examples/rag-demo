"""Optional live OpenAI provider smoke test. Needs OPENAI_API_KEY; no Couchbase.

The embedding and chat model settings are read from both app entrypoints, so
this checks the exact models and ChatOpenAI arguments the apps use.
"""

import ast
import math
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS = ["chat_with_pdf_query.py", "chat_with_pdf.py"]

# Both README index definitions use a 1536-dim vector field.
EMBEDDING_DIMS = 1536

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set; skipping live OpenAI provider smoke test",
)


def _constructor_calls(name):
    """Literal keyword arguments of every `name(...)` call in the apps."""
    calls = set()
    for app in APPS:
        for node in ast.walk(ast.parse((REPO_ROOT / app).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                calls.add(tuple(sorted(kwargs.items())))
    assert calls, f"no {name}(...) call found in {APPS}"
    return [dict(c) for c in sorted(calls)]


def _ids(calls):
    return [",".join(f"{k}={v}" for k, v in c.items()) or "defaults" for c in calls]


EMBEDDING_CALLS = _constructor_calls("OpenAIEmbeddings")
CHAT_CALLS = _constructor_calls("ChatOpenAI")


@pytest.mark.parametrize("kwargs", EMBEDDING_CALLS, ids=_ids(EMBEDDING_CALLS))
def test_embeddings_return_numeric_vector(kwargs):
    from langchain_openai import OpenAIEmbeddings

    vector = OpenAIEmbeddings(**kwargs).embed_query("Couchbase vector search")
    assert len(vector) == EMBEDDING_DIMS
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)
    assert any(x != 0 for x in vector)


@pytest.mark.parametrize("kwargs", CHAT_CALLS, ids=_ids(CHAT_CALLS))
def test_chat_model_returns_text(kwargs):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(**kwargs, max_retries=1)
    reply = llm.invoke("Reply with the single word: pong")
    assert isinstance(reply.content, str) and reply.content.strip()
