# AGENTS.md

Notes for contributors and coding agents working on this repo.

## Layout

- `chat_with_pdf_query.py`: default app. Uses `CouchbaseQueryVectorStore` (Query + Index services, Hyperscale/Composite vector indexes).
- `chat_with_pdf.py`: uses `CouchbaseSearchVectorStore` (Search service, needs `INDEX_NAME`).
- `tests/smoke/`: Streamlit + Playwright smoke tests for both entrypoints (no credentials).
- `tests/provider/`: optional live OpenAI provider smoke test (needs `OPENAI_API_KEY`, no Couchbase).
- `.github/workflows/smoke.yml`: runs the smoke tests on every PR and push to `main`, plus the optional provider job.

## Test tiers

| Tier | What | When | Required? |
| --- | --- | --- | --- |
| 1 | `tests/smoke` Streamlit + Playwright smoke, no secrets | Every PR, push to `main`, manual dispatch | Yes. The required PR check |
| 2 | `tests/provider` live OpenAI provider smoke | CI job `openai-provider-smoke` when `OPENAI_API_KEY` is available; skips with a notice otherwise | No. Optional signal |
| 3 | Full Couchbase-backed RAG run (checklist below) | Manually, by a maintainer, before merging dependency updates | Yes. Evidence goes in the PR |

Tiers 1 and 2 do not replace tier 3.

## Install

```bash
pip install -r requirements-dev.txt          # app deps + pytest + playwright
python -m playwright install --with-deps chromium
```

`requirements.txt` holds only the app's runtime dependencies. Dependabot bumps both files.

## Smoke tests (CI, no credentials)

```bash
python -m pytest tests/smoke -v
```

These tests need no Couchbase cluster and no OpenAI key. They start each app headless on a free port, open it in headless Chromium, and fail if a Python traceback or `st.exception` appears in the page or the server log. For each entrypoint they check:

1. **Unconfigured start**: the real app with no settings loads, imports every real dependency, and shows the `OPENAI_API_KEY environment variable is not set` message.
2. **Auth gate**: with `AUTH_ENABLED=True`, the password form renders and rejects a wrong password.
3. **Stubbed full flow**: `tests/smoke/stubbed_app.py` runs the app with the Couchbase cluster, vector store, and cache, plus the OpenAI embeddings and chat model, swapped for in-memory LangChain fakes. The test checks that the expected controls render, uploads a small generated PDF (real `pypdf` loader and text splitter), asks a question, and asserts that the RAG answer used the retrieved PDF context and the pure-LLM answer did not.

The tests set placeholder values for every app setting and use an isolated `HOME`, so a local `.streamlit/secrets.toml` or real credentials in your shell are never used.

**What the smoke tests do not cover:** real Couchbase connectivity, vector index definitions, `langchain-couchbase` query/search behaviour against a cluster, OpenAI API compatibility (model names, request shapes), and the LLM cache against a real collection. The optional provider smoke covers the OpenAI part. The rest is why live validation stays manual (below).

## OpenAI provider smoke (optional, needs `OPENAI_API_KEY`)

```bash
OPENAI_API_KEY=... python -m pytest tests/provider -v -rs
```

This calls the real OpenAI API and does not touch Couchbase. It reads the `OpenAIEmbeddings(...)` and `ChatOpenAI(...)` arguments from both app entrypoints, so it checks the exact models and options the apps use:

- **Embeddings**: `embed_query` returns a 1536-dim vector of finite floats (the README index definitions use 1536 dims).
- **Chat**: each distinct `ChatOpenAI` configuration returns a non-empty reply to a one-line prompt.

Without `OPENAI_API_KEY` the tests are skipped with a clear reason. With an invalid key or a retired model they fail. In CI the `openai-provider-smoke` job uses the `OPENAI_API_KEY` Actions secret. Secrets are not passed to fork or Dependabot PRs, so there the job posts a "skipped" notice and passes. To check the live OpenAI path for such a PR, run the workflow manually (**Actions → Streamlit smoke tests → Run workflow**) on the PR branch.

## Manual live validation for dependency-update PRs

A green smoke run is necessary but not sufficient to merge a dependency bump (for example `langchain-*`, `couchbase`, `streamlit`, `pypdf`). Before merging, a maintainer runs the app against a real cluster and records this evidence in the PR:

1. **App boot**: `streamlit run chat_with_pdf_query.py` (and `streamlit run chat_with_pdf.py` if the bump touches `langchain-couchbase`, `couchbase`, or Search) starts with no errors, and the installed versions match the PR (`pip freeze | grep -iE "langchain|couchbase|streamlit|pypdf|openai"`).
2. **Couchbase / vector index state**: the cluster version, the bucket/scope/collection used, and that the index exists. For the Query app that is the Hyperscale/Composite vector index from the README's "Index Configuration" section. For the Search app it is the Search index named by `INDEX_NAME` (see "Create the Search Index").
3. **Provider credentials**: confirm that `OPENAI_API_KEY` was available and that the configured models were accessible. State only that it was set, never the value.
4. **Sample input / query**: the PDF you uploaded (name and approximate size) and the question you asked.
5. **Observed response**: the RAG answer and the pure-LLM answer. Ask the same question a second time to confirm the cached response is returned.
6. **Screenshots or traces** of the upload confirmation and the chat answers, when practical.

## Required settings / secret names

Set these in `.streamlit/secrets.toml` (copy `.streamlit/secrets.example.toml`) or as environment variables. Never commit real values, and never paste them into PRs, issues, or logs.

| Name | Used by |
| --- | --- |
| `OPENAI_API_KEY` | both apps |
| `DB_CONN_STR`, `DB_USERNAME`, `DB_PASSWORD` | both apps |
| `DB_BUCKET`, `DB_SCOPE`, `DB_COLLECTION`, `CACHE_COLLECTION` | both apps |
| `INDEX_NAME` | `chat_with_pdf.py` only |
| `AUTH_ENABLED`, `LOGIN_PASSWORD` | optional app password gate |

The required CI smoke job needs **no** repository secrets. The optional `openai-provider-smoke` job uses the `OPENAI_API_KEY` Actions secret when it is available.
