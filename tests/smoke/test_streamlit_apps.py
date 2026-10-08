"""Streamlit + Playwright smoke tests for both RAG demo entrypoints.

These run without Couchbase or OpenAI credentials. Live RAG validation is
still manual; see AGENTS.md.
"""

import re

import pytest
from playwright.sync_api import Page, expect

from support import PDF_MARKER, make_pdf

APPS = {
    "chat_with_pdf_query.py": "Chat with PDF (Query Vector Store)",
    "chat_with_pdf.py": "Chat with PDF (Search Vector Store)",
}
TIMEOUT_MS = 30_000


def wait_for_app(page: Page, url: str):
    page.goto(url)
    expect(page.get_by_test_id("stApp")).to_be_visible(timeout=TIMEOUT_MS)
    # Streamlit shows a "Running..." status widget while the script executes.
    expect(page.get_by_test_id("stStatusWidget")).to_have_count(0, timeout=TIMEOUT_MS)


def assert_no_errors(page: Page, server):
    expect(page.get_by_test_id("stException")).to_have_count(0)
    expect(page.get_by_text("Traceback (most recent call last)")).to_have_count(0)
    assert "Traceback (most recent call last)" not in server.log(), server.log()


@pytest.mark.parametrize("entrypoint", APPS)
def test_unconfigured_app_reports_missing_settings(page, start_app, entrypoint):
    """Real dependencies import and the app fails gracefully without settings."""
    server = start_app(entrypoint, stubbed=False)
    wait_for_app(page, server.url)

    expect(
        page.get_by_text("OPENAI_API_KEY environment variable is not set")
    ).to_be_visible(timeout=TIMEOUT_MS)
    assert_no_errors(page, server)


@pytest.mark.parametrize("entrypoint", APPS)
def test_auth_gate_renders(page, start_app, entrypoint):
    server = start_app(entrypoint, stubbed=True, env={"AUTH_ENABLED": "True"})
    wait_for_app(page, server.url)

    page.get_by_label("Enter password").fill("wrong-password")
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_text("Incorrect password")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text(APPS[entrypoint])).to_have_count(0)
    assert_no_errors(page, server)


@pytest.mark.parametrize("entrypoint", APPS)
def test_stubbed_app_upload_and_chat(page, start_app, entrypoint, tmp_path):
    """Full UI flow with Couchbase/OpenAI replaced by in-memory fakes."""
    server = start_app(entrypoint, stubbed=True)
    wait_for_app(page, server.url)

    # Expected controls and text render.
    expect(page.get_by_role("heading", name=APPS[entrypoint])).to_be_visible()
    expect(page.get_by_text("Upload your PDF")).to_be_visible()
    expect(page.get_by_role("button", name="Upload", exact=True)).to_be_visible()
    expect(page.get_by_text("How does it work?")).to_be_visible()
    expect(
        page.get_by_text("Hi, I'm a chatbot who can chat with the PDF.")
    ).to_be_visible()
    chat_input = page.get_by_placeholder("Ask a question based on the PDF")
    expect(chat_input).to_be_visible()
    assert_no_errors(page, server)

    # Upload a PDF: exercises pypdf loading, text splitting and indexing.
    pdf = make_pdf(tmp_path / "smoke.pdf", f"The secret code is {PDF_MARKER}.")
    page.locator('input[type="file"]').set_input_files(pdf)
    expect(page.get_by_text("smoke.pdf")).to_be_visible(timeout=TIMEOUT_MS)
    page.get_by_role("button", name="Upload", exact=True).click()
    expect(
        page.get_by_text(re.compile(r"PDF loaded into vector store in \d+ documents"))
    ).to_be_visible(timeout=TIMEOUT_MS)

    # Ask a question: RAG answer must include retrieved PDF context, the
    # pure-LLM answer must not.
    chat_input.fill("What is the secret code?")
    chat_input.press("Enter")
    expect(page.get_by_text("What is the secret code?")).to_be_visible(timeout=TIMEOUT_MS)
    expect(
        page.get_by_text(f"Fake LLM answer using PDF context ({PDF_MARKER}).")
    ).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("Fake LLM answer without PDF context.")).to_be_visible(
        timeout=TIMEOUT_MS
    )
    assert_no_errors(page, server)
