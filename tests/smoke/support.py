"""Shared helpers for the smoke tests (importable without side effects)."""

from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Marker text embedded in the smoke-test PDF. The fake LLM reports whether it
# saw it in the prompt, which proves the PDF went through the retriever.
PDF_MARKER = "SMOKE-MARKER-7f3a"


class FakeCluster:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def wait_until_ready(self, *args: Any, **kwargs: Any) -> None:
        pass


class FakeChatModel(BaseChatModel):
    """Answers deterministically and reports whether the PDF context was used."""

    @property
    def _llm_type(self) -> str:
        return "smoke-fake-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if PDF_MARKER in prompt:
            answer = f"Fake LLM answer using PDF context ({PDF_MARKER})."
        else:
            answer = "Fake LLM answer without PDF context."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


def make_pdf(path: Path, text: str) -> Path:
    """Write a minimal single-page PDF containing `text`."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path
