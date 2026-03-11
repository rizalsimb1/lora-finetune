"""LLM-based answer generator with context injection."""
from typing import Generator, List


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always ground your answers in the context. If the context doesn't contain the answer, say so clearly.
Be concise, accurate, and cite sources when possible."""

RAG_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


class AnswerGenerator:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.model = llm_model
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            self.client = None

    def _build_context(self, docs) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.content}"
            for d in docs
        )

    def generate(self, question: str, context_docs: list) -> str:
        if not self.client:
            return f"[LLM not configured] Context retrieved for: {question}"
        context = self._build_context(context_docs)
        prompt = RAG_TEMPLATE.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def stream(self, question: str, context_docs: list) -> Generator[str, None, None]:
        if not self.client:
            yield f"[LLM not configured] Context retrieved for: {question}"
            return
        context = self._build_context(context_docs)
        prompt = RAG_TEMPLATE.format(context=context, question=question)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
