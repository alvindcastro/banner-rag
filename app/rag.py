"""
app/rag.py
Core RAG logic:
  1. Embed the user's question
  2. Hybrid search (vector + keyword) against Azure AI Search
  3. Build a grounded prompt with retrieved chunks
  4. Call Azure OpenAI GPT-4o for the final answer
"""
import logging
from typing import Optional

from azure.search.documents.models import VectorizedQuery
from tenacity import retry, stop_after_attempt, wait_exponential

from app.azure_clients import get_openai_client, get_search_client, embed_text
from app.config import settings
from app.models import AskRequest, AskResponse, SourceChunk

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Ellucian Banner ERP upgrade assistant for a higher-education institution.
Your job is to help IT staff and functional analysts answer questions about Banner module upgrades,
patch releases, prerequisites, known issues, configuration steps, and compatibility.

Rules:
- Answer ONLY using the provided context chunks from Banner release notes and documentation.
- If the context does not contain enough information to answer, say so clearly — do NOT hallucinate.
- When referencing specific steps or requirements, cite the source document name.
- Be concise but thorough. Use numbered lists for multi-step procedures.
- If a version or module is mentioned in the question, focus your answer on that version/module.
"""

# ─── Filters ─────────────────────────────────────────────────────────────────

def _build_filter(version_filter: Optional[str], module_filter: Optional[str]) -> Optional[str]:
    """Build an OData filter string for Azure AI Search."""
    filters = []
    if version_filter:
        filters.append(f"banner_version eq '{version_filter}'")
    if module_filter:
        filters.append(f"banner_module eq '{module_filter}'")
    return " and ".join(filters) if filters else None


# ─── Retrieval ────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def retrieve_chunks(
    question: str,
    top_k: int = 5,
    version_filter: Optional[str] = None,
    module_filter: Optional[str] = None,
) -> list[SourceChunk]:
    """
    Hybrid search: combines vector similarity with keyword BM25 search.
    Returns a ranked list of source chunks.
    """
    client = get_search_client()

    # Embed the question for vector search
    question_vector = embed_text(question)

    vector_query = VectorizedQuery(
        vector=question_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    odata_filter = _build_filter(version_filter, module_filter)

    results = client.search(
        search_text=question,          # BM25 keyword leg
        vector_queries=[vector_query], # Vector leg
        filter=odata_filter,
        select=["id", "filename", "page_number", "banner_module", "banner_version", "chunk_text"],
        top=top_k,
    )

    chunks = []
    for r in results:
        chunks.append(
            SourceChunk(
                filename=r.get("filename", "unknown"),
                page=r.get("page_number"),
                banner_module=r.get("banner_module"),
                banner_version=r.get("banner_version"),
                chunk_text=r.get("chunk_text", ""),
                score=r.get("@search.score", 0.0),
            )
        )

    logger.info(f"Retrieved {len(chunks)} chunks for question: {question!r}")
    return chunks


# ─── Generation ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(question: str, chunks: list[SourceChunk]) -> str:
    """Send the question + retrieved context to GPT-4o and return the answer."""
    client = get_openai_client()

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = f"[{i}] {chunk.filename}"
        if chunk.page:
            source_label += f" (page {chunk.page})"
        if chunk.banner_module:
            source_label += f" | Module: {chunk.banner_module}"
        if chunk.banner_version:
            source_label += f" | Version: {chunk.banner_version}"
        context_parts.append(f"{source_label}\n{chunk.chunk_text}")

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""Use the following Banner documentation excerpts to answer the question.

=== CONTEXT ===
{context_text}

=== QUESTION ===
{question}

=== ANSWER ==="""

    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # Low temp for factual / grounded answers
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def ask(request: AskRequest) -> AskResponse:
    """
    Full RAG pipeline:
      question → embed → search → retrieve chunks → generate answer → return
    """
    chunks = retrieve_chunks(
        question=request.question,
        top_k=request.top_k,
        version_filter=request.version_filter,
        module_filter=request.module_filter,
    )

    if not chunks:
        return AskResponse(
            question=request.question,
            answer=(
                "I could not find any relevant documentation for your question. "
                "Please ensure Banner upgrade documents have been ingested into the knowledge base."
            ),
            sources=[],
            retrieval_count=0,
        )

    answer = generate_answer(request.question, chunks)

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=chunks,
        retrieval_count=len(chunks),
    )
