"""
Retrieval-Augmented Generation (RAG) tool for Voice AI.
Supports topic guardrails, semantic search, verified link enrichment,
media availability notifications, and turn-efficient JIT instructions.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple, Union
from edgevoiceai.rag.chunk import DocumentChunk
from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext


class KnowledgeBaseTool(BaseTool):
    """
    Modular RAG tool querying a vector database with guardrail filtering and JIT guidance.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        retriever_func: Callable[..., Coroutine[Any, Any, Sequence[Union[Dict[str, Any], DocumentChunk]] | Sequence[Any]]],
        topic_guard_func: Optional[Callable[[str], Tuple[bool, str]]] = None,
        query_param: str = "query",
        limit: int = 3,
        score_threshold: float = 0.45,
        media_directory: str = "audio_experiences",
        resource_enricher_func: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        jit_instruction_builder: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        default_jit_instruction: Optional[str] = None,
        no_results_message: Optional[str] = None,
        prompt_instruction: Optional[str] = None,
    ):
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self.retriever_func = retriever_func
        self.topic_guard_func = topic_guard_func
        self.query_param = query_param
        self.limit = limit
        self.score_threshold = score_threshold
        self.media_directory = media_directory
        self.resource_enricher_func = resource_enricher_func
        self.jit_instruction_builder = jit_instruction_builder
        self.default_jit_instruction = default_jit_instruction or (
            "Based on the retrieved knowledge, answer the user's question clearly and concisely in 2-3 spoken sentences. "
            "Do not cite technical URLs or mention internal systems ('database', 'knowledge base') aloud."
        )
        self.no_results_message = no_results_message or (
            "No specific knowledge found on this topic. "
            "Please politely inform the user that this information is currently not available in the database."
        )
        self._prompt_instruction = prompt_instruction or (
            f"• KNOWLEDGE RETRIEVAL RULE (`{self._name}`):\n"
            f"  1. When the user asks factual, domain, or technical questions, call `{self._name}` to retrieve relevant knowledge.\n"
            "  2. Base your response strictly on the retrieved knowledge without inventing facts.\n"
            "  3. Synthesize the answer concisely in a natural conversational voice (1-3 sentences)."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._parameters_schema

    @property
    def prompt_instruction(self) -> Optional[str]:
        return self._prompt_instruction


    async def execute(self, args: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        query = args.get(self.query_param, "")
        print(f"\n🔍 [{self.name}] Searching knowledge base for: '{query}'")

        # ── 1. Topic Guardrail Check ──────────────────────────────────────────
        if self.topic_guard_func:
            is_related, off_topic_msg = self.topic_guard_func(query)
            if not is_related:
                print(f"⚠️  [{self.name} Guardrail] Off-topic query blocked: '{query}'")
                return ToolResult(
                    status="off_topic",
                    message=off_topic_msg,
                )

        # ── 2. Vector Search Retrieval ────────────────────────────────────────
        matches: List[Dict[str, Any]] = []
        try:
            raw_matches = await self.retriever_func(query, limit=self.limit, score_threshold=self.score_threshold)
            matches = [
                m.to_dict() if isinstance(m, DocumentChunk) else (m if isinstance(m, dict) else {"text": str(m)})
                for m in (raw_matches or [])
            ]
        except Exception as exc:
            print(f"⚠️  [{self.name}] Vector search error: {exc}")
            return ToolResult(
                status="error",
                message=f"Search error: {exc}",
            )

        client_evt: Optional[Dict[str, Any]] = None
        queued_audio: Optional[str] = None

        if matches:
            vstore_name = matches[0].get("vector_store", "Vector Store")
            print(f"✅ [{self.name}] Retrieved {len(matches)} chunks from {vstore_name}.")
            results_text = "\n\n".join([
                f"- [{m.get('title', '')}]: {m.get('text', '')}"
                for m in matches
            ])

            # Check if any result has an associated audio file that exists on disk
            for m in matches:
                audio_url = m.get("audio_url")
                if audio_url:
                    chk_fn = audio_url.replace("\\", "/").split("/")[-1]
                    chk_fp = os.path.join(self.media_directory, chk_fn)
                    if os.path.isfile(chk_fp):
                        client_evt = {
                            "type": "experience_available",
                            "farmer_name": m.get("farmer_name", "હસમુખભાઈ પટેલ"),
                            "district": m.get("district", "વલસાડ"),
                            "audio_url": audio_url,
                            "experience_years": m.get("experience_years", "૫ વર્ષ"),
                        }
                        queued_audio = audio_url
                        break

            # ── 3. Resource Enrichment (e.g., official book PDFs, verified videos) ──
            if self.resource_enricher_func:
                enrichment = self.resource_enricher_func(query, matches)
                if enrichment:
                    results_text += enrichment

            # ── 4. JIT Instruction Construction ──────────────────────────────
            if self.jit_instruction_builder:
                jit_inst = self.jit_instruction_builder(query, matches)
            else:
                jit_inst = self.default_jit_instruction

            return ToolResult(
                status="success",
                data={"results": results_text},
                jit_instruction=jit_inst,
                client_event=client_evt,
                queued_audio_url=queued_audio,
            )
        else:
            print(f"ℹ️ [{self.name}] No matches found for: '{query}'")
            if self.jit_instruction_builder:
                jit_inst = self.jit_instruction_builder(query, [])
            else:
                jit_inst = "Politely and naturally state that this specific information is currently not available. Do not mention internal technical terms aloud."

            return ToolResult(
                status="no_results",
                data={"results": self.no_results_message},
                jit_instruction=jit_inst,
            )
