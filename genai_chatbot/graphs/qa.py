"""LangGraph workflow that answers questions about the active document with checked citations.

resolve_question -> plan_queries -> direct:     retrieve_direct -----> synthesize -> validate_citations
                                 -> decomposed: retrieve_decomposed --/                 -> respond

resolve_question rewrites follow-ups using the last eight messages (no model call for a first question).
A planning failure falls back to one search with the question itself. A configuration problem (bad key
or model name) at any model step, or a failed synthesis, skips straight to respond with a service
message. The conversation lives in an in-memory checkpointer, one thread per session and document.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from genai_chatbot.citations import CitationError, passage_labels, resolve_citations
from genai_chatbot.observability import Correlation, log_event, log_stage, new_id
from genai_chatbot.prompts import answer_messages, condense_messages, plan_messages
from genai_chatbot.providers import LLMRouter, ProviderError
from genai_chatbot.schemas import (
    HISTORY_MESSAGES,
    AnswerDraft,
    ChatAnswer,
    ChatTurn,
    QueryPlan,
    RetrievedPassage,
    StandaloneQuestion,
)

MAX_QUESTION_CHARS = 2000

UNSUPPORTED = "I couldn't find an answer to that in the document."
UNVERIFIED = (
    "I couldn't match an answer to the document's text, so I'm not giving one. Try asking in a different way."
)
UNEXPECTED = "Something went wrong while answering. Please try again."

# Types the checkpointer may rebuild from its stored state; anything else is refused.
CHECKPOINT_TYPES = [
    ("genai_chatbot.schemas", name)
    for name in ("AnswerDraft", "ChatAnswer", "ChatTurn", "Citation", "QueryPlan", "RetrievedPassage")
]


class PassageSearch(Protocol):
    """What the graph needs from the index (see vector_store.DocumentIndex)."""

    document_id: str

    def search(self, query: str) -> list[RetrievedPassage]: ...


class QAState(TypedDict, total=False):
    messages: Annotated[list[ChatTurn], operator.add]  # the conversation, kept across turns
    # Everything below describes the current turn and is reset by Conversation.ask.
    question: str
    request_id: str
    standalone_question: str
    plan: QueryPlan | None
    route: Literal["direct", "decomposed"]
    passages: list[RetrievedPassage]
    draft: AnswerDraft | None
    failure: str | None
    answer: ChatAnswer | None


def new_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer that only revives the app's own models from stored state."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def _progress(step: str, message: str) -> None:
    get_stream_writer()({"step": step, "message": message})


def build_qa_graph(
    index: PassageSearch,
    router: LLMRouter,
    *,
    session_id: str,
    filename: str,
    page_count: int,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph:
    def correlation(state: QAState) -> Correlation:
        return Correlation(session_id=session_id, request_id=state["request_id"])

    def resolve_question(state: QAState) -> QAState:
        history = state["messages"][:-1][-HISTORY_MESSAGES:]
        if not history:
            return {"standalone_question": state["question"]}
        _progress("resolve", "Reading your question in context…")
        with log_stage("resolve", correlation(state)) as details:
            try:
                result = router.invoke_structured(
                    StandaloneQuestion,
                    condense_messages(history, state["question"]),
                    stage="resolve",
                    correlation=correlation(state),
                )
            except ProviderError as exc:
                details.update(outcome="fallback", error_kind=exc.kind.value)
                if exc.is_configuration_problem:
                    return {"failure": exc.user_message}
                return {"standalone_question": state["question"]}
        return {"standalone_question": " ".join(result.question.split()) or state["question"]}

    def plan_queries(state: QAState) -> QAState:
        _progress("plan", "Planning the search…")
        with log_stage("plan", correlation(state)) as details:
            try:
                plan: QueryPlan | None = router.invoke_structured(
                    QueryPlan,
                    plan_messages(filename, page_count, state["standalone_question"]),
                    stage="plan",
                    correlation=correlation(state),
                )
            except ProviderError as exc:
                details.update(outcome="fallback", error_kind=exc.kind.value)
                if exc.is_configuration_problem:
                    return {"failure": exc.user_message}
                plan = None
            route: Literal["direct", "decomposed"] = "decomposed" if plan and plan.decompose else "direct"
            details.update(
                route=route,
                query_count=len(plan.queries) if plan else 1,
                response_style=plan.response_style if plan else "prose",
            )
        return {"plan": plan, "route": route}

    def retrieve(state: QAState, queries: list[str]) -> QAState:
        _progress("retrieve", "Searching the document…")
        with log_stage(
            "retrieve", correlation(state), route=state["route"], query_count=len(queries)
        ) as details:
            passages: list[RetrievedPassage] = []
            seen: set[str] = set()
            retrieved = 0
            for query in queries:
                results = index.search(query)
                retrieved += len(results)
                for passage in results:
                    keys = {passage.chunk_id, " ".join(passage.text.split())}
                    if seen.isdisjoint(keys):
                        seen |= keys
                        passages.append(passage)
            details.update(retrieval_count=retrieved, passage_count=len(passages))
        return {"passages": passages}

    def retrieve_direct(state: QAState) -> QAState:
        plan = state.get("plan")
        return retrieve(state, [plan.queries[0] if plan else state["standalone_question"]])

    def retrieve_decomposed(state: QAState) -> QAState:
        return retrieve(state, state["plan"].queries)

    def synthesize(state: QAState) -> QAState:
        passages = state["passages"]
        if not passages:
            return {"draft": None}
        _progress("answer", "Writing the answer…")
        plan = state.get("plan")
        with log_stage("answer", correlation(state), passage_count=len(passages)) as details:
            try:
                draft = router.invoke_structured(
                    AnswerDraft,
                    answer_messages(
                        filename,
                        state["standalone_question"],
                        passage_labels(passages),
                        plan.response_style if plan else "prose",
                        plan,
                    ),
                    stage="answer",
                    correlation=correlation(state),
                )
            except ProviderError as exc:
                details.update(outcome="error", error_kind=exc.kind.value)
                return {"failure": exc.user_message}
            details.update(outcome=draft.status, response_style=draft.response_style)
        return {"draft": draft}

    def validate_citations(state: QAState) -> QAState:
        _progress("validate", "Checking the sources…")
        draft = state.get("draft")
        with log_stage("validate", correlation(state)) as details:
            if draft is None or draft.status == "unsupported":
                answer = ChatAnswer(status="unsupported", text=UNSUPPORTED)
            else:
                try:
                    resolved = resolve_citations(
                        draft.answer, draft.citation_ids, passage_labels(state["passages"])
                    )
                except CitationError:
                    details["outcome"] = "invalid"
                    answer = ChatAnswer(status="unsupported", text=UNVERIFIED)
                else:
                    answer = ChatAnswer(
                        status="answered",
                        response_style=draft.response_style,
                        text=resolved.text,
                        citations=resolved.citations,
                    )
            details["citation_count"] = len(answer.citations)
        return {"answer": answer}

    def respond(state: QAState) -> QAState:
        answer = state.get("answer")
        if state.get("failure") or answer is None:
            answer = ChatAnswer(status="error", text=state.get("failure") or UNEXPECTED)
        log_event(
            "answer",
            correlation(state),
            outcome=answer.status,
            response_style=answer.response_style,
            citation_count=len(answer.citations),
        )
        return {
            "answer": answer,
            "messages": [ChatTurn(role="assistant", content=answer.text, answer=answer)],
        }

    def after_resolve(state: QAState) -> Literal["plan_queries", "respond"]:
        return "respond" if state.get("failure") else "plan_queries"

    def after_plan(state: QAState) -> Literal["retrieve_direct", "retrieve_decomposed", "respond"]:
        if state.get("failure"):
            return "respond"
        return "retrieve_decomposed" if state["route"] == "decomposed" else "retrieve_direct"

    def after_synthesis(state: QAState) -> Literal["validate_citations", "respond"]:
        return "respond" if state.get("failure") else "validate_citations"

    graph = StateGraph(QAState)
    graph.add_node("resolve_question", resolve_question)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("retrieve_direct", retrieve_direct)
    graph.add_node("retrieve_decomposed", retrieve_decomposed)
    graph.add_node("synthesize", synthesize)
    graph.add_node("validate_citations", validate_citations)
    graph.add_node("respond", respond)
    graph.add_edge(START, "resolve_question")
    graph.add_conditional_edges("resolve_question", after_resolve)
    graph.add_conditional_edges("plan_queries", after_plan)
    graph.add_edge("retrieve_direct", "synthesize")
    graph.add_edge("retrieve_decomposed", "synthesize")
    graph.add_conditional_edges("synthesize", after_synthesis)
    graph.add_edge("validate_citations", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer, name="question_answering")


class Conversation:
    """One session's chat about one document: the QA graph plus its checkpointer thread."""

    def __init__(
        self,
        index: PassageSearch,
        router: LLMRouter,
        *,
        session_id: str,
        filename: str,
        page_count: int,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        self.session_id = session_id
        self.thread_id = f"{session_id}-{index.document_id[:16]}-{new_id()}"
        self._checkpointer = checkpointer or new_checkpointer()
        self.graph = build_qa_graph(
            index,
            router,
            session_id=session_id,
            filename=filename,
            page_count=page_count,
            checkpointer=self._checkpointer,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    def ask(self, question: str, on_progress: Callable[[dict[str, Any]], None] | None = None) -> ChatAnswer:
        """Answer one question, adding it and the answer to the conversation."""
        question = question.strip()[:MAX_QUESTION_CHARS]
        if not question:
            raise ValueError("the question is empty")
        turn: QAState = {
            "messages": [ChatTurn(role="user", content=question)],
            "question": question,
            "request_id": new_id(),
            "standalone_question": question,
            "plan": None,
            "route": "direct",
            "passages": [],
            "draft": None,
            "failure": None,
            "answer": None,
        }
        final: QAState = {}
        try:
            for mode, payload in self.graph.stream(turn, self.config, stream_mode=["custom", "values"]):
                if mode == "custom" and on_progress is not None:
                    on_progress(payload)
                elif mode == "values":
                    final = payload
        except Exception as exc:
            correlation = Correlation(session_id=self.session_id, request_id=turn["request_id"])
            log_event("answer", correlation, outcome="error", error_kind=type(exc).__name__)
            answer = ChatAnswer(status="error", text=UNEXPECTED)
            self.graph.update_state(
                self.config, {"messages": [ChatTurn(role="assistant", content=answer.text, answer=answer)]}
            )
            return answer
        return final["answer"]

    def history(self) -> list[ChatTurn]:
        return list(self.graph.get_state(self.config).values.get("messages", []))

    def clear(self) -> None:
        """Forget the conversation by deleting its checkpointer thread."""
        self._checkpointer.delete_thread(self.thread_id)
