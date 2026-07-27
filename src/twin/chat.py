"""One turn of conversation with the twin: retrieve, frame, generate."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..guard.injection import REFUSAL, inspect
from ..guard.pii import redact
from ..memory.store import MemoryStore
from . import identity as ident
from .prompt import build_system_prompt
from .retrieve import Snippet, gather
from .router import Reply, generate

# Enough turns to hold a thread, few enough that the archive memories -- the part
# that makes this a twin rather than a chatbot -- still dominate the context.
HISTORY_TURNS = 8


@dataclass
class Turn:
    role: str  # user | assistant
    content: str


@dataclass
class TwinReply:
    text: str
    backend: str
    model: str
    memories: list[Snippet] = field(default_factory=list)
    outcome: str = "answered"  # answered | refused | redacted
    redactions: tuple[str, ...] = ()


class Twin:
    def __init__(
        self,
        store: MemoryStore | None = None,
        guest: bool = False,
        guest_label: str = "",
    ):
        self.store = store or MemoryStore()
        self.guest = guest
        self.guest_label = guest_label  # who is talking, if they said
        self.identity = ident.load()
        self.history: list[Turn] = []

    # A follow-up like "aur?" or "phir kya hua" carries no searchable content and
    # needs the turn before it. A message that stands on its own does not, and
    # borrowing context anyway hijacks it: asked about college right after a
    # question about a 2017 trip, retrieval returned the trip's train photos and
    # the twin reported having no classroom photos — while holding several.
    STANDALONE_WORDS = 4

    def _mood_rule(self) -> str:
        """A recent camera reading, if the browser has sent one. Never blocking."""
        try:
            from .mood import connect, current, tone_rule

            return tone_rule(current(connect()))
        except Exception:
            return ""

    def _retrieval_query(self, message: str) -> str:
        if len(message.split()) >= self.STANDALONE_WORDS:
            return message.strip()
        recent = [t.content for t in self.history[-2:] if t.role == "user"]
        return " ".join([*recent, message]).strip()

    def reply(
        self,
        message: str,
        counterpart: str = "",
        backend: str = "auto",
        memory_limit: int = 8,
    ) -> TwinReply:
        # Guests get an extra pass before anything is generated. The real defence
        # is that their retrieval never touches private memories, but an obvious
        # attempt to talk the twin out of its rules should not reach a model at
        # all — and should be on the record.
        if self.guest:
            verdict = inspect(message)
            if verdict.blocked:
                self._audit(message, REFUSAL, "refused", reason=verdict.reason)
                return TwinReply(
                    text=REFUSAL, backend="guard", model="injection", outcome="refused"
                )

        snippets = gather(
            self._retrieval_query(message),
            store=self.store,
            limit=memory_limit,
            guest=self.guest,
        )
        system = build_system_prompt(
            self.identity,
            snippets,
            counterpart=counterpart,
            guest=self.guest,
            message=message,
            mood_rule=self._mood_rule(),
        )
        messages = [
            {"role": t.role, "content": t.content} for t in self.history[-HISTORY_TURNS:]
        ]
        messages.append({"role": "user", "content": message})

        out: Reply = generate(system, messages, backend=backend)

        text, outcome, removed = out.text, "answered", ()
        if self.guest:
            # Trust-tier retrieval should already have made this unnecessary.
            # It runs anyway, because "should" is not a guarantee and a model can
            # reconstruct a number from context that was individually harmless.
            text, findings = redact(out.text)
            if findings:
                outcome = "redacted"
                removed = tuple(sorted({f.kind for f in findings}))
            self._audit(message, text, outcome, redactions=removed, memories=len(snippets))

        self.history.append(Turn("user", message))
        self.history.append(Turn("assistant", text))
        return TwinReply(
            text=text,
            backend=out.backend,
            model=out.model,
            memories=snippets,
            outcome=outcome,
            redactions=removed,
        )

    def _audit(
        self,
        message: str,
        reply: str,
        outcome: str,
        reason: str = "",
        redactions: tuple[str, ...] = (),
        memories: int = 0,
    ) -> None:
        """Log a guest turn. Owner conversations are never recorded."""
        try:
            from ..guard import audit

            audit.record(
                audit.connect(),
                audit.Turn(
                    message=message,
                    reply=reply,
                    outcome=outcome,
                    guest_label=self.guest_label,
                    blocked_reason=reason,
                    redactions=redactions,
                    memories_used=memories,
                ),
            )
        except Exception:
            pass  # a failed log must never cost the user their reply
