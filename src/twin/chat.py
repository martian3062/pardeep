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
        # Default mode for CLI use (`Twin(guest=True)`). The API serves one shared
        # instance to several callers at once, so it passes `guest` per call to
        # reply() instead — see the note on _history below.
        self.guest = guest
        self.guest_label = guest_label  # who is talking, if they said
        self.identity = ident.load()
        # Owner and guest conversations are kept apart. One shared list let a guest's
        # turns become context for the owner's next question (and the reverse), which
        # is both wrong and a leak: guest text steered owner retrieval.
        self._history: dict[bool, list[Turn]] = {False: [], True: []}

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

    def _transcript(self, guest: bool) -> list[Turn]:
        """The transcript for one mode, created on first use.

        Lazy because tests build a Twin with `Twin.__new__` to skip loading bge-m3
        and the persona card, so __init__ has not necessarily run.
        """
        if not hasattr(self, "_history"):
            self._history: dict[bool, list[Turn]] = {False: [], True: []}
        return self._history[guest]

    @property
    def history(self) -> list[Turn]:
        """The transcript for this instance's default mode (CLI callers use this)."""
        return self._transcript(self.guest)

    @history.setter
    def history(self, turns: list[Turn]) -> None:
        self._transcript(self.guest)  # ensure the dict exists
        self._history[self.guest] = list(turns)

    def transcripts(self) -> list[list[Turn]]:
        """Both transcripts — for callers that need to clear everything."""
        self._transcript(False)
        return list(self._history.values())

    def _retrieval_query(self, message: str, guest: bool | None = None) -> str:
        is_guest = self.guest if guest is None else guest
        if len(message.split()) >= self.STANDALONE_WORDS:
            return message.strip()
        recent = [t.content for t in self._transcript(is_guest)[-2:] if t.role == "user"]
        return " ".join([*recent, message]).strip()

    def reply(
        self,
        message: str,
        counterpart: str = "",
        backend: str = "auto",
        memory_limit: int = 8,
        guest: bool | None = None,
        guest_label: str | None = None,
    ) -> TwinReply:
        # Resolved per call, never stored: the API serves one Twin to concurrent
        # callers, and setting an instance flag meant two overlapping requests could
        # swap modes — a guest turn answered without the guard, or an owner turn
        # PII-redacted. The instance value is only the default for CLI use.
        is_guest = self.guest if guest is None else guest
        label = self.guest_label if guest_label is None else guest_label

        # Guests get an extra pass before anything is generated. The real defence
        # is that their retrieval never touches private memories, but an obvious
        # attempt to talk the twin out of its rules should not reach a model at
        # all — and should be on the record.
        if is_guest:
            verdict = inspect(message)
            if verdict.blocked:
                self._audit(
                    message, REFUSAL, "refused", reason=verdict.reason, guest_label=label
                )
                return TwinReply(
                    text=REFUSAL, backend="guard", model="injection", outcome="refused"
                )

        snippets = gather(
            self._retrieval_query(message, is_guest),
            store=self.store,
            limit=memory_limit,
            guest=is_guest,
        )
        system = build_system_prompt(
            self.identity,
            snippets,
            counterpart=counterpart,
            guest=is_guest,
            message=message,
            mood_rule=self._mood_rule(),
        )
        transcript = self._transcript(is_guest)
        messages = [
            {"role": t.role, "content": t.content} for t in transcript[-HISTORY_TURNS:]
        ]
        messages.append({"role": "user", "content": message})

        out: Reply = generate(system, messages, backend=backend)

        text, outcome, removed = out.text, "answered", ()
        if is_guest:
            # Trust-tier retrieval should already have made this unnecessary.
            # It runs anyway, because "should" is not a guarantee and a model can
            # reconstruct a number from context that was individually harmless.
            text, findings = redact(out.text)
            if findings:
                outcome = "redacted"
                removed = tuple(sorted({f.kind for f in findings}))
            self._audit(
                message,
                text,
                outcome,
                redactions=removed,
                memories=len(snippets),
                guest_label=label,
            )

        transcript.append(Turn("user", message))
        transcript.append(Turn("assistant", text))
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
        guest_label: str | None = None,
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
                    guest_label=self.guest_label if guest_label is None else guest_label,
                    blocked_reason=reason,
                    redactions=redactions,
                    memories_used=memories,
                ),
            )
        except Exception:
            pass  # a failed log must never cost the user their reply
