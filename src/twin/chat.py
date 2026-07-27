"""One turn of conversation with the twin: retrieve, frame, generate."""
from __future__ import annotations

from dataclasses import dataclass, field

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


class Twin:
    def __init__(self, store: MemoryStore | None = None, guest: bool = False):
        self.store = store or MemoryStore()
        self.guest = guest
        self.identity = ident.load()
        self.history: list[Turn] = []

    def _retrieval_query(self, message: str) -> str:
        """Search with a little context: a bare "and then?" retrieves nothing on
        its own, but carries meaning after the turn before it."""
        recent = [t.content for t in self.history[-2:] if t.role == "user"]
        return " ".join([*recent, message]).strip()

    def reply(
        self,
        message: str,
        counterpart: str = "",
        backend: str = "auto",
        memory_limit: int = 8,
    ) -> TwinReply:
        snippets = gather(
            self._retrieval_query(message),
            store=self.store,
            limit=memory_limit,
            guest=self.guest,
        )
        system = build_system_prompt(
            self.identity, snippets, counterpart=counterpart, guest=self.guest
        )
        messages = [
            {"role": t.role, "content": t.content} for t in self.history[-HISTORY_TURNS:]
        ]
        messages.append({"role": "user", "content": message})

        out: Reply = generate(system, messages, backend=backend)

        self.history.append(Turn("user", message))
        self.history.append(Turn("assistant", out.text))
        return TwinReply(
            text=out.text, backend=out.backend, model=out.model, memories=snippets
        )
