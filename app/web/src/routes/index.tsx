import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'
import {
  getStats,
  photoUrl,
  resetConversation,
  sendCorrection,
  sendMessage,
  type Memory,
  type Stats,
} from '~/api'

export const Route = createFileRoute('/')({ component: Chat })

type Turn = { role: 'me' | 'twin'; text: string; via?: string; askedWith?: string }

/** "I'd say it like this" — the rewrite becomes a DPO pair. A thumb alone says
 *  something was wrong but not what right looks like. */
function Correction({ prompt, said }: { prompt: string; said: string }) {
  const [open, setOpen] = useState(false)
  const [instead, setInstead] = useState('')
  const [done, setDone] = useState(false)

  if (done) return <div className="meta">saved — this will train the next version</div>
  if (!open)
    return (
      <div className="meta">
        <button className="linklike" onClick={() => setOpen(true)}>
          I'd say it differently
        </button>
      </div>
    )

  return (
    <div className="correction">
      <textarea
        value={instead}
        onChange={(e) => setInstead(e.target.value)}
        placeholder="how you'd actually say it…"
        rows={2}
      />
      <button
        disabled={!instead.trim()}
        onClick={async () => {
          await sendCorrection({ prompt, said, instead })
          setDone(true)
        }}
      >
        save
      </button>
    </div>
  )
}

function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [memories, setMemories] = useState<Memory[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => setStats(null))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, busy])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    const message = draft.trim()
    if (!message || busy) return
    setDraft('')
    setError('')
    setTurns((t) => [...t, { role: 'me', text: message }])
    setBusy(true)
    try {
      const res = await sendMessage(message)
      setTurns((t) => [
        ...t,
        { role: 'twin', text: res.reply, via: `${res.backend}/${res.model}`, askedWith: message },
      ])
      setMemories(res.memories)
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  async function clear() {
    await resetConversation().catch(() => {})
    setTurns([])
    setMemories([])
    setError('')
  }

  return (
    <div className="app">
      <header>
        <h1>Pardeep_Self</h1>
        {stats && (
          <span className="stat">
            {stats.memories.toLocaleString()} memories ·{' '}
            {Object.entries(stats.by_kind)
              .map(([k, v]) => `${v.toLocaleString()} ${k}`)
              .join(' · ')}{' '}
            · {stats.backends.join(', ') || 'no backend'}
          </span>
        )}
        <span className="spacer" />
        <Link to="/diary">
          <button>diary</button>
        </Link>
        <button onClick={clear}>new conversation</button>
      </header>

      <div className="body">
        <section className="chat">
          <div className="messages">
            {turns.length === 0 && !busy && (
              <p className="empty">
                Ask about anything in your archive — a year, a person, a place.
                <br />
                <em>“2017 june me kya kar raha tha?”</em>
              </p>
            )}
            {turns.map((t, i) => (
              <div key={i}>
                <div className={`turn ${t.role}`}>
                  <div className="bubble">{t.text}</div>
                </div>
                {t.via && <div className="meta">via {t.via}</div>}
                {t.role === 'twin' && t.askedWith && (
                  <Correction prompt={t.askedWith} said={t.text} />
                )}
              </div>
            ))}
            {busy && (
              <div className="turn twin">
                <div className="bubble" style={{ color: 'var(--dim)' }}>
                  remembering…
                </div>
              </div>
            )}
            {error && <p className="error">{error}</p>}
            <div ref={endRef} />
          </div>

          <div className="composer">
            <form onSubmit={submit}>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  // Enter sends, Shift+Enter makes a new line — he writes in bursts
                  if (e.key === 'Enter' && !e.shiftKey) submit(e as unknown as React.FormEvent)
                }}
                placeholder="likh kuch bhi…"
                rows={1}
              />
              <button type="submit" disabled={busy || !draft.trim()}>
                send
              </button>
            </form>
          </div>
        </section>

        <aside>
          <h2>What it remembered</h2>
          {memories.length === 0 && <p className="empty">nothing yet</p>}
          {memories.map((m, i) => (
            <div className="mem" key={i}>
              <div className="head">
                <span className="tag">{m.kind}</span>
                <span>{m.timestamp.slice(0, 10)}</span>
                {m.counterpart && <span>· {m.counterpart}</span>}
                <span style={{ marginLeft: 'auto' }}>{m.score.toFixed(2)}</span>
              </div>
              <div className="text">{m.text.slice(0, 260)}</div>
              {m.photo_path && (
                <img src={photoUrl(m.photo_path)} alt="" loading="lazy" />
              )}
            </div>
          ))}
        </aside>
      </div>
    </div>
  )
}
