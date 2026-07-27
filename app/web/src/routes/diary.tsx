import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { listDiary, writeDiary, type DiaryEntry } from '~/api'

export const Route = createFileRoute('/diary')({ component: Diary })

function Diary() {
  const [entries, setEntries] = useState<DiaryEntry[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [reaction, setReaction] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    listDiary().then(setEntries).catch((e) => setError(String(e)))
  }, [])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    const text = draft.trim()
    if (!text || busy) return
    setBusy(true)
    setError('')
    try {
      const res = await writeDiary(text)
      setReaction(res.reaction)
      setDraft('')
      setEntries(await listDiary())
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  const byDay = entries.reduce<Record<string, DiaryEntry[]>>((acc, e) => {
    ;(acc[e.day] ||= []).push(e)
    return acc
  }, {})

  return (
    <div className="app">
      <header>
        <h1>Diary</h1>
        <span className="stat">
          {entries.length} entries · everything here is private to you
        </span>
        <span className="spacer" />
        <Link to="/">
          <button>back to chat</button>
        </Link>
      </header>

      <div className="body">
        <section className="chat">
          <div className="messages">
            {reaction && (
              <div>
                <div className="turn twin">
                  <div className="bubble">{reaction}</div>
                </div>
                <div className="meta">the twin, on today</div>
              </div>
            )}
            {Object.keys(byDay).length === 0 && !busy && (
              <p className="empty">
                Nothing written yet.
                <br />
                <em>Say how the day went — it becomes something the twin remembers.</em>
              </p>
            )}
            {Object.entries(byDay).map(([day, rows]) => (
              <div key={day} style={{ maxWidth: 720, margin: '0 auto 18px' }}>
                <div className="meta" style={{ margin: '0 0 6px' }}>
                  {day}
                </div>
                {rows.map((r) => (
                  <div className="mem" key={r.id}>
                    <div className="head">
                      <span className="tag">{r.source}</span>
                      <span>{r.created_at.slice(11, 16)}</span>
                    </div>
                    <div className="text">{r.text}</div>
                  </div>
                ))}
              </div>
            ))}
            {error && <p className="error">{error}</p>}
          </div>

          <div className="composer">
            <form onSubmit={submit}>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) submit(e as unknown as React.FormEvent)
                }}
                placeholder="aaj kya hua…"
                rows={1}
              />
              <button type="submit" disabled={busy || !draft.trim()}>
                {busy ? 'saving' : 'save'}
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  )
}
