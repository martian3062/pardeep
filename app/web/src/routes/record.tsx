import { createFileRoute, Link } from '@tanstack/react-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  exportManifest,
  getQuestions,
  uploadClip,
  type ClipResult,
  type Question,
  type SessionProgress,
} from '~/api'
import { Recorder } from '~/recorder'

export const Route = createFileRoute('/record')({ component: RecordSession })

type Line =
  | { kind: 'ask'; question: Question }
  | { kind: 'answer'; result: ClipResult; url: string; question: Question }

// He talks in bursts with gaps, so a short pause must not end the answer.
const SILENCE_TO_ADVANCE = 2.2
const MIN_BEFORE_ADVANCE = 4

function RecordSession() {
  const [queue, setQueue] = useState<Question[]>([])
  const [lines, setLines] = useState<Line[]>([])
  const [progress, setProgress] = useState<SessionProgress | null>(null)
  const [live, setLive] = useState(false)
  const [auto, setAuto] = useState(true)
  const [level, setLevel] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const rec = useRef(new Recorder())
  const endRef = useRef<HTMLDivElement>(null)
  const current = queue[0]

  useEffect(() => {
    getQuestions()
      .then((d) => {
        setQueue(d.questions)
        setProgress(d.progress)
        if (d.questions[0]) setLines([{ kind: 'ask', question: d.questions[0] }])
      })
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines, live])

  const finish = useCallback(
    async (q: Question) => {
      const out = rec.current.stop()
      setLive(false)
      setLevel(0)
      setElapsed(0)
      if (!out) return
      setBusy(true)
      try {
        const result = await uploadClip(out.blob, q)
        setProgress(result.progress)
        setLines((l) => [
          ...l,
          { kind: 'answer', result, url: URL.createObjectURL(out.blob), question: q },
        ])
        // a rejected clip means the question is re-asked, not skipped
        const rest = result.accepted ? queue.slice(1) : queue
        setQueue(rest)
        if (rest[0]) setLines((l) => [...l, { kind: 'ask', question: rest[0] }])
      } catch (e) {
        setError(String(e))
      } finally {
        setBusy(false)
      }
    },
    [queue],
  )

  // level meter, timer, and auto-advance on sustained silence
  useEffect(() => {
    if (!live || !current) return
    const id = setInterval(() => {
      setLevel(rec.current.level)
      const secs = rec.current.elapsed
      setElapsed(secs)
      if (auto && secs > MIN_BEFORE_ADVANCE && rec.current.silentFor(0.015, SILENCE_TO_ADVANCE)) {
        void finish(current)
      }
    }, 100)
    return () => clearInterval(id)
  }, [live, auto, current, finish])

  async function begin() {
    if (!current) return
    setError('')
    try {
      await rec.current.start()
      setLive(true)
    } catch (e) {
      setError(`mic blocked: ${e}`)
    }
  }

  function skip() {
    if (!current) return
    if (live) rec.current.stop()
    setLive(false)
    const rest = queue.slice(1)
    setQueue(rest)
    if (rest[0]) setLines((l) => [...l, { kind: 'ask', question: rest[0] }])
  }

  const pct = progress?.pct ?? 0
  const done = pct >= 100

  return (
    <div className="app">
      <header>
        <h1>Voice session</h1>
        {progress && (
          <span className="stat">
            {progress.minutes} of {progress.target_minutes} min &middot; {progress.clips} clips
            {progress.rejected > 0 && ` · ${progress.rejected} rejected`}
            {Object.keys(progress.per_language_minutes).length > 0 &&
              ' · ' +
                Object.entries(progress.per_language_minutes)
                  .map(([k, v]) => `${k} ${v}m`)
                  .join(' · ')}
          </span>
        )}
        <span className="spacer" />
        <label className="toggle">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          auto-advance
        </label>
        <Link to="/">
          <button>back</button>
        </Link>
      </header>

      <div className="bar">
        <div className="bar-fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </div>

      <div className="body">
        <section className="chat">
          <div className="messages">
            {lines.length === 0 && !error && <p className="empty">loading questions…</p>}

            {lines.map((line, i) =>
              line.kind === 'ask' ? (
                <div key={i}>
                  <div className="turn twin">
                    <div className="bubble">{line.question.text}</div>
                  </div>
                  <div className="meta">
                    {line.question.lang} &middot; {line.question.register}
                  </div>
                </div>
              ) : (
                <div key={i}>
                  <div className="turn me">
                    <div className="bubble clip">
                      <audio src={line.url} controls preload="metadata" />
                      <div className="clipmeta">
                        {line.result.seconds}s &middot; {line.result.sample_rate} Hz &middot;{' '}
                        {line.result.bandwidth_hz} Hz real
                      </div>
                    </div>
                  </div>
                  {line.result.problem && (
                    <div className={line.result.accepted ? 'meta' : 'error'}>
                      {line.result.accepted ? '' : '✕ '}
                      {line.result.problem}
                      {!line.result.accepted && ' — asking again'}
                    </div>
                  )}
                </div>
              ),
            )}

            {live && (
              <div className="turn me">
                <div className="bubble recording">
                  <span className="dot" /> recording {elapsed.toFixed(1)}s
                  <span className="meter">
                    <span className="meter-fill" style={{ width: `${Math.min(100, level * 320)}%` }} />
                  </span>
                </div>
              </div>
            )}
            {busy && <div className="meta">checking the clip…</div>}
            {error && <p className="error">{error}</p>}
            {done && (
              <p className="empty">
                Target reached — {progress?.minutes} minutes.
                <br />
                <button onClick={() => exportManifest().then((r) => setError(`manifest: ${r.manifest}`))}>
                  export training manifest
                </button>
              </p>
            )}
            {!current && !done && lines.length > 0 && (
              <p className="empty">No questions left. Export what you have.</p>
            )}
            <div ref={endRef} />
          </div>

          <div className="composer">
            <div className="controls">
              {!live ? (
                <button className="big" onClick={begin} disabled={!current || busy}>
                  {lines.some((l) => l.kind === 'answer') ? 'answer this one' : 'start recording'}
                </button>
              ) : (
                <button className="big stop" onClick={() => current && finish(current)}>
                  done — next question
                </button>
              )}
              <button onClick={skip} disabled={!current || busy}>
                skip
              </button>
            </div>
            <p className="hint">
              Answer out loud, naturally — long answers are better than short ones.
              {auto
                ? ' It moves on by itself when you stop talking.'
                : ' Press “done” when you finish.'}
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
