export type Memory = {
  text: string
  kind: 'episodic' | 'photo' | 'fact' | string
  timestamp: string
  counterpart: string
  photo_path: string
  score: number
}

export type ChatResponse = {
  reply: string
  backend: string
  model: string
  memories: Memory[]
}

export type Stats = {
  memories: number
  by_kind: Record<string, number>
  backends: string[]
}

/** Vite proxies /api to the Python server, so the browser stays same-origin. */
export async function sendMessage(
  message: string,
  opts: { counterpart?: string; guest?: boolean } = {},
): Promise<ChatResponse> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      counterpart: opts.counterpart ?? '',
      guest: opts.guest ?? false,
    }),
  })
  if (!res.ok) throw new Error(`chat failed: ${res.status} ${await res.text()}`)
  return res.json()
}

export async function searchMemory(q: string, limit = 12): Promise<Memory[]> {
  const res = await fetch(`/api/memory?q=${encodeURIComponent(q)}&limit=${limit}`)
  if (!res.ok) throw new Error(`memory search failed: ${res.status}`)
  return res.json()
}

export async function getStats(): Promise<Stats> {
  const res = await fetch('/api/stats')
  if (!res.ok) throw new Error(`stats failed: ${res.status}`)
  return res.json()
}

export function photoUrl(path: string): string {
  return `/api/photo?path=${encodeURIComponent(path)}`
}

export function resetConversation(): Promise<Response> {
  return fetch('/api/reset', { method: 'POST' })
}

export type Question = { id: string; text: string; lang: string; register: string }

export type SessionProgress = {
  clips: number
  minutes: number
  target_minutes: number
  pct: number
  per_language_minutes: Record<string, number>
  rejected: number
}

export type ClipResult = {
  id: number
  accepted: boolean
  problem: string
  seconds: number
  sample_rate: number
  bandwidth_hz: number
  rms: number
  progress: SessionProgress
}

export async function getQuestions(): Promise<{
  questions: Question[]
  progress: SessionProgress
}> {
  const res = await fetch('/api/record/questions')
  if (!res.ok) throw new Error(`questions failed: ${res.status}`)
  return res.json()
}

export async function uploadClip(blob: Blob, q: Question): Promise<ClipResult> {
  const form = new FormData()
  form.append('data', blob, `${q.id}.wav`)
  const params = new URLSearchParams({ question_id: q.id, question: q.text, lang: q.lang })
  const res = await fetch(`/api/record/clip?${params}`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`upload failed: ${res.status} ${await res.text()}`)
  return res.json()
}

export async function exportManifest(): Promise<{ manifest: string }> {
  const res = await fetch('/api/record/export', { method: 'POST' })
  if (!res.ok) throw new Error(`export failed: ${res.status}`)
  return res.json()
}

export type DiaryEntry = {
  id: number
  day: string
  created_at: string
  source: string
  text: string
}

export type DiaryResponse = { entry_id: number; indexed: number; reaction: string }

export async function writeDiary(text: string): Promise<DiaryResponse> {
  const res = await fetch('/api/diary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`diary failed: ${res.status} ${await res.text()}`)
  return res.json()
}

export async function listDiary(limit = 30): Promise<DiaryEntry[]> {
  const res = await fetch(`/api/diary?limit=${limit}`)
  if (!res.ok) throw new Error(`diary list failed: ${res.status}`)
  return res.json()
}

/** A rewrite is the only feedback that can retrain the twin; a bare thumb cannot. */
export async function sendCorrection(input: {
  prompt: string
  said: string
  instead?: string
  verdict?: 'up' | 'down'
}): Promise<{ id: number; trainable: boolean }> {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: input.prompt,
      said: input.said,
      instead: input.instead ?? '',
      verdict: input.verdict ?? 'down',
    }),
  })
  if (!res.ok) throw new Error(`feedback failed: ${res.status}`)
  return res.json()
}
