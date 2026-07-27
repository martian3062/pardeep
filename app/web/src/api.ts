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
