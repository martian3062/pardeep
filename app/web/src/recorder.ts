/**
 * Microphone capture for voice-model training.
 *
 * Two decisions here are the whole point of the file.
 *
 * MediaRecorder is not used. It encodes to Opus, which band-limits and applies
 * lossy compression — exactly the damage that made 32 hours of archived call
 * recordings unusable for training. Raw Float32 frames are pulled from an
 * AudioWorklet and written as uncompressed WAV instead.
 *
 * Every browser audio "enhancement" is switched off. echoCancellation,
 * noiseSuppression and autoGainControl are designed to make voice calls
 * intelligible, and they do it by gating, ducking and reshaping the signal. A
 * model trained on their output learns the processing, not the voice.
 */

export type Recording = { blob: Blob; seconds: number; sampleRate: number; peak: number }

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0]
    if (ch) this.port.postMessage(new Float32Array(ch))
    return true
  }
}
registerProcessor('tap', Tap)
`

export class Recorder {
  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private node: AudioWorkletNode | null = null
  private chunks: Float32Array[] = []
  private workletUrl: string | null = null

  /** Live level, so the UI can show that the mic is actually picking him up. */
  level = 0
  recording = false

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 48000,
      },
    })
    this.ctx = new AudioContext({ sampleRate: 48000 })
    this.workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
    await this.ctx.audioWorklet.addModule(this.workletUrl)

    const source = this.ctx.createMediaStreamSource(this.stream)
    this.node = new AudioWorkletNode(this.ctx, 'tap')
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const frame = e.data
      this.chunks.push(frame)
      let peak = 0
      for (let i = 0; i < frame.length; i++) {
        const v = Math.abs(frame[i])
        if (v > peak) peak = v
      }
      // smooth, so the meter reads as a level rather than flickering
      this.level = this.level * 0.8 + peak * 0.2
    }
    source.connect(this.node)
    this.chunks = []
    this.recording = true
  }

  /** Stop and hand back a WAV. Returns null if nothing was captured. */
  stop(): Recording | null {
    this.recording = false
    const sampleRate = this.ctx?.sampleRate ?? 48000
    this.node?.port.close()
    this.node?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    void this.ctx?.close()
    if (this.workletUrl) URL.revokeObjectURL(this.workletUrl)
    this.ctx = null
    this.stream = null
    this.node = null
    this.level = 0

    const total = this.chunks.reduce((n, c) => n + c.length, 0)
    if (total === 0) return null

    const samples = new Float32Array(total)
    let at = 0
    let peak = 0
    for (const c of this.chunks) {
      samples.set(c, at)
      at += c.length
      for (let i = 0; i < c.length; i++) {
        const v = Math.abs(c[i])
        if (v > peak) peak = v
      }
    }
    this.chunks = []
    return {
      blob: encodeWav(samples, sampleRate),
      seconds: total / sampleRate,
      sampleRate,
      peak,
    }
  }

  /** Seconds captured so far. */
  get elapsed(): number {
    const total = this.chunks.reduce((n, c) => n + c.length, 0)
    return total / (this.ctx?.sampleRate ?? 48000)
  }

  /** Sustained quiet, used to advance to the next question on its own. */
  silentFor(threshold = 0.015, windowSec = 1.6): boolean {
    const sr = this.ctx?.sampleRate ?? 48000
    const need = Math.floor(sr * windowSec)
    let seen = 0
    for (let i = this.chunks.length - 1; i >= 0 && seen < need; i--) {
      const c = this.chunks[i]
      for (let j = c.length - 1; j >= 0 && seen < need; j--, seen++) {
        if (Math.abs(c[j]) > threshold) return false
      }
    }
    return seen >= need
  }
}

/** 16-bit PCM WAV. Uncompressed on purpose. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const str = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i))
  }
  str(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  str(8, 'WAVE')
  str(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  str(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let at = 44
  for (let i = 0; i < samples.length; i++, at += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(at, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}
