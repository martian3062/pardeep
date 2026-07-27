import { useEffect, useRef, useState } from 'react'
import { createLandmarker, readMood, reportMood, smooth, type Reading } from '~/mood'

/**
 * Opt-in camera. Off until switched on, and the video element is never rendered
 * larger than a thumbnail — the point is the reading, not the picture.
 */
export function MoodCamera() {
  const [on, setOn] = useState(false)
  const [reading, setReading] = useState<Reading | null>(null)
  const [error, setError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const stopRef = useRef<() => void>(() => {})

  useEffect(() => {
    if (!on) {
      stopRef.current()
      setReading(null)
      return
    }
    let cancelled = false
    let raf = 0
    let stream: MediaStream | null = null
    const window_: Reading[] = []
    let lastSent = 0

    ;(async () => {
      try {
        const landmarker = await createLandmarker()
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        const video = videoRef.current
        if (!video || cancelled) return
        video.srcObject = stream
        await video.play()

        const tick = () => {
          if (cancelled || !video.videoWidth) {
            raf = requestAnimationFrame(tick)
            return
          }
          const result = landmarker.detectForVideo(video, performance.now())
          const now = readMood(result)
          window_.push(now)
          if (window_.length > 30) window_.shift()

          const settled = smooth(window_)
          setReading(settled)
          // one reading every 15s is plenty to colour a conversation
          if (Date.now() - lastSent > 15000) {
            lastSent = Date.now()
            void reportMood(settled)
          }
          raf = requestAnimationFrame(tick)
        }
        tick()
      } catch (err) {
        setError(String(err))
        setOn(false)
      }
    })()

    stopRef.current = () => {
      cancelled = true
      cancelAnimationFrame(raf)
      stream?.getTracks().forEach((t) => t.stop())
    }
    return () => stopRef.current()
  }, [on])

  return (
    <div className="mood">
      <button className="linklike" onClick={() => setOn((v) => !v)}>
        {on ? 'camera on' : 'camera off'}
      </button>
      {reading && reading.mood !== 'neutral' && (
        <span className="tag">
          {reading.mood} {reading.score.toFixed(2)}
        </span>
      )}
      <video ref={videoRef} muted playsInline className={on ? 'peek' : 'hidden'} />
      {error && <span className="error">{error}</span>}
      {on && <span className="note">video stays in this tab</span>}
    </div>
  )
}
