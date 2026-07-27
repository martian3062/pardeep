/**
 * Mood from the camera, computed entirely in this tab.
 *
 * MediaPipe runs as WASM in the page. Frames go from the webcam into that
 * WASM module and nowhere else — what leaves the browser is one label and a
 * number. The rest of the project keeps raw data on the machine; video is the
 * most invasive raw data there is, so it does not even get as far as the local
 * API.
 */
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from '@mediapipe/tasks-vision'

export type MoodLabel = 'happy' | 'tired' | 'stressed' | 'surprised' | 'neutral'
export type Reading = { mood: MoodLabel; score: number }

/** Blendshape names MediaPipe emits, grouped into the moods we act on. */
const SIGNALS: Record<Exclude<MoodLabel, 'neutral'>, string[]> = {
  happy: ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight'],
  tired: ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight'],
  stressed: ['browDownLeft', 'browDownRight', 'mouthPressLeft', 'mouthPressRight'],
  surprised: ['browInnerUp', 'eyeWideLeft', 'eyeWideRight', 'jawOpen'],
}

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(
    '/mediapipe/wasm',
  )
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/mediapipe/face_landmarker.task', delegate: 'GPU' },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1,
  })
}

export function readMood(result: FaceLandmarkerResult): Reading {
  const shapes = result.faceBlendshapes?.[0]?.categories
  if (!shapes || shapes.length === 0) return { mood: 'neutral', score: 0 }

  const byName = new Map(shapes.map((c) => [c.categoryName, c.score]))
  const mean = (names: string[]) => {
    const vals = names.map((n) => byName.get(n) ?? 0)
    return vals.reduce((a, b) => a + b, 0) / vals.length
  }

  let best: MoodLabel = 'neutral'
  let bestScore = 0
  for (const [mood, names] of Object.entries(SIGNALS)) {
    const score = mean(names)
    if (score > bestScore) {
      best = mood as MoodLabel
      bestScore = score
    }
  }
  // A face at rest scores low on everything; calling that "tired" would have the
  // twin softening its tone at random.
  return bestScore < 0.2 ? { mood: 'neutral', score: bestScore } : { mood: best, score: bestScore }
}

/** Smooth over a window — a single blink is not tiredness. */
export function smooth(recent: Reading[]): Reading {
  if (recent.length === 0) return { mood: 'neutral', score: 0 }
  const totals = new Map<MoodLabel, number>()
  for (const r of recent) totals.set(r.mood, (totals.get(r.mood) ?? 0) + r.score)
  let best: MoodLabel = 'neutral'
  let bestScore = 0
  for (const [mood, total] of totals) {
    if (total > bestScore) {
      best = mood
      bestScore = total
    }
  }
  return { mood: best, score: bestScore / recent.length }
}

export async function reportMood(reading: Reading): Promise<void> {
  await fetch('/api/mood', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reading),
  }).catch(() => {})
}
