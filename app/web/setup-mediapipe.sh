#!/usr/bin/env bash
# MediaPipe assets are served from this app rather than a CDN, so the mood camera
# works offline and the page loads nothing from a third party. Not committed:
# ~4MB of vendor binaries that npm can reproduce.
set -e
cd "$(dirname "$0")"
mkdir -p public/mediapipe/wasm
cp -r node_modules/@mediapipe/tasks-vision/wasm/* public/mediapipe/wasm/
curl -sL -o public/mediapipe/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
echo "mediapipe assets ready"
