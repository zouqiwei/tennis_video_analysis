dou'ky'ei# Tennis Analysis Precision Upgrade Design

Date: 2026-06-18

## Goal

Improve the current tennis swing analysis MVP so its pose-based scores and swing phase estimates are more stable, more explainable, and closer to coaching intuition while staying compatible with the existing FastAPI API.

This upgrade keeps the project focused on human pose analysis. It does not add ball tracking, racket detection, court-line detection, user accounts, or production queue infrastructure.

## Scope

Included:

- Smooth pose landmark sequences before metric calculation.
- Infer the dominant swing wrist instead of always using `left_wrist`.
- Detect a richer swing phase sequence:
  - `ready`
  - `backswing_peak`
  - `estimated_contact`
  - `follow_through_peak`
- Estimate contact using multiple motion signals rather than wrist movement alone.
- Add more specific pose metrics:
  - `knee_bend`
  - `torso_stability`
  - `swing_tempo`
  - `arm_extension`
  - `contact_confidence`
- Improve coaching feedback for low-confidence and low-score cases.
- Preserve the existing response shape for `/api/jobs/{job_id}/result`.
- Add focused tests for hand selection, phase detection, confidence, and feedback.

Excluded:

- Tennis ball tracking.
- Racket detection.
- Camera calibration.
- Biomechanical diagnosis.
- ML model training.
- Database persistence.

## Current Problem

The existing MVP works, but the analysis is coarse:

- Contact is estimated from the largest `left_wrist` x-position change.
- Metrics assume the left wrist is the important hitting-side signal.
- Raw landmark noise can change scores noticeably.
- The report only returns one phase, `estimated_contact`.
- Feedback does not communicate confidence or stage-specific limitations.

These choices are reasonable for an MVP, but they make the result less reliable when the player is right-handed, landmarks jitter, or the swing path is not cleanly horizontal in the camera view.

## Design

### Landmark Preprocessing

Add a preprocessing step inside `app.metrics` before computing metrics:

- Preserve the public `LandmarkFrame` dataclass.
- Add an internal helper that applies a small moving average to x/y landmark coordinates.
- Keep visibility values unsmoothed or averaged conservatively so missing/low-confidence points still affect visibility.
- Use a default window of 3 frames.

This should reduce frame-to-frame jitter without hiding large swing motion.

### Dominant Wrist Selection

Add an internal helper that compares `left_wrist` and `right_wrist` motion across the sequence:

- Calculate total travel for each wrist across sampled frames.
- Prefer the wrist with larger total travel.
- If both wrists are close, keep `left_wrist` as the deterministic fallback.
- Use the selected wrist consistently in backswing, contact, follow-through, tempo, and extension metrics.

The result remains heuristic, but it removes the biggest current bias.

### Swing Phase Detection

Replace the single contact-only phase estimate with a richer phase detector:

- `ready`: first usable frame.
- `backswing_peak`: pre-contact frame where the selected wrist is farthest from the final follow-through direction.
- `estimated_contact`: frame with the strongest combined contact signal.
- `follow_through_peak`: post-contact frame where the selected wrist reaches maximum continuation.

Contact signal should combine:

- Selected wrist speed.
- Wrist distance from hip center.
- Shoulder-hip rotation activity.
- Local motion peak near the middle portion of the swing when available.

The phase detector must return valid frame indexes even for short sequences. If the video is too short or noisy, it should still produce a best-effort sequence and lower `contact_confidence`.

### New And Updated Metrics

Keep existing metric names for compatibility and add new ones.

Updated metrics:

- `backswing`: use selected wrist travel before contact.
- `contact_position`: use selected wrist distance from hip center at estimated contact.
- `follow_through`: use selected wrist continuation after contact.
- `shoulder_hip_separation`: keep current concept, but calculate from smoothed frames.

New metrics:

- `knee_bend`: opening-stage lower-body loading from hip, knee, and ankle geometry.
- `torso_stability`: penalize excessive shoulder-center or hip-center vertical wobble.
- `swing_tempo`: compare pre-contact and post-contact duration balance.
- `arm_extension`: estimate hitting-arm extension at contact from shoulder, elbow, and wrist geometry.
- `contact_confidence`: summarize how trustworthy the contact estimate is based on visibility, motion peak strength, and sequence length.

Overall score should continue to be the mean of metric scores. This keeps behavior simple and avoids introducing hidden weights in this iteration.

### Feedback

Update `app.feedback` to handle the new metrics:

- If `contact_confidence` is low, add a clear note that the phase estimate may be unstable.
- Give specific advice for knee bend, torso stability, tempo, and arm extension.
- Keep the final disclaimer that the result is a pose-keypoint heuristic, not a biomechanical diagnosis.
- Preserve existing Chinese user-facing tone.

### API Compatibility

No route changes.

`AnalysisReport` remains:

```json
{
  "job_id": "uuid",
  "overall_score": 82.5,
  "metrics": [],
  "phases": [],
  "feedback": [],
  "annotated_video_path": "data/outputs/uuid/annotated.mp4",
  "key_frame_paths": []
}
```

The `phases` array will contain four entries instead of one when enough data is available. The `metrics` array will include additional metric names. Existing clients that render metrics dynamically will continue to work.

### Testing

Add or update tests for:

- Dominant wrist selection chooses the wrist with greater travel.
- Phase detection returns ordered phase indexes for a synthetic swing.
- Contact confidence drops when visibility is low or sequence length is too short.
- Existing high-quality synthetic swing still scores well.
- Feedback includes low-confidence and new metric-specific advice.
- API result schema remains compatible.

## Risks

- The algorithm is still heuristic and depends on stable side-view footage.
- Auto-selecting the hitting wrist by travel can be wrong if the non-hitting hand moves more.
- More metrics can make the overall score stricter than before.
- Smoothing can slightly delay detected peaks if the window is too large.

Mitigations:

- Use a small smoothing window.
- Keep deterministic fallbacks.
- Include confidence feedback.
- Add tests that lock down expected behavior on synthetic sequences.

## Acceptance Criteria

- Existing tests pass.
- New tests cover hand selection, phase detection, confidence, and feedback.
- `/api/jobs/{job_id}/result` keeps the same top-level JSON shape.
- Reports include multiple swing phases.
- Reports include the new metrics.
- Low-quality pose sequences produce lower confidence and user-facing caution.
- No new runtime dependency is required.
