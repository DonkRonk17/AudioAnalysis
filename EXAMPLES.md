# AudioAnalysis - Examples

**10+ Practical Examples of Audio Analysis**

---

## Table of Contents

1. [Basic Analysis](#1-basic-analysis)
2. [Voltage Gauge Deep Dive](#2-voltage-gauge-deep-dive)
3. [Finding Key Moments](#3-finding-key-moments)
4. [Activity Timeline](#4-activity-timeline)
5. [Music Analysis for Video Scoring](#5-music-analysis-for-video-scoring)
6. [Podcast Analysis](#6-podcast-analysis)
7. [Mood Classification](#7-mood-classification)
8. [Batch Processing Multiple Files](#8-batch-processing-multiple-files)
9. [Integration with VideoAnalysis](#9-integration-with-videoanalysis)
10. [Custom Analysis Pipeline](#10-custom-analysis-pipeline)
11. [JSON Export and Processing](#11-json-export-and-processing)
12. [Voice Memo Transcription](#12-voice-memo-transcription)

---

## 1. Basic Analysis

The simplest way to analyze audio.

### CLI

```bash
# Analyze a music file
audioanalysis analyze music.mp3

# Output:
# ============================================================
# AUDIO ANALYSIS COMPLETE
# ============================================================
# File: music.mp3
# Duration: 03:45
# Format: mp3 (mp3)
# Sample rate: 44100 Hz
# Bitrate: 320 kbps
# Size: 9.2 MB
#
# Voltage/dB Analysis:
#   Peak: -1.5 dB
#   Average: -12.3 dB
#   Dynamic range: 28.5 dB
#
# Key Moments (5):
#   [00:45] build_start: Build starts, rises 8.2 dB over 12.0s
#   [01:30] peak: Peak at -3.2 dB
#   [02:10] climax: Main climax at -1.5 dB
#   [02:45] drop: Drop of 15.3 dB
#   [03:30] peak: Peak at -5.1 dB
#
# Tempo: 120.0 BPM - Walking pace (moderato)
#
# Mood: Uplifting
# Energy: High
# Arc: Builds from calm to powerful climax
#
# Processing time: 2.3s
```

### Python

```python
from audioanalysis import AudioAnalyzer

analyzer = AudioAnalyzer()
result = analyzer.analyze("music.mp3")

print(f"File: {result.file_name}")
print(f"Duration: {result.duration}")
print(f"Peak dB: {result.peak_db}")
print(f"Mood: {result.mood.primary_mood}")
```

---

## 2. Voltage Gauge Deep Dive

Logan's core innovation - treating audio amplitude like voltage readings.

### CLI

```bash
# Get voltage readings
audioanalysis voltage music.mp3 -s 20

# Output:
# Voltage Analysis for: music.mp3
# ==================================================
# Samples: 4500
# Peak: -1.5 dB
# Average: -12.3 dB
# Dynamic range: 28.5 dB
#
# Sample readings:
#   [00:00]  -30.2 dB [####]
#   [00:01]  -25.3 dB [#####]
#   [00:02]  -18.2 dB [######]
#   [00:03]  -12.5 dB [########]
#   [00:04]   -6.1 dB [#########]
#   [00:05]   -3.2 dB [##########]
#   ...
```

### Python - Understanding the Voltage Gauge

```python
from audioanalysis import VoltageGauge

# Create gauge with 20 samples per second
gauge = VoltageGauge(samples_per_second=20)

# Read "voltage" from audio
readings = gauge.read_voltages("music.mp3")

print(f"Total readings: {len(readings)}")

# Each reading is like a voltage meter snapshot
for reading in readings[:10]:
    # Visual bar for dB level
    bar_length = max(0, int((reading.db_level + 50) / 5))
    bar = '#' * bar_length
    
    print(f"[{reading.timestamp}] "
          f"RMS: {reading.amplitude_rms:.4f} "
          f"→ {reading.db_level:6.1f} dB [{bar}]")

# Get statistics
stats = gauge.get_statistics(readings)
print(f"\nPeak: {stats['peak_db']} dB")
print(f"Average: {stats['average_db']} dB")
print(f"Dynamic range: {stats['dynamic_range_db']} dB")
```

### Understanding dB Values

```python
# dB reference points
from audioanalysis import VoltageGauge

gauge = VoltageGauge()

print("dB Reference Points:")
print(f"  1.0 amplitude → {gauge._amplitude_to_db(1.0):6.1f} dB (maximum)")
print(f"  0.5 amplitude → {gauge._amplitude_to_db(0.5):6.1f} dB (half perceived)")
print(f"  0.1 amplitude → {gauge._amplitude_to_db(0.1):6.1f} dB")
print(f"  0.01 amplitude → {gauge._amplitude_to_db(0.01):6.1f} dB")
print(f"  0.001 amplitude → {gauge._amplitude_to_db(0.001):6.1f} dB")

# Output:
# dB Reference Points:
#   1.0 amplitude →    0.0 dB (maximum)
#   0.5 amplitude →   -6.0 dB (half perceived)
#   0.1 amplitude →  -20.0 dB
#   0.01 amplitude →  -40.0 dB
#   0.001 amplitude → -60.0 dB
```

---

## 3. Finding Key Moments

Detect peaks, drops, builds, and climaxes.

### CLI

```bash
# Find top 10 key moments
audioanalysis moments music.mp3 --top 10 -d 6.0

# Output:
# Key Moments for: music.mp3
# ==================================================
# 1. [00:45] build_start
#    Build starts, rises 8.2 dB over 12.0s
#    Level: -18.5 dB, Delta: +8.2 dB
#
# 2. [01:30] peak
#    Peak at -3.2 dB
#    Level: -3.2 dB, Delta: +6.5 dB
#
# 3. [02:10] climax
#    Main climax at -1.5 dB
#    Level: -1.5 dB, Delta: +12.1 dB
#
# 4. [02:45] drop
#    Drop of 15.3 dB
#    Level: -16.8 dB, Delta: -15.3 dB
```

### Python

```python
from audioanalysis import VoltageGauge, DeltaDetector

gauge = VoltageGauge(samples_per_second=10)
detector = DeltaDetector(threshold_db=6.0)

# Get readings
readings = gauge.read_voltages("music.mp3")

# Find different types of moments
peaks = detector.find_peaks(readings, top_n=5)
drops = detector.find_drops(readings)
builds = detector.find_builds(readings)
climaxes = detector.find_climaxes(readings, peaks)

print("=== PEAKS ===")
for p in peaks:
    print(f"  [{p.timestamp}] {p.db_level:.1f} dB")

print("\n=== DROPS ===")
for d in drops[:5]:
    print(f"  [{d.timestamp}] Drop of {-d.delta_from_previous:.1f} dB")

print("\n=== BUILDS ===")
for b in builds:
    print(f"  [{b.timestamp}] {b.description}")

print("\n=== CLIMAXES ===")
for c in climaxes:
    print(f"  [{c.timestamp}] {c.description}")
```

---

## 4. Activity Timeline

Track energy levels across the audio.

### CLI

```bash
# 10-second buckets
audioanalysis timeline music.mp3 -b 10

# Output:
# Activity Timeline for: music.mp3
# ==================================================
#   00:00-00:10: low      [###       ] avg:-35.2dB max:-28.5dB
#   00:10-00:20: low      [####      ] avg:-30.1dB max:-22.3dB
#   00:20-00:30: medium   [######    ] avg:-18.5dB max:-10.2dB
#   00:30-00:40: medium   [#######   ] avg:-15.2dB max:-8.1dB
#   00:40-00:50: high     [#########] avg:-8.3dB max:-3.5dB
#   00:50-01:00: high     [##########] avg:-5.1dB max:-1.8dB
```

### Python

```python
from audioanalysis import VoltageGauge, DeltaDetector

gauge = VoltageGauge()
detector = DeltaDetector()

readings = gauge.read_voltages("music.mp3")
timeline = detector.get_activity_timeline(readings, bucket_seconds=15)

print("Activity Over Time:")
print("=" * 60)

for bucket in timeline:
    # Create visual bar
    bar_len = max(0, int((bucket.avg_db + 50) / 5))
    bar = '#' * bar_len
    
    # Color coding (conceptual)
    level_icon = {
        'high': '🔊',
        'medium': '🔉',
        'low': '🔈',
        'silent': '🔇'
    }.get(bucket.activity_level, '❓')
    
    print(f"{level_icon} {bucket.start_time}-{bucket.end_time}: "
          f"{bucket.activity_level:8s} [{bar:<10}] "
          f"avg:{bucket.avg_db:.1f}dB")
```

---

## 5. Music Analysis for Video Scoring

Find the perfect music for your video (WSL_CLIO's original use case!).

### Python

```python
from audioanalysis import AudioAnalyzer

analyzer = AudioAnalyzer()

# Analyze music track
result = analyzer.analyze("inspiring-cinematic-music.mp3")

print("=== MUSIC ANALYSIS FOR VIDEO SCORING ===")
print(f"Track: {result.file_name}")
print(f"Duration: {result.duration}")
print()

# Suitability analysis
print("SUITABILITY ASSESSMENT:")
print("-" * 40)

# Energy check
if result.mood and result.mood.energy_level == "High":
    print("✅ High energy - good for action/triumph")
elif result.mood and result.mood.energy_level == "Low":
    print("✅ Low energy - good for calm/reflective")
else:
    print("✅ Medium energy - versatile")

# Arc check
if result.mood and "climax" in result.mood.emotional_arc.lower():
    print("✅ Has climax - great for 'breakthrough' moments")

if result.mood and "build" in result.mood.emotional_arc.lower():
    print("✅ Builds intensity - perfect for narrative arc")

# Tempo check
if result.tempo:
    if 100 <= result.tempo.bpm <= 140:
        print(f"✅ {result.tempo.bpm} BPM - keeps energy without overwhelming")
    elif result.tempo.bpm < 100:
        print(f"✅ {result.tempo.bpm} BPM - calm, contemplative pace")
    else:
        print(f"✅ {result.tempo.bpm} BPM - high energy, action-oriented")

# Key moments for sync
print()
print("KEY MOMENTS TO SYNC WITH VIDEO:")
print("-" * 40)
for moment in result.key_moments:
    if moment.moment_type == 'climax':
        print(f"🎯 [{moment.timestamp}] CLIMAX - sync with key reveal!")
    elif moment.moment_type == 'build_start':
        print(f"📈 [{moment.timestamp}] Build starts - start building tension")
    elif moment.moment_type == 'drop':
        print(f"📉 [{moment.timestamp}] Drop - use for emotional beat")
    elif moment.moment_type == 'peak':
        print(f"🔝 [{moment.timestamp}] Peak - impactful moment")

print()
print(f"Summary: {result.summary}")
```

---

## 6. Podcast Analysis

Analyze speech content with transcription.

### CLI

```bash
audioanalysis analyze podcast.mp3 --speech --type comprehensive

# Output includes speech detection and transcript
```

### Python

```python
from audioanalysis import AudioAnalyzer

# Enable speech detection
analyzer = AudioAnalyzer(
    samples_per_second=5,      # Less granular for speech
    delta_threshold_db=10.0,   # Higher threshold
    detect_speech=True         # Enable transcription
)

result = analyzer.analyze("podcast.mp3")

print(f"Podcast Analysis: {result.file_name}")
print(f"Duration: {result.duration}")
print()

# Check for speech
if result.speech:
    if result.speech.has_speech:
        print("Speech detected!")
        print(f"Speech ratio: {result.speech.speech_ratio:.1%}")
        print()
        print("Transcript:")
        print("-" * 40)
        print(result.speech.full_transcript[:500])
        print("...")
    else:
        print("No speech detected (music/ambient only)")
else:
    print("Speech detection not available")

# Activity for finding quiet/loud sections
print()
print("Audio Activity:")
for bucket in result.activity_timeline[:10]:
    print(f"  {bucket.start_time}-{bucket.end_time}: {bucket.activity_level}")
```

---

## 7. Mood Classification

Understand the emotional characteristics of audio.

### Python

```python
from audioanalysis import AudioAnalyzer, VoltageGauge, DeltaDetector, MoodClassifier

# Full analysis approach
analyzer = AudioAnalyzer()
result = analyzer.analyze("music.mp3")

if result.mood:
    print("=== MOOD ANALYSIS ===")
    print(f"Primary Mood: {result.mood.primary_mood}")
    print(f"Energy Level: {result.mood.energy_level}")
    print(f"Emotional Arc: {result.mood.emotional_arc}")
    print(f"Confidence: {result.mood.confidence:.0%}")
    print()
    print("Characteristics:")
    for char in result.mood.characteristics:
        print(f"  • {char}")

# Manual mood classification
print()
print("=== MANUAL CLASSIFICATION ===")

gauge = VoltageGauge()
detector = DeltaDetector()
classifier = MoodClassifier()

readings = gauge.read_voltages("music.mp3")
moments = detector.get_all_key_moments(readings)

# Classify without tempo (simpler)
mood = classifier.classify(readings, None, moments)

print(f"Mood (no tempo): {mood.primary_mood}")
print(f"Energy: {mood.energy_level}")
```

---

## 8. Batch Processing Multiple Files

Analyze multiple audio files efficiently.

### Python

```python
from pathlib import Path
from audioanalysis import AudioAnalyzer
import json

# Find all audio files
audio_dir = Path("./music")
audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

print(f"Found {len(audio_files)} audio files")

analyzer = AudioAnalyzer(detect_tempo=True, detect_speech=False)

results = []
for audio_path in audio_files:
    print(f"Analyzing: {audio_path.name}...", end=" ")
    
    try:
        result = analyzer.analyze(str(audio_path), analysis_type="quick")
        results.append({
            "file": audio_path.name,
            "duration": result.duration,
            "peak_db": result.peak_db,
            "avg_db": result.average_db,
            "mood": result.mood.primary_mood if result.mood else "Unknown",
            "key_moments": len(result.key_moments)
        })
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")
        results.append({"file": audio_path.name, "error": str(e)})

# Save batch results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("=== SUMMARY ===")
successful = [r for r in results if "error" not in r]
print(f"Successfully analyzed: {len(successful)}/{len(audio_files)}")

# Sort by energy
print()
print("Files by average dB (loudest first):")
for r in sorted(successful, key=lambda x: x["avg_db"], reverse=True)[:5]:
    print(f"  {r['file']}: {r['avg_db']:.1f} dB ({r['mood']})")
```

---

## 9. Integration with VideoAnalysis

Combine audio and video analysis for comprehensive media analysis.

### Python

```python
from audioanalysis import AudioAnalyzer
# Assuming VideoAnalysis is available
# from videoanalysis import VideoAnalyzer

def analyze_media_file(video_path: str):
    """Analyze both video and audio components."""
    
    print(f"=== COMPREHENSIVE MEDIA ANALYSIS ===")
    print(f"File: {video_path}")
    print()
    
    # Audio analysis
    print("--- AUDIO ANALYSIS ---")
    audio_analyzer = AudioAnalyzer()
    audio_result = audio_analyzer.analyze(video_path)
    
    print(f"Duration: {audio_result.duration}")
    print(f"Audio: Peak {audio_result.peak_db} dB, Avg {audio_result.average_db} dB")
    
    if audio_result.mood:
        print(f"Mood: {audio_result.mood.primary_mood} ({audio_result.mood.energy_level})")
    
    if audio_result.tempo:
        print(f"Tempo: {audio_result.tempo.bpm} BPM")
    
    print()
    print("Audio Key Moments:")
    for m in audio_result.key_moments[:5]:
        print(f"  [{m.timestamp}] {m.moment_type}: {m.db_level:.1f} dB")
    
    # Video analysis would go here
    # video_result = video_analyzer.analyze(video_path)
    
    # Sync audio peaks with video scene changes
    print()
    print("--- SYNC OPPORTUNITIES ---")
    # Match audio climax to video key frame, etc.
    
    return audio_result  # , video_result

# Usage
result = analyze_media_file("demo_video.mp4")
```

---

## 10. Custom Analysis Pipeline

Build your own specialized analysis.

### Python

```python
from audioanalysis import (
    VoltageGauge,
    DeltaDetector,
    MetadataExtractor,
    MoodClassifier
)
from dataclasses import dataclass
from typing import List

@dataclass
class MusicFitScore:
    """Score how well music fits a specific use case."""
    video_type: str
    score: float  # 0-100
    reasons: List[str]

def score_music_for_use_case(audio_path: str, use_case: str) -> MusicFitScore:
    """Score how well audio fits a specific use case."""
    
    # Extract data
    metadata = MetadataExtractor.extract(audio_path)
    gauge = VoltageGauge(samples_per_second=10)
    detector = DeltaDetector(threshold_db=6.0)
    classifier = MoodClassifier()
    
    readings = gauge.read_voltages(audio_path)
    stats = gauge.get_statistics(readings)
    moments = detector.get_all_key_moments(readings)
    mood = classifier.classify(readings, None, moments)
    
    score = 50  # Start neutral
    reasons = []
    
    if use_case == "action":
        # Action videos need high energy
        if mood.energy_level == "High":
            score += 30
            reasons.append("High energy matches action content")
        if stats['dynamic_range_db'] > 20:
            score += 10
            reasons.append("Wide dynamic range adds intensity")
        if len([m for m in moments if m.moment_type == 'peak']) > 5:
            score += 10
            reasons.append("Multiple peaks for action cuts")
            
    elif use_case == "documentary":
        # Documentaries need steady, not distracting
        if mood.energy_level in ["Low", "Medium"]:
            score += 20
            reasons.append("Moderate energy won't distract")
        if stats['dynamic_range_db'] < 15:
            score += 15
            reasons.append("Consistent dynamics for narration")
        if not [m for m in moments if m.moment_type == 'climax']:
            score += 15
            reasons.append("No dramatic climaxes to distract")
            
    elif use_case == "triumph":
        # Triumph/success videos need builds and climax
        builds = [m for m in moments if m.moment_type == 'build_start']
        climaxes = [m for m in moments if m.moment_type == 'climax']
        
        if builds:
            score += 25
            reasons.append(f"Has {len(builds)} building sections")
        if climaxes:
            score += 25
            reasons.append("Has clear climax for 'success moment'")
        if "Uplifting" in mood.primary_mood:
            score += 20
            reasons.append("Uplifting mood matches triumph")
    
    return MusicFitScore(
        video_type=use_case,
        score=min(100, max(0, score)),
        reasons=reasons
    )

# Usage
fit = score_music_for_use_case("inspiring-cinematic.mp3", "triumph")
print(f"Score for '{fit.video_type}': {fit.score}/100")
print("Reasons:")
for r in fit.reasons:
    print(f"  ✓ {r}")
```

---

## 11. JSON Export and Processing

Export and process analysis results.

### CLI

```bash
# Export to JSON
audioanalysis analyze music.mp3 -o analysis.json
```

### Python - Export and Process

```python
from audioanalysis import AudioAnalyzer
from dataclasses import asdict
import json

# Analyze
analyzer = AudioAnalyzer()
result = analyzer.analyze("music.mp3")

# Convert to dict
data = asdict(result)

# Save to JSON
with open("analysis.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved to analysis.json")

# Process JSON data
with open("analysis.json") as f:
    loaded = json.load(f)

# Extract specific information
print(f"\nLoaded analysis for: {loaded['file_name']}")
print(f"Duration: {loaded['duration']}")

# Find loudest moment
if loaded['key_moments']:
    loudest = max(loaded['key_moments'], key=lambda m: m['db_level'])
    print(f"Loudest moment: {loudest['timestamp']} at {loudest['db_level']} dB")

# Count activity levels
activity_counts = {}
for bucket in loaded['activity_timeline']:
    level = bucket['activity_level']
    activity_counts[level] = activity_counts.get(level, 0) + 1

print("Activity distribution:")
for level, count in sorted(activity_counts.items()):
    print(f"  {level}: {count} buckets")
```

---

## 12. Voice Memo Transcription

Transcribe voice recordings.

### CLI

```bash
audioanalysis analyze voice_memo.m4a --speech
```

### Python

```python
from audioanalysis import AudioAnalyzer

# Enable speech detection
analyzer = AudioAnalyzer(
    samples_per_second=2,     # Lower for speech
    detect_tempo=False,        # Not relevant
    detect_speech=True         # Enable transcription
)

result = analyzer.analyze("voice_memo.m4a")

print("=== VOICE MEMO ANALYSIS ===")
print(f"Duration: {result.duration}")
print()

if result.speech and result.speech.has_speech:
    print("TRANSCRIPT:")
    print("-" * 40)
    print(result.speech.full_transcript)
    print("-" * 40)
    
    # Save transcript
    with open("transcript.txt", "w") as f:
        f.write(result.speech.full_transcript)
    print("\nSaved to transcript.txt")
else:
    print("No speech detected in recording")

# Activity timeline shows speaking patterns
print()
print("Speaking patterns (by activity):")
for bucket in result.activity_timeline:
    if bucket.activity_level in ['medium', 'high']:
        print(f"  [{bucket.start_time}-{bucket.end_time}] Speaking")
    else:
        print(f"  [{bucket.start_time}-{bucket.end_time}] Pause/Silence")
```

---

## Summary

AudioAnalysis provides flexible tools for any audio analysis task:

| Use Case | Best Approach |
|----------|---------------|
| Quick overview | `audioanalysis analyze file.mp3` |
| Detailed voltage | `audioanalysis voltage file.mp3 -s 20` |
| Find moments | `audioanalysis moments file.mp3 --top 10` |
| Track activity | `audioanalysis timeline file.mp3 -b 10` |
| Music for video | Full analysis + mood + key moments |
| Speech/podcast | `--speech` flag enabled |
| Batch processing | Python API loop |
| Custom scoring | Build pipeline from components |

---

**Requested by:** Logan Smith (via WSL_CLIO)
**Voltage Gauge Concept:** Logan Smith
**Built by:** ATLAS (Team Brain)

**For the Maximum Benefit of Life.**
*Together for all time!*
