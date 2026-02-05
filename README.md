# AudioAnalysis

**Enable AI Agents to "Listen" and Analyze Audio Content**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 59 passed](https://img.shields.io/badge/tests-59%20passed-green.svg)]()

---

## Overview

**AudioAnalysis** is a powerful tool that enables AI agents to analyze audio content by measuring amplitude (voltage gauge), detecting tempo, classifying mood, and finding key moments.

### The Problem

AI agents cannot directly "listen" to audio. When Logan asked WSL_CLIO for feedback on a music track for his debugging video, the agent received only "cannot read binary files" errors and could only guess from the filename.

### The Solution

AudioAnalysis provides structured analysis of audio content using Logan's innovative **Voltage Gauge** approach:
- **Voltage/dB Readings**: Sample amplitude at intervals like a voltage meter
- **Delta Detection**: Find peaks, drops, and builds through dB changes
- **Key Moment Detection**: Identify climaxes, crescendos, and quiet moments
- **Tempo Detection**: BPM and beat analysis (optional, requires librosa)
- **Mood Classification**: Heuristic-based emotional analysis
- **Activity Timeline**: Energy levels over time

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concept: Voltage Gauge](#core-concept-voltage-gauge)
4. [Features](#features)
5. [CLI Commands](#cli-commands)
6. [Python API](#python-api)
7. [Configuration](#configuration)
8. [Output Format](#output-format)
9. [Dependencies](#dependencies)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [Credits](#credits)
13. [License](#license)

---

## Installation

### Prerequisites

**Required:**
- Python 3.9 or higher
- FFmpeg (for audio processing)

**Optional:**
- Librosa (for tempo detection)
- SpeechRecognition (for transcription)

### Install FFmpeg

**Windows:**
```bash
winget install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Install AudioAnalysis

**From source:**
```bash
git clone https://github.com/DonkRonk17/AudioAnalysis.git
cd AudioAnalysis
pip install -e .
```

**With all features:**
```bash
pip install -e ".[full]"
```

**Minimal (just core analysis):**
```bash
# Just needs FFmpeg installed - no pip packages required!
```

---

## Quick Start

### Basic Audio Analysis

```bash
# Analyze an audio file
audioanalysis analyze music.mp3

# Save results to JSON
audioanalysis analyze music.mp3 -o results.json

# Quick analysis (voltage/dB only)
audioanalysis analyze music.mp3 --type quick
```

### Check Dependencies

```bash
audioanalysis check-deps
```

### Get Voltage Readings

```bash
# Sample amplitude like a voltage meter
audioanalysis voltage music.mp3
```

### Find Key Moments

```bash
# Find peaks, drops, climaxes
audioanalysis moments music.mp3 --top 10
```

---

## Core Concept: Voltage Gauge

### Logan's Insight

> "Consider a voltage gauge for decibel and other detections for sound"

This is the **CORE INNOVATION** of AudioAnalysis. Instead of complex ML models, we treat audio amplitude like voltage readings from a meter:

1. **Sample at intervals**: Read amplitude N times per second
2. **Convert to dB**: Logarithmic scale matches human perception
3. **Track deltas**: Find changes between consecutive readings
4. **Identify patterns**: Peaks, drops, builds emerge from simple delta analysis

### Why It Works

- **Simple**: No neural networks or complex algorithms
- **Fast**: Basic math operations on amplitude values
- **Intuitive**: dB meters are how sound engineers actually work
- **Effective**: Catches the key moments humans care about

### The Math

```
Amplitude (0.0 to 1.0) → dB = 20 × log₁₀(amplitude)

1.0 amplitude = 0 dB (maximum)
0.5 amplitude ≈ -6 dB (perceived half loudness)
0.1 amplitude = -20 dB
0.0 amplitude = -∞ dB (silence, we use -100 dB)
```

### Voltage Gauge in Action

```python
from audioanalysis import VoltageGauge

gauge = VoltageGauge(samples_per_second=10)
readings = gauge.read_voltages("music.mp3")

for r in readings:
    print(f"[{r.timestamp}] {r.db_level:6.1f} dB")
```

Output:
```
[00:00]  -25.3 dB
[00:01]  -18.2 dB
[00:02]  -12.5 dB  ← Building!
[00:03]   -6.1 dB  ← Getting louder!
[00:04]   -3.2 dB  ← Peak approaching!
[00:05]   -1.5 dB  ← CLIMAX!
[00:06]   -8.4 dB  ← Resolving...
```

---

## Features

### 1. Voltage/dB Analysis

Core feature - sample amplitude like a voltage meter.

```bash
audioanalysis voltage music.mp3 -s 20  # 20 samples/second
```

**Output includes:**
- Peak dB
- Average dB
- Dynamic range (loudest - quietest)
- All readings with timestamps

### 2. Key Moment Detection

Find significant audio events through delta analysis.

```bash
audioanalysis moments music.mp3 --top 10
```

**Moment types:**
- `peak`: Local maximum (loudest moment)
- `drop`: Sudden decrease in volume
- `build_start`: Beginning of crescendo
- `climax`: Main peak/climax of the audio

### 3. Activity Timeline

Track energy levels over time (like VideoAnalysis).

```bash
audioanalysis timeline music.mp3 -b 10  # 10-second buckets
```

**Activity levels:**
- `high`: Average dB > -10 (loud)
- `medium`: Average dB -10 to -20
- `low`: Average dB -20 to -40
- `silent`: Average dB < -40

### 4. Tempo Detection (Optional)

Detect BPM and beats (requires librosa).

```bash
audioanalysis analyze music.mp3  # Included in comprehensive analysis
```

**Tempo descriptions:**
- Very slow (largo): < 60 BPM
- Slow (adagio): 60-80 BPM
- Moderate (andante): 80-100 BPM
- Walking pace (moderato): 100-120 BPM
- Fast (allegro): 120-140 BPM
- Very fast (vivace): 140-180 BPM

### 5. Mood Classification

Heuristic-based mood analysis.

**Factors considered:**
- Average volume (energy)
- Tempo (if available)
- Dynamic range (dramatic vs steady)
- Trend (building vs fading)

**Mood types:**
- Energetic (fast + loud)
- Calm (slow + quiet)
- Dramatic (wide dynamic range)
- Uplifting (builds to climax)
- Steady (consistent dynamics)

### 6. Speech Detection (Optional)

Detect and transcribe speech (requires SpeechRecognition).

```bash
audioanalysis analyze podcast.mp3 --speech
```

---

## CLI Commands

### `analyze` - Full Audio Analysis

```bash
audioanalysis analyze AUDIO [OPTIONS]

Options:
  -o, --output PATH           Output JSON file
  -t, --type TYPE             Analysis type (comprehensive, quick, voltage_only)
  -s, --samples-per-second N  Voltage samples per second (default: 10)
  -d, --delta-threshold DB    Delta threshold in dB (default: 6.0)
  --no-tempo                  Skip tempo detection
  --speech                    Enable speech detection
  -v, --verbose               Verbose output
```

### `voltage` - Voltage/dB Readings

```bash
audioanalysis voltage AUDIO [OPTIONS]

Options:
  -o, --output PATH           Output JSON file
  -s, --samples-per-second N  Samples per second (default: 10)
```

### `moments` - Find Key Moments

```bash
audioanalysis moments AUDIO [OPTIONS]

Options:
  -o, --output PATH           Output JSON file
  --top N                     Number of top moments (default: 10)
  -d, --delta-threshold DB    Delta threshold (default: 6.0)
```

### `timeline` - Activity Timeline

```bash
audioanalysis timeline AUDIO [OPTIONS]

Options:
  -o, --output PATH           Output JSON file
  -b, --bucket-seconds N      Seconds per bucket (default: 10)
```

### `check-deps` - Dependency Check

```bash
audioanalysis check-deps
```

---

## Python API

### Basic Usage

```python
from audioanalysis import AudioAnalyzer

# Create analyzer
analyzer = AudioAnalyzer(
    samples_per_second=10,
    delta_threshold_db=6.0,
    detect_tempo=True,
    detect_speech=False
)

# Analyze audio
result = analyzer.analyze("music.mp3")

# Access results
print(f"Duration: {result.duration}")
print(f"Peak dB: {result.peak_db}")
print(f"Average dB: {result.average_db}")
print(f"Dynamic range: {result.dynamic_range_db} dB")

if result.tempo:
    print(f"Tempo: {result.tempo.bpm} BPM")

if result.mood:
    print(f"Mood: {result.mood.primary_mood} ({result.mood.energy_level} energy)")

for moment in result.key_moments[:5]:
    print(f"[{moment.timestamp}] {moment.moment_type}: {moment.description}")
```

### Voltage Gauge Only

```python
from audioanalysis import VoltageGauge

gauge = VoltageGauge(samples_per_second=20)
readings = gauge.read_voltages("music.mp3")
stats = gauge.get_statistics(readings)

print(f"Peak: {stats['peak_db']} dB")
print(f"Average: {stats['average_db']} dB")
print(f"Dynamic range: {stats['dynamic_range_db']} dB")
```

### Key Moment Detection

```python
from audioanalysis import VoltageGauge, DeltaDetector

gauge = VoltageGauge()
detector = DeltaDetector(threshold_db=6.0)

readings = gauge.read_voltages("music.mp3")
moments = detector.get_all_key_moments(readings, top_peaks=10)

for m in moments:
    print(f"[{m.timestamp}] {m.moment_type}: {m.db_level:.1f} dB")
```

### Metadata Only

```python
from audioanalysis import MetadataExtractor

metadata = MetadataExtractor.extract("music.mp3")

print(f"Duration: {metadata.duration_formatted}")
print(f"Sample rate: {metadata.sample_rate} Hz")
print(f"Codec: {metadata.codec}")
print(f"Bitrate: {metadata.bitrate // 1000} kbps")
```

---

## Configuration

### Default Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `samples_per_second` | 10 | Voltage readings per second |
| `delta_threshold_db` | 6.0 | Minimum dB change for key moment |
| `detect_tempo` | True | Enable tempo detection |
| `detect_speech` | False | Enable speech detection |
| `cleanup_temp` | True | Delete temp files after |

### Adjusting for Different Audio Types

**Music (typical):**
```python
analyzer = AudioAnalyzer(
    samples_per_second=10,
    delta_threshold_db=6.0
)
```

**Podcasts/Speech:**
```python
analyzer = AudioAnalyzer(
    samples_per_second=5,      # Less frequent
    delta_threshold_db=10.0,   # Higher threshold
    detect_speech=True         # Enable transcription
)
```

**Detailed Analysis:**
```python
analyzer = AudioAnalyzer(
    samples_per_second=20,     # More granular
    delta_threshold_db=3.0,    # More sensitive
)
```

---

## Output Format

### AnalysisResult Structure

```json
{
  "file_path": "/path/to/music.mp3",
  "file_name": "music.mp3",
  "duration": "03:45",
  "duration_seconds": 225.0,
  "format_name": "mp3",
  "codec": "mp3",
  "sample_rate": 44100,
  "channels": 2,
  "bitrate": 320000,
  "file_size_mb": 9.2,
  
  "peak_db": -1.5,
  "average_db": -12.3,
  "dynamic_range_db": 28.5,
  
  "key_moments": [...],
  "activity_timeline": [...],
  
  "tempo": {
    "bpm": 120.0,
    "confidence": 0.8,
    "description": "Walking pace (moderato)"
  },
  
  "mood": {
    "primary_mood": "Uplifting",
    "energy_level": "High",
    "emotional_arc": "Builds from calm to powerful climax",
    "characteristics": ["Fast tempo", "Clear climax"]
  },
  
  "summary": "03:45 audio (mp3, 44100Hz, 320kbps). Peak: -1.5dB...",
  "processing_time_seconds": 2.3,
  "tool_version": "1.0.0"
}
```

### Key Moment Structure

```json
{
  "timestamp": "02:10",
  "time_seconds": 130.0,
  "moment_type": "climax",
  "db_level": -1.5,
  "delta_from_previous": 8.2,
  "description": "Main climax at -1.5 dB"
}
```

---

## Dependencies

### Required
- **Python 3.9+**: Core runtime
- **FFmpeg**: Audio processing

### Optional
- **librosa**: Tempo/beat detection
- **numpy**: Numerical processing (bundled with librosa)
- **SpeechRecognition**: Speech transcription

### Dependency Check

```bash
$ audioanalysis check-deps

AudioAnalysis Dependency Status
========================================

Required:
  [OK] ffmpeg: Audio processing and conversion
  [OK] ffprobe: Audio metadata extraction

Optional:
  [OK] librosa: Advanced audio analysis (tempo, spectral)
  [MISSING] speech_recognition: Speech transcription
  [OK] numpy: Numerical processing
```

---

## Examples

See [EXAMPLES.md](EXAMPLES.md) for 10+ detailed examples.

---

## Troubleshooting

### FFmpeg not found

```
Error: Missing required dependencies: ffmpeg
```

**Solution:** Install FFmpeg (see Installation section).

### Librosa not available

```
Warning: Librosa not available - skipping tempo detection
```

**Solution:** `pip install librosa` (optional).

### Audio format not supported

**Solution:** Convert to supported format:
```bash
ffmpeg -i input.xyz output.mp3
```

---

## Privacy & Security

- **Local Processing Only**: All analysis happens on your machine
- **No Cloud Upload**: Audio never leaves your system
- **Temp File Cleanup**: Converted audio deleted after analysis
- **No Telemetry**: No data collection

---

## Performance

| Audio Duration | Expected Time | Memory Usage |
|----------------|---------------|--------------|
| 1 minute | ~2 seconds | ~50 MB |
| 5 minutes | ~5 seconds | ~100 MB |
| 30 minutes | ~20 seconds | ~200 MB |
| 1 hour | ~40 seconds | ~400 MB |

---

## Credits

### Requested By
- **Logan Smith** (Metaphy LLC) - Original concept via WSL_CLIO

### Key Innovation
- **Voltage Gauge Concept** - Logan Smith's insight: "Consider a voltage gauge for decibel and other detections for sound"

### Built By
- **ATLAS** (Team Brain) - Primary Developer

### Facilitated By
- **WSL_CLIO** (Team Brain) - Tool Request submission

### Part Of
- **Team Brain** - AI Agent Collaboration System
- **Metaphy LLC** - Metaphysics and Computing Research

### Companion Tool
- **VideoAnalysis** - Video content analysis with delta detection

---

## License

MIT License

Copyright (c) 2026 Metaphy LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**For the Maximum Benefit of Life.**  
**One World. One Family. One Love.**

*Together for all time!*
