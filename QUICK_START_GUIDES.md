# AudioAnalysis - Quick Start Guides

## 📖 ABOUT THESE GUIDES

Each Team Brain agent has a **5-minute quick-start guide** tailored to their role and workflows. AudioAnalysis enables AI agents to "listen" to audio content through voltage readings, key moment detection, mood classification, and more.

**Logan's Core Insight:** "Consider a voltage gauge for decibel and other detections for sound" - treating audio amplitude like voltage readings from a meter for simple, direct measurement.

**Choose your guide:**
- [Forge (Orchestrator)](#-forge-quick-start)
- [Atlas (Executor)](#-atlas-quick-start)
- [Clio (Linux Agent)](#-clio-quick-start)
- [Nexus (Multi-Platform)](#-nexus-quick-start)
- [Bolt (Free Executor)](#-bolt-quick-start)
- [Logan (The Architect)](#-logan-quick-start)

---

## 🔥 FORGE QUICK START

**Role:** Orchestrator / Reviewer  
**Time:** 5 minutes  
**Goal:** Learn to use AudioAnalysis for reviewing audio content quality

### Step 1: Installation Check

```bash
# Verify AudioAnalysis is available
cd C:\Users\logan\OneDrive\Documents\AutoProjects\AudioAnalysis
python audioanalysis.py check-deps
```

**Expected Output:**
```
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

### Step 2: First Use - Analyze Audio

```python
# In your Forge session
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")
from AudioAnalysis.audioanalysis import AudioAnalyzer

# Quick analysis for review purposes
analyzer = AudioAnalyzer(detect_tempo=True, detect_speech=False)
result = analyzer.analyze("path/to/audio.mp3", analysis_type="quick")

print(f"File: {result.file_name}")
print(f"Duration: {result.duration}")
print(f"Peak Level: {result.peak_db} dB")
print(f"Average Level: {result.average_db} dB")
if result.mood:
    print(f"Mood: {result.mood.primary_mood}")
```

### Step 3: Quality Review Workflow

**Use Case 1: Check Audio Quality Before Session Review**

```python
def review_audio_quality(audio_path: str) -> str:
    """Quick quality assessment for audio files."""
    analyzer = AudioAnalyzer()
    result = analyzer.analyze(audio_path, analysis_type="quick")
    
    issues = []
    recommendations = []
    
    # Check for clipping
    if result.peak_db > -1:
        issues.append(f"[!] Possible clipping (peak: {result.peak_db}dB)")
        recommendations.append("Consider reducing gain")
    
    # Check for too quiet
    if result.average_db < -25:
        issues.append(f"[!] Audio may be too quiet (avg: {result.average_db}dB)")
        recommendations.append("Consider normalizing audio")
    
    # Check dynamic range
    if result.dynamic_range_db > 40:
        issues.append(f"[!] Very wide dynamic range ({result.dynamic_range_db}dB)")
        recommendations.append("Consider compression for consistent levels")
    
    # Build report
    report = [
        f"Audio Quality Review: {result.file_name}",
        f"=" * 50,
        f"Duration: {result.duration}",
        f"Format: {result.format_name} ({result.codec})",
        f"Peak: {result.peak_db}dB | Avg: {result.average_db}dB | Range: {result.dynamic_range_db}dB",
        f"",
    ]
    
    if issues:
        report.append("Issues Found:")
        report.extend(f"  {i}" for i in issues)
        report.append("")
        report.append("Recommendations:")
        report.extend(f"  - {r}" for r in recommendations)
    else:
        report.append("[OK] Audio quality looks good!")
    
    return "\n".join(report)

# Usage
print(review_audio_quality("demo_narration.mp3"))
```

**Use Case 2: Review Music for Debugging Video**

```python
# Forge reviewing background music
result = analyzer.analyze("background_music.mp3")

print(f"""
Music Review Summary:
- Tempo: {result.tempo.bpm if result.tempo else 'N/A'} BPM
- Mood: {result.mood.primary_mood if result.mood else 'N/A'}
- Energy: {result.mood.energy_level if result.mood else 'N/A'}
- Arc: {result.mood.emotional_arc if result.mood else 'N/A'}
- Key Moments: {len(result.key_moments)} detected
""")
```

### Step 4: Common Forge Commands

```bash
# CLI quick analysis
python audioanalysis.py analyze music.mp3 -t quick

# Get key moments only
python audioanalysis.py moments music.mp3 --top 5

# Full analysis with JSON output
python audioanalysis.py analyze podcast.mp3 -o podcast_analysis.json
```

### Next Steps for Forge
1. Read [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) - Forge section
2. Try [EXAMPLES.md](EXAMPLES.md) - Quality review examples
3. Add audio review to your session review checklist

---

## ⚡ ATLAS QUICK START

**Role:** Executor / Builder  
**Time:** 5 minutes  
**Goal:** Learn to use AudioAnalysis in tool development and testing

### Step 1: Installation Check

```bash
cd C:\Users\logan\OneDrive\Documents\AutoProjects\AudioAnalysis
python -c "from audioanalysis import AudioAnalyzer; print('[OK] AudioAnalysis imported')"
```

### Step 2: First Use - Build Integration

```python
# In your Atlas session
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")
from AudioAnalysis.audioanalysis import AudioAnalyzer, DependencyChecker

# Check dependencies during tool builds
def verify_audio_env():
    """Verify audio processing environment."""
    deps = DependencyChecker.check_all(include_optional=False)
    
    if not deps['ffmpeg']:
        print("[X] FFmpeg not found!")
        print("    Install: winget install ffmpeg")
        return False
    
    print("[OK] Audio environment verified")
    return True

# Verify before building audio-related tools
verify_audio_env()
```

### Step 3: Integration with Build Workflows

**During Tool Creation:**

```python
# When building a tool that processes audio
from AudioAnalysis.audioanalysis import (
    AudioAnalyzer, 
    AnalysisResult,
    DependencyChecker
)

class MyAudioTool:
    """Example tool using AudioAnalysis."""
    
    def __init__(self):
        # Verify dependencies at init
        DependencyChecker.verify_required()
        self.analyzer = AudioAnalyzer()
    
    def process(self, audio_path: str) -> dict:
        """Process audio file."""
        result = self.analyzer.analyze(audio_path, analysis_type="quick")
        
        return {
            'duration': result.duration_seconds,
            'peak_db': result.peak_db,
            'mood': result.mood.primary_mood if result.mood else None,
            'key_moments': len(result.key_moments)
        }
```

**Test Suite Integration:**

```python
import unittest
from AudioAnalysis.audioanalysis import AudioAnalyzer, AudioNotFoundError

class TestMyAudioTool(unittest.TestCase):
    """Test audio tool functionality."""
    
    def setUp(self):
        self.analyzer = AudioAnalyzer()
    
    def test_handles_missing_file(self):
        """Test error handling for missing files."""
        with self.assertRaises(AudioNotFoundError):
            self.analyzer.analyze("/nonexistent/audio.mp3")
    
    def test_voltage_readings_captured(self):
        """Test voltage readings are captured."""
        # Create test audio first or use mock
        # result = self.analyzer.analyze("test.mp3")
        # self.assertGreater(len(result.voltage_readings), 0)
        pass
```

### Step 4: Common Atlas Commands

```bash
# Check all dependencies (including optional)
python audioanalysis.py check-deps

# Voltage readings analysis (core feature)
python audioanalysis.py voltage test_audio.mp3

# Generate test data for documentation
python audioanalysis.py analyze sample.mp3 -o sample_output.json
```

### Next Steps for Atlas
1. Integrate into Holy Grail automation
2. Add to tool build checklist
3. Use for testing audio features in tools

---

## 🐧 CLIO QUICK START

**Role:** Linux / Ubuntu Agent  
**Time:** 5 minutes  
**Goal:** Learn to use AudioAnalysis in Linux environment for automation

### Step 1: Linux Installation

```bash
# Install FFmpeg (required)
sudo apt update
sudo apt install -y ffmpeg

# Clone AudioAnalysis (if not present)
cd ~/OneDrive/Documents/AutoProjects
# AudioAnalysis should already be there

# Install optional dependencies
pip3 install librosa numpy

# Verify installation
python3 -c "from audioanalysis import AudioAnalyzer; print('[OK] AudioAnalysis ready')"
```

### Step 2: First Use - CLI Analysis

```bash
# Quick analysis from command line
cd ~/OneDrive/Documents/AutoProjects/AudioAnalysis

# Basic analysis
python3 audioanalysis.py analyze /path/to/audio.mp3

# Voltage readings only (fastest)
python3 audioanalysis.py voltage /path/to/audio.mp3

# Key moments detection
python3 audioanalysis.py moments /path/to/audio.mp3 --top 10
```

### Step 3: Integration with Clio Workflows

**Bash Script for Batch Processing:**

```bash
#!/bin/bash
# clio_audio_batch.sh - Batch audio analysis

AUDIO_DIR="${1:-.}"
OUTPUT_DIR="${2:-./analysis_output}"

mkdir -p "$OUTPUT_DIR"

echo "Analyzing audio files in: $AUDIO_DIR"
echo "Output to: $OUTPUT_DIR"
echo "================================"

for audio_file in "$AUDIO_DIR"/*.{mp3,wav,flac,ogg} 2>/dev/null; do
    [ -f "$audio_file" ] || continue
    
    filename=$(basename "$audio_file")
    name="${filename%.*}"
    
    echo "Processing: $filename"
    
    python3 ~/OneDrive/Documents/AutoProjects/AudioAnalysis/audioanalysis.py \
        analyze "$audio_file" \
        -t quick \
        -o "$OUTPUT_DIR/${name}_analysis.json"
    
    if [ $? -eq 0 ]; then
        echo "  [OK] $filename analyzed"
    else
        echo "  [X] Failed: $filename"
    fi
done

echo "================================"
echo "Batch analysis complete!"
ls -la "$OUTPUT_DIR"
```

**Usage:**
```bash
chmod +x clio_audio_batch.sh
./clio_audio_batch.sh /home/logan/music ./analysis_results
```

**Cron Job for Scheduled Analysis:**

```bash
# Add to crontab: crontab -e
# Analyze new audio files every hour
0 * * * * /home/logan/scripts/clio_audio_batch.sh /home/logan/inbox /home/logan/analyzed
```

### Step 4: Common Clio Commands

```bash
# Full comprehensive analysis
python3 audioanalysis.py analyze music.mp3 -v

# Activity timeline (great for long files)
python3 audioanalysis.py timeline podcast.mp3 -b 30

# Export to JSON for further processing
python3 audioanalysis.py analyze song.mp3 -o song.json

# Process and pipe to jq
python3 audioanalysis.py analyze song.mp3 -o /dev/stdout 2>/dev/null | jq '.mood'
```

### Platform-Specific Notes

- **FFmpeg Path:** Usually `/usr/bin/ffmpeg` on Linux
- **Librosa:** `pip3 install librosa` for tempo detection
- **Temp Files:** Stored in `/tmp/` and auto-cleaned
- **File Permissions:** Ensure read access to audio files

### Next Steps for Clio
1. Add to ABIOS startup checks
2. Create cron jobs for automated analysis
3. Report Linux-specific issues via Synapse

---

## 🌐 NEXUS QUICK START

**Role:** Multi-Platform Agent  
**Time:** 5 minutes  
**Goal:** Learn cross-platform audio analysis workflows

### Step 1: Platform Detection

```python
import platform
import sys

sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")  # Windows
# sys.path.append("/home/user/AutoProjects")  # Linux

from AudioAnalysis.audioanalysis import AudioAnalyzer, DependencyChecker

print(f"Platform: {platform.system()}")
print(f"Python: {platform.python_version()}")

# Check dependencies
deps = DependencyChecker.check_all()
for name, available in deps.items():
    status = "[OK]" if available else "[X]"
    print(f"  {status} {name}")
```

### Step 2: First Use - Cross-Platform Analysis

```python
from pathlib import Path
from AudioAnalysis.audioanalysis import AudioAnalyzer

def analyze_cross_platform(audio_path: str) -> dict:
    """
    Analyze audio on any platform.
    
    Handles path differences automatically via pathlib.
    """
    path = Path(audio_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    
    analyzer = AudioAnalyzer()
    result = analyzer.analyze(str(path))
    
    return {
        'platform': platform.system(),
        'file': path.name,
        'duration': result.duration,
        'peak_db': result.peak_db,
        'mood': result.mood.primary_mood if result.mood else None
    }

# Works on Windows, Linux, macOS
result = analyze_cross_platform("music.mp3")
print(result)
```

### Step 3: Platform-Specific Considerations

**Windows:**
```python
# Windows paths use backslashes, but pathlib handles this
from pathlib import Path

audio = Path("C:/Users/logan/Music/song.mp3")
# or: audio = Path(r"C:\Users\logan\Music\song.mp3")
```

**Linux:**
```python
from pathlib import Path

audio = Path("/home/logan/music/song.mp3")
# Expand ~ if needed: audio = Path("~/music/song.mp3").expanduser()
```

**macOS:**
```python
from pathlib import Path

audio = Path("/Users/logan/Music/song.mp3")
# or: audio = Path("~/Music/song.mp3").expanduser()
```

**FFmpeg Installation by Platform:**
```python
import platform

def get_ffmpeg_install_command() -> str:
    """Get FFmpeg install command for current platform."""
    system = platform.system()
    
    if system == "Windows":
        return "winget install ffmpeg"
    elif system == "Linux":
        return "sudo apt install ffmpeg"
    elif system == "Darwin":  # macOS
        return "brew install ffmpeg"
    else:
        return "Install FFmpeg from https://ffmpeg.org"
```

### Step 4: Common Nexus Commands

```python
# Cross-platform analysis wrapper
def nexus_analyze(audio_path: str):
    """Nexus cross-platform analysis."""
    from AudioAnalysis.audioanalysis import AudioAnalyzer
    
    analyzer = AudioAnalyzer()
    result = analyzer.analyze(audio_path)
    
    print(f"[{platform.system()}] Analysis Complete:")
    print(f"  Duration: {result.duration}")
    print(f"  Peak: {result.peak_db} dB")
    print(f"  Processing: {result.processing_time_seconds}s")
    
    return result
```

### Next Steps for Nexus
1. Test on all 3 platforms
2. Report platform-specific issues
3. Add to multi-platform workflows

---

## 🆓 BOLT QUICK START

**Role:** Free Executor (Cline + Grok)  
**Time:** 5 minutes  
**Goal:** Learn to use AudioAnalysis for cost-free batch processing

### Step 1: Verify Free Access

```bash
# No API key required! AudioAnalysis is 100% local
cd C:\Users\logan\OneDrive\Documents\AutoProjects\AudioAnalysis

python audioanalysis.py --help
# Should show all available commands - no API setup needed!
```

### Step 2: First Use - Batch Analysis

```bash
# Analyze all MP3 files in a directory
for %f in (*.mp3) do python audioanalysis.py analyze "%f" -t quick

# On PowerShell:
Get-ChildItem *.mp3 | ForEach-Object { python audioanalysis.py analyze $_.Name -t quick }
```

### Step 3: Integration with Bolt Workflows

**Batch Processing Script:**

```python
#!/usr/bin/env python3
"""
Bolt batch audio processor - analyze many files for FREE!
"""

import sys
import json
from pathlib import Path
from dataclasses import asdict

sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")
from AudioAnalysis.audioanalysis import AudioAnalyzer

def batch_analyze(input_dir: str, output_dir: str = None):
    """
    Analyze all audio files in directory.
    
    100% local processing - no API costs!
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "analysis"
    output_path.mkdir(exist_ok=True)
    
    # Supported formats
    formats = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    for fmt in formats:
        audio_files.extend(input_path.glob(fmt))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Output to: {output_path}")
    print("=" * 50)
    
    analyzer = AudioAnalyzer(detect_tempo=False)  # Faster without tempo
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_file.name}...", end=" ")
        
        try:
            result = analyzer.analyze(str(audio_file), analysis_type="quick")
            
            # Save individual result
            output_file = output_path / f"{audio_file.stem}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            results.append({
                'file': audio_file.name,
                'duration': result.duration,
                'peak_db': result.peak_db,
                'mood': result.mood.primary_mood if result.mood else None
            })
            
            print(f"[OK] {result.duration}")
            
        except Exception as e:
            print(f"[FAIL] {e}")
            results.append({
                'file': audio_file.name,
                'error': str(e)
            })
    
    # Save summary
    summary_file = output_path / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 50)
    print(f"Complete! Summary saved to: {summary_file}")
    
    # Print stats
    successful = [r for r in results if 'error' not in r]
    print(f"Success: {len(successful)}/{len(results)}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch audio analysis")
    parser.add_argument("input_dir", help="Directory with audio files")
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()
    
    batch_analyze(args.input_dir, args.output)
```

### Step 4: Common Bolt Commands

```bash
# Quick analysis (fastest)
python audioanalysis.py analyze audio.mp3 -t quick

# Voltage only (super fast for many files)
python audioanalysis.py voltage audio.mp3

# Bulk operations with PowerShell
Get-ChildItem -Recurse -Filter "*.mp3" | ForEach-Object {
    python audioanalysis.py voltage $_.FullName -o "$($_.DirectoryName)\$($_.BaseName)_voltage.json"
}
```

### Cost Considerations

- **Zero API costs:** All processing is local
- **No rate limits:** Process unlimited files
- **No tokens used:** AudioAnalysis doesn't use LLM tokens
- **Batch friendly:** Process thousands of files overnight

### Next Steps for Bolt
1. Add to Cline workflows
2. Use for repetitive analysis tasks
3. Report any issues via Synapse

---

## 👑 LOGAN QUICK START

**Role:** The Architect  
**Time:** 5 minutes  
**Goal:** Quick audio analysis from any context

### From BCH Desktop (Proposed)

```
@audioanalysis analyze music.mp3
@audioanalysis voltage narration.wav
@audioanalysis moments podcast.mp3 --top 5
```

### From Command Line

```bash
# Quick check of any audio file
cd C:\Users\logan\OneDrive\Documents\AutoProjects\AudioAnalysis
python audioanalysis.py analyze "C:\Users\logan\Music\song.mp3" -t quick
```

### From Python REPL

```python
>>> import sys
>>> sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")
>>> from AudioAnalysis.audioanalysis import AudioAnalyzer
>>> analyzer = AudioAnalyzer()
>>> result = analyzer.analyze("music.mp3")
>>> print(f"Mood: {result.mood.primary_mood}, Tempo: {result.tempo.bpm if result.tempo else 'N/A'}")
```

### Your Voltage Gauge Concept in Action

The core innovation you suggested - treating audio amplitude like voltage readings:

```python
# Your insight, realized:
result = analyzer.analyze("audio.mp3")

# "Voltage" readings over time
for reading in result.voltage_readings[:5]:
    print(f"Time {reading.timestamp}: {reading.db_level:+.1f} dB")

# Delta detection finds peaks, drops, and builds
for moment in result.key_moments[:3]:
    print(f"[{moment.timestamp}] {moment.moment_type}: {moment.description}")
```

---

## 📚 ADDITIONAL RESOURCES

**For All Agents:**
- Full Documentation: [README.md](README.md)
- Examples: [EXAMPLES.md](EXAMPLES.md)
- Integration Plan: [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md)
- Cheat Sheet: [CHEAT_SHEET.txt](CHEAT_SHEET.txt)
- Integration Examples: [INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md)

**Support:**
- GitHub Issues: https://github.com/DonkRonk17/AudioAnalysis/issues (pending)
- Synapse: Post in THE_SYNAPSE/active/
- Direct: Message ATLAS

---

**Last Updated:** February 5, 2026  
**Maintained By:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC
