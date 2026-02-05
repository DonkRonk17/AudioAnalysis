# AudioAnalysis - Integration Examples

## 🎯 INTEGRATION PHILOSOPHY

**AudioAnalysis** is designed to work seamlessly with other Team Brain tools. This document provides **copy-paste-ready code examples** for common integration patterns.

**Core Concept:** Logan's "Voltage Gauge" insight - treat audio amplitude like voltage readings from a meter for simple, direct measurement.

---

## 📚 TABLE OF CONTENTS

1. [Pattern 1: AudioAnalysis + VideoAnalysis](#pattern-1-audioanalysis--videoanalysis)
2. [Pattern 2: AudioAnalysis + SynapseLink](#pattern-2-audioanalysis--synapselink)
3. [Pattern 3: AudioAnalysis + AgentHealth](#pattern-3-audioanalysis--agenthealth)
4. [Pattern 4: AudioAnalysis + SessionReplay](#pattern-4-audioanalysis--sessionreplay)
5. [Pattern 5: AudioAnalysis + TaskQueuePro](#pattern-5-audioanalysis--taskqueuepro)
6. [Pattern 6: AudioAnalysis + ContextCompressor](#pattern-6-audioanalysis--contextcompressor)
7. [Pattern 7: AudioAnalysis + LogHunter](#pattern-7-audioanalysis--loghunter)
8. [Pattern 8: AudioAnalysis + EmotionalTextureAnalyzer](#pattern-8-audioanalysis--emotionaltextureanalyzer)
9. [Pattern 9: Multi-Tool Workflow](#pattern-9-multi-tool-workflow)
10. [Pattern 10: Full Team Brain Stack](#pattern-10-full-team-brain-stack)

---

## Pattern 1: AudioAnalysis + VideoAnalysis

**Use Case:** Complete media analysis - both visual and audio tracks

**Why:** Video files contain audio tracks. Analyze both for comprehensive media understanding.

**Code:**

```python
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from VideoAnalysis.videoanalysis import VideoAnalyzer

def analyze_media_complete(video_path: str) -> dict:
    """
    Comprehensive media analysis - video + audio.
    
    Args:
        video_path: Path to video file (MP4, AVI, MKV, etc.)
        
    Returns:
        Combined analysis with video and audio data
    """
    print(f"Analyzing media: {video_path}")
    
    # Video analysis (visual content)
    print("  Analyzing video track...")
    video_analyzer = VideoAnalyzer()
    video_result = video_analyzer.analyze(video_path)
    
    # Audio analysis (audio track - FFmpeg extracts it)
    print("  Analyzing audio track...")
    audio_analyzer = AudioAnalyzer(detect_tempo=True)
    audio_result = audio_analyzer.analyze(video_path)  # FFmpeg handles extraction
    
    # Combine results
    combined = {
        'file': video_result.file_name,
        'duration': audio_result.duration,
        
        # Video analysis
        'video': {
            'frame_count': video_result.frame_count,
            'scene_count': len(video_result.scene_changes),
            'key_frames': [kf.timestamp for kf in video_result.key_frames[:5]],
            'activity_level': video_result.average_activity
        },
        
        # Audio analysis
        'audio': {
            'peak_db': audio_result.peak_db,
            'average_db': audio_result.average_db,
            'dynamic_range': audio_result.dynamic_range_db,
            'mood': audio_result.mood.primary_mood if audio_result.mood else None,
            'energy': audio_result.mood.energy_level if audio_result.mood else None,
            'tempo_bpm': audio_result.tempo.bpm if audio_result.tempo else None
        },
        
        # Combined assessment
        'combined': {
            'audio_video_duration_match': abs(
                video_result.duration_seconds - audio_result.duration_seconds
            ) < 1.0,
            'is_music_video': (
                audio_result.tempo is not None and 
                audio_result.tempo.bpm > 80
            ),
            'is_dialog_heavy': audio_result.average_db > -20
        }
    }
    
    print("  Analysis complete!")
    return combined

# Example usage
result = analyze_media_complete("demo_video.mp4")
print(f"Video scenes: {result['video']['scene_count']}")
print(f"Audio mood: {result['audio']['mood']}")
print(f"Duration match: {result['combined']['audio_video_duration_match']}")
```

**Result:** Comprehensive understanding of both visual and audio content in media files.

---

## Pattern 2: AudioAnalysis + SynapseLink

**Use Case:** Notify Team Brain when audio analysis completes

**Why:** Keep team informed of analysis results automatically

**Code:**

```python
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer, AnalysisResult
from SynapseLink.synapselink import quick_send

def analyze_and_notify(
    audio_path: str,
    notify_to: str = "TEAM",
    priority: str = "NORMAL"
) -> AnalysisResult:
    """
    Analyze audio and notify team of results via Synapse.
    
    Args:
        audio_path: Path to audio file
        notify_to: Recipients (TEAM, FORGE, ATLAS, etc.)
        priority: Message priority (NORMAL, HIGH, CRITICAL)
        
    Returns:
        Analysis result
    """
    # Perform analysis
    analyzer = AudioAnalyzer(detect_tempo=True)
    result = analyzer.analyze(audio_path)
    
    # Build notification message
    mood_info = ""
    if result.mood:
        mood_info = f"""
Mood Analysis:
- Primary Mood: {result.mood.primary_mood}
- Energy Level: {result.mood.energy_level}
- Emotional Arc: {result.mood.emotional_arc}
"""
    
    tempo_info = ""
    if result.tempo:
        tempo_info = f"""
Tempo: {result.tempo.bpm} BPM ({result.tempo.description})
"""
    
    message = f"""Audio Analysis Complete: {result.file_name}

Duration: {result.duration}
Format: {result.format_name} ({result.codec})

Voltage/dB Analysis:
- Peak Level: {result.peak_db} dB
- Average Level: {result.average_db} dB
- Dynamic Range: {result.dynamic_range_db} dB
{mood_info}{tempo_info}
Key Moments Detected: {len(result.key_moments)}

Processing Time: {result.processing_time_seconds}s

Analysis by AudioAnalysis v{result.tool_version}
"""
    
    # Send notification
    quick_send(
        notify_to,
        f"Audio Analysis: {result.file_name}",
        message,
        priority=priority
    )
    
    print(f"[OK] Notification sent to {notify_to}")
    return result

# Example usage
result = analyze_and_notify("important_audio.mp3", notify_to="FORGE", priority="HIGH")
```

**Result:** Team automatically informed of analysis results without manual status updates.

---

## Pattern 3: AudioAnalysis + AgentHealth

**Use Case:** Track audio analysis operations in agent health metrics

**Why:** Monitor agent performance during audio processing tasks

**Code:**

```python
import sys
import time
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer, AudioAnalysisError
from AgentHealth.agenthealth import AgentHealth

def analyze_with_health_tracking(
    audio_path: str,
    agent_name: str = "ATLAS"
) -> dict:
    """
    Analyze audio with full health tracking.
    
    Args:
        audio_path: Path to audio file
        agent_name: Name of agent performing analysis
        
    Returns:
        Dict with result and health metrics
    """
    health = AgentHealth()
    analyzer = AudioAnalyzer()
    
    session_id = f"audio_analysis_{int(time.time())}"
    
    # Start health tracking
    health.start_session(agent_name, session_id=session_id)
    health.heartbeat(agent_name, status="analyzing_audio")
    
    start_time = time.time()
    
    try:
        # Perform analysis
        result = analyzer.analyze(audio_path)
        
        # Log success metrics
        health.log_event(agent_name, "audio_analysis_complete", {
            "file": result.file_name,
            "duration_seconds": result.duration_seconds,
            "processing_time": result.processing_time_seconds,
            "peak_db": result.peak_db,
            "key_moments": len(result.key_moments)
        })
        
        health.heartbeat(agent_name, status="idle")
        
        return {
            'success': True,
            'result': result,
            'metrics': {
                'session_id': session_id,
                'elapsed_time': time.time() - start_time,
                'file_duration': result.duration_seconds,
                'processing_ratio': result.processing_time_seconds / max(result.duration_seconds, 0.1)
            }
        }
        
    except AudioAnalysisError as e:
        health.log_error(agent_name, f"Audio analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'session_id': session_id
        }
        
    finally:
        health.end_session(agent_name, session_id=session_id)

# Example usage
result = analyze_with_health_tracking("podcast.mp3", agent_name="ATLAS")

if result['success']:
    print(f"Analysis complete!")
    print(f"Processing ratio: {result['metrics']['processing_ratio']:.2f}x realtime")
else:
    print(f"Analysis failed: {result['error']}")
```

**Result:** Full visibility into agent performance during audio analysis tasks.

---

## Pattern 4: AudioAnalysis + SessionReplay

**Use Case:** Record audio analysis sessions for debugging and review

**Why:** Replay analysis steps when issues occur or for training

**Code:**

```python
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer, AudioAnalysisError
from SessionReplay.sessionreplay import SessionReplay

def analyze_with_replay(audio_path: str, agent_name: str = "ATLAS") -> dict:
    """
    Analyze audio with full session recording.
    
    Perfect for debugging and reviewing analysis workflows.
    """
    replay = SessionReplay()
    analyzer = AudioAnalyzer(detect_tempo=True)
    
    # Start recording session
    session_id = replay.start_session(
        agent_name, 
        task=f"Audio analysis: {audio_path}"
    )
    
    replay.log_input(session_id, f"Starting analysis of: {audio_path}")
    
    try:
        # Log configuration
        replay.log_input(session_id, "Configuration: detect_tempo=True")
        
        # Perform analysis
        result = analyzer.analyze(audio_path)
        
        # Log key outputs
        replay.log_output(session_id, f"Duration: {result.duration}")
        replay.log_output(session_id, f"Format: {result.format_name} ({result.codec})")
        replay.log_output(session_id, f"Peak: {result.peak_db} dB")
        replay.log_output(session_id, f"Average: {result.average_db} dB")
        replay.log_output(session_id, f"Key moments found: {len(result.key_moments)}")
        
        if result.mood:
            replay.log_output(session_id, f"Mood: {result.mood.primary_mood}")
            replay.log_output(session_id, f"Energy: {result.mood.energy_level}")
        
        if result.tempo:
            replay.log_output(session_id, f"Tempo: {result.tempo.bpm} BPM")
        
        # Log top key moments
        for moment in result.key_moments[:3]:
            replay.log_output(
                session_id, 
                f"[{moment.timestamp}] {moment.moment_type}: {moment.description}"
            )
        
        replay.end_session(session_id, status="COMPLETED")
        
        return {
            'success': True,
            'result': result,
            'session_id': session_id
        }
        
    except AudioAnalysisError as e:
        replay.log_error(session_id, f"Analysis failed: {e}")
        replay.end_session(session_id, status="FAILED")
        
        return {
            'success': False,
            'error': str(e),
            'session_id': session_id
        }

# Example usage
result = analyze_with_replay("music.mp3")

if result['success']:
    print(f"Session recorded: {result['session_id']}")
    print("Review with: sessionreplay show " + result['session_id'])
```

**Result:** Complete replay capability for debugging and reviewing analysis sessions.

---

## Pattern 5: AudioAnalysis + TaskQueuePro

**Use Case:** Queue multiple audio files for batch analysis

**Why:** Manage and track audio analysis tasks in centralized queue

**Code:**

```python
import sys
from pathlib import Path
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from TaskQueuePro.taskqueuepro import TaskQueuePro

def queue_audio_analyses(
    audio_files: list,
    agent: str = "BOLT",
    priority: int = 3
) -> list:
    """
    Queue multiple audio files for analysis.
    
    Args:
        audio_files: List of audio file paths
        agent: Agent to assign tasks to
        priority: Task priority (1=highest, 5=lowest)
        
    Returns:
        List of created task IDs
    """
    queue = TaskQueuePro()
    task_ids = []
    
    for audio_path in audio_files:
        filename = Path(audio_path).name
        
        task_id = queue.create_task(
            title=f"Analyze audio: {filename}",
            agent=agent,
            priority=priority,
            metadata={
                "tool": "AudioAnalysis",
                "file_path": str(audio_path),
                "analysis_type": "comprehensive"
            }
        )
        task_ids.append(task_id)
        print(f"  Queued: {filename} -> {task_id}")
    
    print(f"\n[OK] Queued {len(task_ids)} audio analysis tasks")
    return task_ids


def process_audio_task(task_id: str) -> dict:
    """
    Process a queued audio analysis task.
    
    Args:
        task_id: Task ID from queue
        
    Returns:
        Processing result
    """
    queue = TaskQueuePro()
    analyzer = AudioAnalyzer()
    
    task = queue.get_task(task_id)
    
    if not task:
        return {'error': 'Task not found'}
    
    # Mark as in progress
    queue.start_task(task_id)
    
    try:
        # Get file path from metadata
        file_path = task.metadata.get('file_path')
        analysis_type = task.metadata.get('analysis_type', 'quick')
        
        # Perform analysis
        result = analyzer.analyze(file_path, analysis_type=analysis_type)
        
        # Complete task with results
        queue.complete_task(task_id, result={
            'duration': result.duration,
            'peak_db': result.peak_db,
            'average_db': result.average_db,
            'mood': result.mood.primary_mood if result.mood else None,
            'key_moments': len(result.key_moments)
        })
        
        return {
            'success': True,
            'task_id': task_id,
            'summary': f"Analyzed {result.file_name}: {result.mood.primary_mood if result.mood else 'N/A'}"
        }
        
    except Exception as e:
        queue.fail_task(task_id, error=str(e))
        return {
            'success': False,
            'task_id': task_id,
            'error': str(e)
        }

# Example usage
audio_files = [
    "song1.mp3",
    "song2.mp3", 
    "podcast.mp3",
    "narration.wav"
]

# Queue all files
task_ids = queue_audio_analyses(audio_files, agent="BOLT")

# Process each (Bolt would do this)
for task_id in task_ids:
    result = process_audio_task(task_id)
    print(f"  {result}")
```

**Result:** Centralized tracking of audio analysis tasks across Team Brain.

---

## Pattern 6: AudioAnalysis + ContextCompressor

**Use Case:** Compress large analysis results for efficient sharing

**Why:** Save tokens when sharing comprehensive analysis results

**Code:**

```python
import sys
import json
from dataclasses import asdict
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from ContextCompressor.contextcompressor import ContextCompressor

def analyze_and_compress(audio_path: str, query: str = None) -> dict:
    """
    Analyze audio and compress results for efficient sharing.
    
    Full analysis can be 50,000+ characters. Compression reduces this
    to key findings only.
    
    Args:
        audio_path: Path to audio file
        query: Optional focus query for compression
        
    Returns:
        Dict with compressed results and stats
    """
    analyzer = AudioAnalyzer(detect_tempo=True)
    compressor = ContextCompressor()
    
    # Perform full analysis
    result = analyzer.analyze(audio_path)
    
    # Convert to JSON (full analysis)
    full_json = json.dumps(asdict(result), indent=2)
    
    # Compress for sharing
    compression_query = query or "key findings mood tempo peaks moments summary"
    
    compressed = compressor.compress_text(
        full_json,
        query=compression_query,
        method="summary"
    )
    
    # Calculate savings
    original_chars = len(full_json)
    compressed_chars = len(compressed.compressed_text)
    savings_percent = ((original_chars - compressed_chars) / original_chars) * 100
    
    print(f"Compression Results:")
    print(f"  Original: {original_chars:,} characters")
    print(f"  Compressed: {compressed_chars:,} characters")
    print(f"  Savings: {savings_percent:.1f}%")
    print(f"  Estimated tokens saved: ~{compressed.estimated_token_savings}")
    
    return {
        'original_size': original_chars,
        'compressed_size': compressed_chars,
        'savings_percent': savings_percent,
        'compressed_text': compressed.compressed_text,
        'full_result': result
    }

# Example usage
result = analyze_and_compress("long_podcast.mp3")

# Share the compressed version
print("\nCompressed Analysis for Sharing:")
print(result['compressed_text'][:500] + "...")
```

**Result:** 70-90% size reduction for efficient sharing of analysis results.

---

## Pattern 7: AudioAnalysis + LogHunter

**Use Case:** Debug audio processing issues by searching logs

**Why:** Find and diagnose audio processing failures

**Code:**

```python
import sys
import logging
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from LogHunter.loghunter import LogHunter

# Configure AudioAnalysis to log to file
log_file = "audioanalysis_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def analyze_with_logging(audio_path: str) -> dict:
    """Analyze audio with detailed logging for debugging."""
    analyzer = AudioAnalyzer()
    
    try:
        result = analyzer.analyze(audio_path)
        return {'success': True, 'result': result}
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def hunt_audio_errors(log_file: str = "audioanalysis_debug.log"):
    """Search logs for audio processing errors."""
    hunter = LogHunter()
    
    print("Searching for audio processing issues...\n")
    
    # Find errors
    errors = hunter.search(log_file, pattern="ERROR|FAIL|Exception", last_n=50)
    if errors:
        print(f"[!] Found {len(errors)} errors:")
        for entry in errors[:5]:
            print(f"  [{entry.timestamp}] {entry.message[:80]}...")
    else:
        print("[OK] No errors found")
    
    # Find slow operations
    slow_ops = hunter.search(log_file, pattern="processing_time.*[1-9][0-9]")
    if slow_ops:
        print(f"\n[!] Found {len(slow_ops)} slow operations (>10s)")
    
    # Find FFmpeg issues
    ffmpeg_issues = hunter.search(log_file, pattern="ffmpeg|FFmpeg|ffprobe")
    if ffmpeg_issues:
        print(f"\n[i] {len(ffmpeg_issues)} FFmpeg-related log entries")
    
    return {
        'errors': len(errors),
        'slow_ops': len(slow_ops),
        'ffmpeg_entries': len(ffmpeg_issues)
    }

# Example usage
# First, run some analyses
analyze_with_logging("test1.mp3")
analyze_with_logging("test2.mp3")

# Then hunt for issues
issues = hunt_audio_errors()
print(f"\nDiagnostic summary: {issues}")
```

**Result:** Quick identification of audio processing issues and failures.

---

## Pattern 8: AudioAnalysis + EmotionalTextureAnalyzer

**Use Case:** Combined audio mood + text emotion analysis

**Why:** Get comprehensive emotional understanding of content with both audio and transcript

**Code:**

```python
import sys
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from EmotionalTextureAnalyzer.emotionaltextureanalyzer import EmotionalTextureAnalyzer

def analyze_emotional_content(
    audio_path: str,
    transcript: str = None
) -> dict:
    """
    Combined emotional analysis from audio and text.
    
    Args:
        audio_path: Path to audio file
        transcript: Optional text transcript (if not provided, uses speech detection)
        
    Returns:
        Combined emotional analysis
    """
    # Audio analysis with speech detection if no transcript
    audio_analyzer = AudioAnalyzer(
        detect_tempo=True,
        detect_speech=(transcript is None)
    )
    audio_result = audio_analyzer.analyze(audio_path)
    
    # Get text to analyze
    text_to_analyze = transcript
    if not text_to_analyze and audio_result.speech and audio_result.speech.has_speech:
        text_to_analyze = audio_result.speech.full_transcript
    
    # Text emotional analysis
    text_emotion = None
    if text_to_analyze:
        emotion_analyzer = EmotionalTextureAnalyzer()
        text_emotion = emotion_analyzer.analyze(text_to_analyze)
    
    # Combine results
    result = {
        'audio_analysis': {
            'mood': audio_result.mood.primary_mood if audio_result.mood else None,
            'energy': audio_result.mood.energy_level if audio_result.mood else None,
            'arc': audio_result.mood.emotional_arc if audio_result.mood else None,
            'tempo': audio_result.tempo.bpm if audio_result.tempo else None,
            'confidence': audio_result.mood.confidence if audio_result.mood else 0
        },
        'text_analysis': None,
        'combined': None
    }
    
    if text_emotion:
        result['text_analysis'] = {
            'emotion': text_emotion.primary_emotion,
            'sentiment': text_emotion.sentiment,
            'intensity': text_emotion.intensity,
            'confidence': text_emotion.confidence
        }
        
        # Combined assessment
        audio_mood = result['audio_analysis']['mood'] or "Unknown"
        text_emotion_str = text_emotion.primary_emotion
        
        # Simple mood alignment check
        mood_aligned = _moods_align(audio_mood, text_emotion_str)
        
        result['combined'] = {
            'audio_mood': audio_mood,
            'text_emotion': text_emotion_str,
            'aligned': mood_aligned,
            'combined_confidence': (
                (result['audio_analysis']['confidence'] + text_emotion.confidence) / 2
            ),
            'interpretation': _interpret_combined(audio_mood, text_emotion_str, mood_aligned)
        }
    
    return result


def _moods_align(audio_mood: str, text_emotion: str) -> bool:
    """Check if audio mood and text emotion align."""
    positive_moods = {'Uplifting', 'Energetic', 'Calm', 'Steady'}
    negative_moods = {'Dramatic', 'Tense', 'Sad', 'Angry'}
    
    positive_emotions = {'happy', 'joy', 'excited', 'content', 'hopeful'}
    negative_emotions = {'sad', 'angry', 'fear', 'anxious', 'frustrated'}
    
    audio_positive = audio_mood in positive_moods
    text_positive = text_emotion.lower() in positive_emotions
    
    return audio_positive == text_positive


def _interpret_combined(audio_mood: str, text_emotion: str, aligned: bool) -> str:
    """Generate interpretation of combined analysis."""
    if aligned:
        return f"Audio ({audio_mood}) and text ({text_emotion}) emotions align - content is emotionally consistent"
    else:
        return f"Audio ({audio_mood}) and text ({text_emotion}) emotions differ - possible irony, contrast, or complexity"

# Example usage
result = analyze_emotional_content(
    "motivational_speech.mp3",
    transcript="Today is the day we change everything. No more excuses. Let's do this!"
)

print(f"Audio mood: {result['audio_analysis']['mood']}")
print(f"Text emotion: {result['text_analysis']['emotion'] if result['text_analysis'] else 'N/A'}")
if result['combined']:
    print(f"Aligned: {result['combined']['aligned']}")
    print(f"Interpretation: {result['combined']['interpretation']}")
```

**Result:** Comprehensive emotional understanding combining audio and text analysis.

---

## Pattern 9: Multi-Tool Workflow

**Use Case:** Complete workflow using multiple tools for audio review

**Why:** Demonstrate real production scenario with full instrumentation

**Code:**

```python
import sys
import time
sys.path.append("C:/Users/logan/OneDrive/Documents/AutoProjects")

from AudioAnalysis.audioanalysis import AudioAnalyzer
from TaskQueuePro.taskqueuepro import TaskQueuePro
from SessionReplay.sessionreplay import SessionReplay
from AgentHealth.agenthealth import AgentHealth
from SynapseLink.synapselink import quick_send

def full_audio_workflow(
    audio_path: str,
    agent_name: str = "ATLAS"
) -> dict:
    """
    Complete instrumented audio analysis workflow.
    
    Demonstrates integration of multiple Team Brain tools
    for a production-grade audio review process.
    """
    # Initialize all tools
    analyzer = AudioAnalyzer(detect_tempo=True)
    queue = TaskQueuePro()
    replay = SessionReplay()
    health = AgentHealth()
    
    # Create tracking IDs
    timestamp = int(time.time())
    task_id = queue.create_task(
        title=f"Audio review: {audio_path}",
        agent=agent_name,
        priority=2
    )
    session_id = replay.start_session(agent_name, task=f"Audio review workflow")
    
    # Start health tracking
    health.start_session(agent_name, session_id=session_id)
    health.heartbeat(agent_name, status="audio_review")
    
    # Mark task started
    queue.start_task(task_id)
    replay.log_input(session_id, f"Starting audio review: {audio_path}")
    
    try:
        # === PHASE 1: Analysis ===
        replay.log_input(session_id, "Phase 1: Audio analysis")
        health.heartbeat(agent_name, status="analyzing")
        
        result = analyzer.analyze(audio_path)
        
        replay.log_output(session_id, f"Duration: {result.duration}")
        replay.log_output(session_id, f"Peak: {result.peak_db}dB")
        replay.log_output(session_id, f"Mood: {result.mood.primary_mood if result.mood else 'N/A'}")
        
        # === PHASE 2: Quality Check ===
        replay.log_input(session_id, "Phase 2: Quality assessment")
        health.heartbeat(agent_name, status="quality_check")
        
        quality_issues = []
        if result.peak_db > -1:
            quality_issues.append("Possible clipping")
        if result.average_db < -25:
            quality_issues.append("Audio too quiet")
        if result.dynamic_range_db > 40:
            quality_issues.append("Wide dynamic range")
        
        replay.log_output(session_id, f"Quality issues: {len(quality_issues)}")
        for issue in quality_issues:
            replay.log_output(session_id, f"  - {issue}")
        
        # === PHASE 3: Complete ===
        replay.log_input(session_id, "Phase 3: Completing workflow")
        
        # Complete task
        queue.complete_task(task_id, result={
            'duration': result.duration,
            'mood': result.mood.primary_mood if result.mood else None,
            'quality_issues': quality_issues
        })
        
        # End session tracking
        replay.end_session(session_id, status="COMPLETED")
        health.end_session(agent_name, session_id=session_id, status="success")
        
        # === NOTIFY TEAM ===
        notification = f"""Audio Review Complete: {result.file_name}

Duration: {result.duration}
Mood: {result.mood.primary_mood if result.mood else 'N/A'}
Quality Issues: {len(quality_issues)} found

{"Issues: " + ", ".join(quality_issues) if quality_issues else "No quality issues detected."}

Task ID: {task_id}
Session: {session_id}"""
        
        quick_send("TEAM", f"Audio Review: {result.file_name}", notification)
        
        return {
            'success': True,
            'task_id': task_id,
            'session_id': session_id,
            'result': {
                'duration': result.duration,
                'mood': result.mood.primary_mood if result.mood else None,
                'quality_issues': quality_issues,
                'peak_db': result.peak_db,
                'key_moments': len(result.key_moments)
            }
        }
        
    except Exception as e:
        # Handle failure
        queue.fail_task(task_id, error=str(e))
        replay.log_error(session_id, str(e))
        replay.end_session(session_id, status="FAILED")
        health.log_error(agent_name, str(e))
        health.end_session(agent_name, session_id=session_id, status="failed")
        
        # Alert on failure
        quick_send(
            "FORGE,LOGAN",
            f"Audio Review Failed: {audio_path}",
            f"Error: {e}\nTask: {task_id}",
            priority="HIGH"
        )
        
        return {
            'success': False,
            'task_id': task_id,
            'session_id': session_id,
            'error': str(e)
        }

# Example usage
workflow_result = full_audio_workflow("demo_audio.mp3")

if workflow_result['success']:
    print(f"Workflow complete!")
    print(f"Task: {workflow_result['task_id']}")
    print(f"Session: {workflow_result['session_id']}")
    print(f"Mood: {workflow_result['result']['mood']}")
else:
    print(f"Workflow failed: {workflow_result['error']}")
```

**Result:** Fully instrumented, coordinated audio review workflow.

---

## Pattern 10: Full Team Brain Stack

**Use Case:** Ultimate integration - all audio-related tools working together

**Why:** Production-grade audio processing for Team Brain operations

**Code:**

```python
# Full stack integration is demonstrated in INTEGRATION_PLAN.md
# Key components:
#
# 1. AudioAnalysis - Core audio analysis
# 2. VideoAnalysis - For video files (shares audio track)
# 3. SynapseLink - Team notifications
# 4. AgentHealth - Performance monitoring
# 5. SessionReplay - Debugging
# 6. TaskQueuePro - Task management
# 7. ContextCompressor - Token optimization
# 8. LogHunter - Log analysis
# 9. EmotionalTextureAnalyzer - Enhanced mood analysis

# See INTEGRATION_PLAN.md for complete stack example
```

---

## 📊 RECOMMENDED INTEGRATION PRIORITY

**Week 1 (Essential):**
1. [x] SynapseLink - Team notifications
2. [x] AgentHealth - Health correlation
3. [x] SessionReplay - Debugging

**Week 2 (Productivity):**
4. [ ] TaskQueuePro - Task management
5. [ ] VideoAnalysis - Media correlation
6. [ ] ContextCompressor - Token optimization

**Week 3 (Advanced):**
7. [ ] LogHunter - Debug analysis
8. [ ] EmotionalTextureAnalyzer - Enhanced mood
9. [ ] Full stack integration

---

## 🔧 TROUBLESHOOTING INTEGRATIONS

### Import Errors

```python
# Ensure all tools are in Python path
import sys
from pathlib import Path

# Add AutoProjects to path
sys.path.append(str(Path.home() / "OneDrive/Documents/AutoProjects"))

# Then import
from AudioAnalysis.audioanalysis import AudioAnalyzer
```

### FFmpeg Not Found

```python
from AudioAnalysis.audioanalysis import DependencyChecker

deps = DependencyChecker.check_all()
if not deps['ffmpeg']:
    print("Install FFmpeg:")
    print("  Windows: winget install ffmpeg")
    print("  Linux: sudo apt install ffmpeg")
    print("  macOS: brew install ffmpeg")
```

### Tempo Detection Fails

```python
# Librosa required for tempo
deps = DependencyChecker.check_all()
if not deps['librosa']:
    print("Install librosa: pip install librosa")

# Or disable tempo detection
analyzer = AudioAnalyzer(detect_tempo=False)
```

### Memory Issues with Large Files

```python
# Use lower sample rate for large files
analyzer = AudioAnalyzer(
    samples_per_second=5,  # Default is 10
    detect_tempo=False,    # Tempo uses lots of memory
    detect_speech=False    # Speech uses memory
)

# Or use quick analysis
result = analyzer.analyze("large_file.mp3", analysis_type="quick")
```

---

**Last Updated:** February 5, 2026  
**Maintained By:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC
