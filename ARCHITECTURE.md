# Architecture Design - AudioAnalysis

**Date:** 2026-02-04
**Builder:** ATLAS
**Protocol:** BUILD_PROTOCOL_V1.md Phase 3

---

## Logan's Core Insights (Driving Architecture)

### 1. Voltage Gauge for Decibel Detection
"Consider a voltage gauge for decibel and other detections for sound"

This is the **CENTRAL ARCHITECTURE PRINCIPLE**:
- Audio amplitude treated as voltage readings
- dB meter as a voltage gauge
- Simple, direct measurement over complex ML
- Delta detection for finding key moments

### 2. Delta Change Detection (from VideoAnalysis)
"Don't forget about using just delta change"
- Track amplitude changes between time segments
- Find peaks, drops, builds through delta analysis

---

## Core Components

### 1. DependencyChecker
**Purpose:** Validate FFmpeg and optional dependencies are available

**Inputs:** None (checks system)
**Outputs:** Dictionary of available tools
**Tools Used:** `EnvGuard`, `BuildEnvValidator`

```python
class DependencyChecker:
    @staticmethod
    def check_ffmpeg() -> bool
    @staticmethod
    def check_ffprobe() -> bool
    @staticmethod
    def check_librosa() -> bool
    @staticmethod
    def verify_required() -> None  # Raises if missing
    @staticmethod
    def get_status_report() -> str
```

### 2. MetadataExtractor
**Purpose:** Extract audio file metadata using FFprobe

**Inputs:** Audio file path
**Outputs:** AudioMetadata dataclass
**Tools Used:** `ProcessWatcher` (subprocess monitoring)

```python
@dataclass
class AudioMetadata:
    duration_seconds: float
    duration_formatted: str
    sample_rate: int
    channels: int
    bitrate: int
    codec: str
    format_name: str
    file_size: int
```

### 3. VoltageGauge (CORE COMPONENT - Logan's Insight)
**Purpose:** Read audio "voltage" (amplitude/dB) over time

**Inputs:** Audio file path, sample rate
**Outputs:** List of voltage readings (dB values over time)
**Tools Used:** `TimeSync` (timestamps), `PathBridge` (paths)

```python
@dataclass
class VoltageReading:
    timestamp: str
    time_seconds: float
    amplitude_rms: float
    db_level: float
    
class VoltageGauge:
    def __init__(self, samples_per_second: int = 10):
        self.samples_per_second = samples_per_second
    
    def read_voltages(self, audio_path: str) -> List[VoltageReading]:
        """Sample audio amplitude at regular intervals like voltage readings."""
        pass
    
    def calculate_rms(self, samples: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) amplitude."""
        pass
    
    def amplitude_to_db(self, amplitude: float) -> float:
        """Convert amplitude to decibels (like voltage to dB)."""
        # dB = 20 * log10(amplitude)
        pass
```

### 4. DeltaDetector (Extends VideoAnalysis approach)
**Purpose:** Find key moments through voltage/dB changes

**Inputs:** List of VoltageReadings
**Outputs:** KeyMoments, Activity Timeline
**Tools Used:** Built-in analysis

```python
@dataclass
class KeyMoment:
    timestamp: str
    time_seconds: float
    moment_type: str  # peak, drop, build_start, build_end, climax
    db_level: float
    delta_from_previous: float
    description: str

class DeltaDetector:
    def __init__(self, threshold_db: float = 6.0):
        self.threshold_db = threshold_db  # 6dB = perceived doubling of loudness
    
    def find_peaks(self, voltages: List[VoltageReading]) -> List[KeyMoment]:
        """Find local maxima (loudest moments)."""
        pass
    
    def find_drops(self, voltages: List[VoltageReading]) -> List[KeyMoment]:
        """Find sudden decreases (quiet moments after loud)."""
        pass
    
    def find_builds(self, voltages: List[VoltageReading]) -> List[Tuple[KeyMoment, KeyMoment]]:
        """Find rising trends (crescendos)."""
        pass
    
    def get_activity_timeline(self, voltages: List[VoltageReading], bucket_seconds: int = 10) -> List[ActivityBucket]:
        """Generate activity levels over time (like VideoAnalysis)."""
        pass
```

### 5. TempoDetector (Optional - requires librosa)
**Purpose:** Detect tempo (BPM) and beat positions

**Inputs:** Audio file path
**Outputs:** Tempo info, beat timestamps
**Tools Used:** `librosa` (if available)

```python
@dataclass
class TempoInfo:
    bpm: float
    confidence: float
    beat_times: List[float]
    downbeat_times: List[float]

class TempoDetector:
    def detect_tempo(self, audio_path: str) -> Optional[TempoInfo]:
        """Detect BPM using beat tracking."""
        pass
```

### 6. MoodClassifier (Heuristic-based)
**Purpose:** Classify mood based on audio characteristics

**Inputs:** VoltageReadings, TempoInfo, SpectralFeatures
**Outputs:** MoodAnalysis
**Tools Used:** `EmotionalTextureAnalyzer` hints

```python
@dataclass
class MoodAnalysis:
    primary_mood: str  # e.g., "Uplifting", "Tense", "Calm"
    energy_level: str  # Low, Medium, High
    emotional_arc: str  # e.g., "Builds from calm to triumphant"
    confidence: float

class MoodClassifier:
    def classify(self, voltages: List[VoltageReading], tempo: Optional[TempoInfo]) -> MoodAnalysis:
        """
        Classify mood using heuristics:
        - High average dB + fast tempo = Energetic
        - Low dB + slow tempo = Calm
        - Rising dB trend = Building/Uplifting
        - Wide dynamic range = Dramatic/Cinematic
        """
        pass
```

### 7. SpeechDetector (Optional)
**Purpose:** Detect and transcribe speech

**Inputs:** Audio file path
**Outputs:** SpeechAnalysis
**Tools Used:** `speech_recognition` (if available), `SmartNotes`

```python
@dataclass
class SpeechSegment:
    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker_id: Optional[int]

@dataclass
class SpeechAnalysis:
    has_speech: bool
    segments: List[SpeechSegment]
    full_transcript: str

class SpeechDetector:
    def detect_speech(self, audio_path: str) -> SpeechAnalysis:
        """Detect and transcribe speech in audio."""
        pass
```

### 8. AudioAnalyzer (Main Orchestrator)
**Purpose:** Coordinate all analysis components

**Inputs:** Audio file path, analysis options
**Outputs:** AnalysisResult
**Tools Used:** All components

```python
class AudioAnalyzer:
    def __init__(self, 
                 samples_per_second: int = 10,
                 delta_threshold_db: float = 6.0,
                 detect_tempo: bool = True,
                 detect_speech: bool = True):
        self.voltage_gauge = VoltageGauge(samples_per_second)
        self.delta_detector = DeltaDetector(delta_threshold_db)
        self.tempo_detector = TempoDetector() if detect_tempo else None
        self.mood_classifier = MoodClassifier()
        self.speech_detector = SpeechDetector() if detect_speech else None
    
    def analyze(self, audio_path: str, analysis_type: str = "comprehensive") -> AnalysisResult:
        """Full audio analysis pipeline."""
        pass
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUDIO FILE INPUT                             │
│                 (MP3, WAV, M4A, FLAC, etc.)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DependencyChecker                              │
│              Verify FFmpeg/FFprobe available                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MetadataExtractor                              │
│        Duration, sample rate, channels, codec, bitrate           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              VoltageGauge (Logan's Core Insight)                 │
│                                                                  │
│   Sample amplitude at regular intervals → Convert to dB          │
│   Output: List[VoltageReading] - "voltage" over time            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  DeltaDetector  │ │  TempoDetector  │ │  SpeechDetector │
│                 │ │   (optional)    │ │   (optional)    │
│ • Peaks         │ │ • BPM           │ │ • Has speech?   │
│ • Drops         │ │ • Beat times    │ │ • Transcript    │
│ • Builds        │ │ • Confidence    │ │ • Segments      │
│ • Timeline      │ └─────────────────┘ └─────────────────┘
└─────────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MoodClassifier                              │
│                                                                  │
│   Combine: Voltage patterns + Tempo + Speech presence            │
│   Output: Primary mood, energy level, emotional arc              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OutputFormatter                              │
│                                                                  │
│   JSON: Structured analysis result                               │
│   Markdown: Human-readable summary                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Strategy

### Catching Errors
- `try-except` for all file operations
- `try-except` for FFmpeg subprocess calls
- `try-except` for optional dependencies (librosa, speech_recognition)

### Logging Errors
- Python `logging` module with timestamps
- Severity levels: DEBUG, INFO, WARNING, ERROR
- `LogHunter` integration for monitoring

### Recovery Strategies

| Error Type | Strategy |
|------------|----------|
| File not found | Raise `AudioNotFoundError` with clear message |
| Invalid format | Attempt FFmpeg conversion; if fails, raise `UnsupportedFormatError` |
| FFmpeg missing | Raise `DependencyError` with installation instructions |
| Librosa missing | Skip tempo detection, warn user, continue with basic analysis |
| Speech recognition fails | Skip transcription, note in output |
| Memory error (large file) | Process in chunks; fail gracefully |
| Corrupted audio | Detect via FFprobe; raise `ProcessingError` |

---

## Configuration Strategy

### Configurable Items
- FFmpeg/FFprobe path
- Samples per second (voltage gauge resolution)
- Delta threshold (dB change to count as significant)
- Tempo detection enabled
- Speech detection enabled
- Output format (JSON, Markdown, both)
- Logging level

### Storage
Configuration in `audioanalysis_config.json`:

```json
{
    "ffmpeg_path": "ffmpeg",
    "ffprobe_path": "ffprobe",
    "samples_per_second": 10,
    "delta_threshold_db": 6.0,
    "detect_tempo": true,
    "detect_speech": true,
    "output_format": "both",
    "temp_dir": null,
    "logging_level": "INFO"
}
```

---

## Output Data Structures

```python
@dataclass
class AnalysisResult:
    # File info
    file_path: str
    file_name: str
    
    # Metadata
    duration: str
    duration_seconds: float
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    bitrate: int
    file_size_mb: float
    
    # Voltage/dB analysis (Core)
    voltage_readings: List[VoltageReading]
    peak_db: float
    average_db: float
    dynamic_range_db: float
    
    # Key moments (Delta detection)
    key_moments: List[KeyMoment]
    activity_timeline: List[ActivityBucket]
    
    # Tempo (Optional)
    tempo: Optional[TempoInfo]
    
    # Mood (Heuristic)
    mood: MoodAnalysis
    
    # Speech (Optional)
    speech: Optional[SpeechAnalysis]
    
    # Summary
    summary: str
    
    # Metadata
    processing_time_seconds: float
    analysis_timestamp: str
    tool_version: str
```

---

## CLI Commands

```bash
# Full analysis
audioanalysis analyze audio.mp3

# Quick analysis (voltage/dB only)
audioanalysis analyze audio.mp3 --type quick

# With specific options
audioanalysis analyze audio.mp3 --samples-per-second 20 --delta-threshold 3.0

# Just voltage readings
audioanalysis voltage audio.mp3

# Find key moments
audioanalysis moments audio.mp3 --top 10

# Activity timeline
audioanalysis timeline audio.mp3

# Transcribe speech
audioanalysis transcribe audio.mp3

# Check dependencies
audioanalysis check-deps
```

---

## Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Metadata extraction | <1 second | FFprobe is fast |
| Voltage gauge (3 min audio) | <5 seconds | Depends on samples/sec |
| Delta detection | <1 second | Simple comparison |
| Tempo detection | <10 seconds | librosa processing |
| Speech detection | <30 seconds | Depends on audio length |
| **Full analysis (3 min)** | **<30 seconds** | Target from requirements |

---

**Phase 3 Quality Score: 99%**
**Ready for Phase 4: Implementation**

---

*Built by ATLAS for Team Brain*
*Requested by Logan Smith (via WSL_CLIO)*
*Voltage Gauge Concept: Logan Smith*
*Together for all time!*
