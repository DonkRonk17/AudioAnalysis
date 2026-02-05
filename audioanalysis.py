#!/usr/bin/env python3
"""
AudioAnalysis - Enable AI Agents to "Listen" to Audio Content

A comprehensive audio analysis tool that extracts metadata, measures amplitude
(voltage gauge), detects tempo, classifies mood, and finds key moments.

KEY INNOVATION:
Voltage Gauge for Decibel Detection - Logan Smith's insight: "Consider a 
voltage gauge for decibel and other detections for sound" - treating audio
amplitude like voltage readings for simple, direct sound analysis.

Delta Change Detection - Applied from VideoAnalysis: track amplitude changes
between time segments to find peaks, drops, and builds.

TOOLS USED IN THIS BUILD:
- PathBridge: Cross-platform path handling
- ConfigManager: Configuration management
- ErrorRecovery: Graceful error handling
- ProcessWatcher: FFmpeg process monitoring
- TimeSync: Accurate timestamps
- EnvGuard: Dependency verification
- EmotionalTextureAnalyzer: Mood analysis hints

Requested by: Logan Smith (via WSL_CLIO)
Voltage Gauge Concept: Logan Smith
Built by: ATLAS for Team Brain
Protocol: BUILD_PROTOCOL_V1.md + Bug Hunt Protocol

For the Maximum Benefit of Life.
One World. One Family. One Love.
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AudioAnalysis')

# Version info
__version__ = '1.0.0'
__author__ = 'ATLAS (Team Brain)'
__description__ = 'Enable AI agents to listen and analyze audio content'


# ============================================================================
# EXCEPTIONS
# ============================================================================

class AudioAnalysisError(Exception):
    """Base exception for AudioAnalysis errors."""
    pass


class DependencyError(AudioAnalysisError):
    """Raised when required dependencies are missing."""
    pass


class AudioNotFoundError(AudioAnalysisError):
    """Raised when audio file is not found."""
    pass


class ProcessingError(AudioAnalysisError):
    """Raised when audio processing fails."""
    pass


class UnsupportedFormatError(AudioAnalysisError):
    """Raised when audio format is not supported."""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AudioMetadata:
    """Audio file metadata."""
    duration_seconds: float = 0.0
    duration_formatted: str = "00:00:00"
    sample_rate: int = 0
    channels: int = 0
    bitrate: int = 0
    codec: str = ""
    format_name: str = ""
    file_size: int = 0


@dataclass
class VoltageReading:
    """
    A single 'voltage' reading from the audio.
    
    Logan's insight: Treat audio amplitude like voltage readings
    from a meter - simple, direct measurement.
    """
    timestamp: str
    time_seconds: float
    amplitude_rms: float  # Root Mean Square amplitude (0.0 to 1.0)
    db_level: float  # Decibel level (negative, 0 is max)


@dataclass
class KeyMoment:
    """A key moment detected through delta/voltage analysis."""
    timestamp: str
    time_seconds: float
    moment_type: str  # peak, drop, build_start, build_end, climax
    db_level: float
    delta_from_previous: float
    description: str = ""


@dataclass
class ActivityBucket:
    """Activity level for a time segment."""
    start_time: str
    end_time: str
    avg_db: float
    max_db: float
    activity_level: str  # high, medium, low, silent
    bucket_index: int = 0


@dataclass
class TempoInfo:
    """Tempo/rhythm information."""
    bpm: float = 0.0
    confidence: float = 0.0
    beat_count: int = 0
    description: str = ""


@dataclass
class MoodAnalysis:
    """Mood classification based on audio characteristics."""
    primary_mood: str = "Unknown"
    energy_level: str = "Medium"  # Low, Medium, High
    emotional_arc: str = ""
    confidence: float = 0.0
    characteristics: List[str] = field(default_factory=list)


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start_time: float
    end_time: float
    text: str = ""
    confidence: float = 0.0


@dataclass
class SpeechAnalysis:
    """Speech detection and transcription results."""
    has_speech: bool = False
    speech_ratio: float = 0.0  # Percentage of audio that is speech
    segments: List[SpeechSegment] = field(default_factory=list)
    full_transcript: str = ""


@dataclass
class AnalysisResult:
    """Complete audio analysis result."""
    # File info
    file_path: str
    file_name: str
    
    # Metadata
    duration: str = "00:00:00"
    duration_seconds: float = 0.0
    format_name: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    bitrate: int = 0
    file_size_mb: float = 0.0
    
    # Voltage/dB analysis (Core - Logan's insight)
    voltage_readings: List[VoltageReading] = field(default_factory=list)
    peak_db: float = -100.0
    average_db: float = -100.0
    dynamic_range_db: float = 0.0
    
    # Key moments (Delta detection)
    key_moments: List[KeyMoment] = field(default_factory=list)
    activity_timeline: List[ActivityBucket] = field(default_factory=list)
    
    # Tempo (Optional)
    tempo: Optional[TempoInfo] = None
    
    # Mood (Heuristic)
    mood: Optional[MoodAnalysis] = None
    
    # Speech (Optional)
    speech: Optional[SpeechAnalysis] = None
    
    # Summary
    summary: str = ""
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    analysis_timestamp: str = ""
    analysis_type: str = "comprehensive"
    tool_version: str = __version__


# ============================================================================
# DEPENDENCY CHECKER
# ============================================================================

class DependencyChecker:
    """Check for required and optional dependencies."""
    
    REQUIRED = {
        'ffmpeg': {
            'command': ['ffmpeg', '-version'],
            'description': 'Audio processing and conversion',
            'install': {
                'Windows': 'winget install ffmpeg',
                'Linux': 'sudo apt install ffmpeg',
                'macOS': 'brew install ffmpeg'
            }
        },
        'ffprobe': {
            'command': ['ffprobe', '-version'],
            'description': 'Audio metadata extraction',
            'install': {
                'Windows': 'Included with ffmpeg',
                'Linux': 'Included with ffmpeg',
                'macOS': 'Included with ffmpeg'
            }
        }
    }
    
    OPTIONAL = {
        'librosa': {
            'check': lambda: DependencyChecker._check_python_module('librosa'),
            'description': 'Advanced audio analysis (tempo, spectral)',
            'install': 'pip install librosa'
        },
        'speech_recognition': {
            'check': lambda: DependencyChecker._check_python_module('speech_recognition'),
            'description': 'Speech transcription',
            'install': 'pip install SpeechRecognition'
        },
        'numpy': {
            'check': lambda: DependencyChecker._check_python_module('numpy'),
            'description': 'Numerical processing',
            'install': 'pip install numpy'
        }
    }
    
    @staticmethod
    def _check_command(command: List[str]) -> bool:
        """Check if a command is available."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    @staticmethod
    def _check_python_module(module_name: str) -> bool:
        """Check if a Python module is available."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    @classmethod
    def check_all(cls, include_optional: bool = True) -> Dict[str, bool]:
        """Check all dependencies."""
        results = {}
        
        # Check required
        for name, info in cls.REQUIRED.items():
            results[name] = cls._check_command(info['command'])
        
        # Check optional
        if include_optional:
            for name, info in cls.OPTIONAL.items():
                results[name] = info['check']()
        
        return results
    
    @classmethod
    def verify_required(cls) -> None:
        """Verify required dependencies, raise if missing."""
        results = cls.check_all(include_optional=False)
        missing = [name for name, available in results.items() if not available]
        
        if missing:
            msg_parts = ["Missing required dependencies:\n"]
            for name in missing:
                info = cls.REQUIRED[name]
                msg_parts.append(f"\n{name}: {info['description']}")
                for platform, cmd in info['install'].items():
                    msg_parts.append(f"\n  {platform}: {cmd}")
            raise DependencyError(''.join(msg_parts))
    
    @classmethod
    def get_status_report(cls) -> str:
        """Generate a status report of all dependencies."""
        lines = ["AudioAnalysis Dependency Status", "=" * 40, "", "Required:"]
        
        results = cls.check_all()
        
        for name, info in cls.REQUIRED.items():
            status = "[OK]" if results.get(name) else "[MISSING]"
            lines.append(f"  {status} {name}: {info['description']}")
        
        lines.append("")
        lines.append("Optional:")
        
        for name, info in cls.OPTIONAL.items():
            status = "[OK]" if results.get(name) else "[MISSING]"
            lines.append(f"  {status} {name}: {info['description']}")
        
        return '\n'.join(lines)


# ============================================================================
# METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """Extract metadata from audio files using FFprobe."""
    
    @staticmethod
    def extract(audio_path: str) -> AudioMetadata:
        """Extract metadata from audio file."""
        path = Path(audio_path)
        
        if not path.exists():
            raise AudioNotFoundError(f"Audio not found: {audio_path}")
        
        try:
            # Run ffprobe
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    str(path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise ProcessingError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract format info
            fmt = data.get('format', {})
            
            # Find audio stream
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            # Build metadata
            duration = float(fmt.get('duration', 0))
            
            return AudioMetadata(
                duration_seconds=duration,
                duration_formatted=MetadataExtractor._format_duration(duration),
                sample_rate=int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                channels=int(audio_stream.get('channels', 0)) if audio_stream else 0,
                bitrate=int(fmt.get('bit_rate', 0)),
                codec=audio_stream.get('codec_name', '') if audio_stream else '',
                format_name=fmt.get('format_name', ''),
                file_size=int(fmt.get('size', 0))
            )
            
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Could not parse FFprobe output: {e}")
        except subprocess.TimeoutExpired:
            raise ProcessingError("FFprobe timed out")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# VOLTAGE GAUGE (Logan's Core Insight)
# ============================================================================

class VoltageGauge:
    """
    Read audio 'voltage' (amplitude/dB) over time.
    
    Logan's insight: "Consider a voltage gauge for decibel and other 
    detections for sound" - treating audio amplitude like voltage readings
    from a meter for simple, direct measurement.
    
    This is the CORE component of AudioAnalysis, analogous to
    DeltaChangeDetector in VideoAnalysis.
    """
    
    def __init__(self, samples_per_second: int = 10):
        """
        Initialize voltage gauge.
        
        Args:
            samples_per_second: How many voltage readings per second
        """
        self.samples_per_second = samples_per_second
        self._numpy_available = self._check_numpy()
    
    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy
            return True
        except ImportError:
            return False
    
    def read_voltages(self, audio_path: str) -> List[VoltageReading]:
        """
        Sample audio amplitude at regular intervals like voltage readings.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of VoltageReading objects
        """
        # Convert to WAV for consistent processing
        wav_path = self._convert_to_wav(audio_path)
        
        try:
            readings = self._read_wav_voltages(wav_path)
            return readings
        finally:
            # Cleanup temp file
            if wav_path != audio_path and Path(wav_path).exists():
                try:
                    os.remove(wav_path)
                except:
                    pass
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio to WAV format for analysis."""
        path = Path(audio_path)
        
        if path.suffix.lower() == '.wav':
            return audio_path
        
        # Create temp WAV file
        temp_wav = tempfile.mktemp(suffix='.wav')
        
        try:
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-i', str(path),
                    '-ar', '44100',  # Standard sample rate
                    '-ac', '1',      # Mono for simpler analysis
                    '-acodec', 'pcm_s16le',
                    temp_wav
                ],
                capture_output=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise ProcessingError(f"FFmpeg conversion failed: {result.stderr.decode()}")
            
            return temp_wav
            
        except subprocess.TimeoutExpired:
            raise ProcessingError("Audio conversion timed out")
    
    def _read_wav_voltages(self, wav_path: str) -> List[VoltageReading]:
        """Read voltage levels from WAV file."""
        readings = []
        
        try:
            with wave.open(wav_path, 'rb') as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                n_frames = wav.getnframes()
                
                duration = n_frames / frame_rate
                
                # Calculate frames per sample
                frames_per_sample = int(frame_rate / self.samples_per_second)
                
                if frames_per_sample < 1:
                    frames_per_sample = 1
                
                current_time = 0.0
                sample_index = 0
                
                while current_time < duration:
                    # Seek to position
                    frame_pos = int(current_time * frame_rate)
                    if frame_pos >= n_frames:
                        break
                    
                    wav.setpos(frame_pos)
                    
                    # Read chunk
                    frames_to_read = min(frames_per_sample, n_frames - frame_pos)
                    raw_data = wav.readframes(frames_to_read)
                    
                    if not raw_data:
                        break
                    
                    # Calculate RMS amplitude
                    rms = self._calculate_rms(raw_data, sample_width, n_channels)
                    db = self._amplitude_to_db(rms)
                    
                    readings.append(VoltageReading(
                        timestamp=self._format_timestamp(current_time),
                        time_seconds=round(current_time, 2),
                        amplitude_rms=round(rms, 6),
                        db_level=round(db, 2)
                    ))
                    
                    current_time += 1.0 / self.samples_per_second
                    sample_index += 1
        
        except wave.Error as e:
            raise ProcessingError(f"Could not read WAV file: {e}")
        
        return readings
    
    def _calculate_rms(self, raw_data: bytes, sample_width: int, n_channels: int) -> float:
        """
        Calculate Root Mean Square (RMS) amplitude.
        RMS represents the "voltage" or power of the signal.
        """
        if not raw_data:
            return 0.0
        
        # Determine format based on sample width
        if sample_width == 1:
            fmt = 'b'  # signed char
            max_val = 128.0
        elif sample_width == 2:
            fmt = 'h'  # short
            max_val = 32768.0
        elif sample_width == 4:
            fmt = 'i'  # int
            max_val = 2147483648.0
        else:
            return 0.0
        
        # Unpack samples
        n_samples = len(raw_data) // sample_width
        if n_samples == 0:
            return 0.0
        
        try:
            samples = struct.unpack(f'<{n_samples}{fmt}', raw_data)
        except struct.error:
            return 0.0
        
        # Calculate RMS
        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / n_samples) / max_val
        
        return min(rms, 1.0)  # Clamp to 0-1
    
    def _amplitude_to_db(self, amplitude: float) -> float:
        """
        Convert amplitude to decibels.
        
        dB = 20 * log10(amplitude)
        0 dB = maximum (amplitude = 1.0)
        -∞ dB = silence (amplitude = 0)
        """
        if amplitude <= 0:
            return -100.0  # Practical minimum
        
        db = 20 * math.log10(amplitude)
        return max(db, -100.0)  # Clamp to reasonable minimum
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_statistics(self, readings: List[VoltageReading]) -> Dict[str, float]:
        """Get statistics from voltage readings."""
        if not readings:
            return {
                'peak_db': -100.0,
                'average_db': -100.0,
                'min_db': -100.0,
                'dynamic_range_db': 0.0
            }
        
        db_values = [r.db_level for r in readings]
        
        peak = max(db_values)
        avg = sum(db_values) / len(db_values)
        min_db = min(v for v in db_values if v > -100)  # Exclude silence
        
        return {
            'peak_db': round(peak, 2),
            'average_db': round(avg, 2),
            'min_db': round(min_db, 2),
            'dynamic_range_db': round(peak - min_db, 2) if min_db > -100 else 0.0
        }


# ============================================================================
# DELTA DETECTOR (Key Moment Detection)
# ============================================================================

class DeltaDetector:
    """
    Find key moments through voltage/dB changes.
    
    Applies the delta detection philosophy from VideoAnalysis:
    Track changes between consecutive samples to find significant moments.
    """
    
    def __init__(self, threshold_db: float = 6.0):
        """
        Initialize delta detector.
        
        Args:
            threshold_db: Minimum dB change to consider significant
                         (6 dB is roughly perceived as doubling of loudness)
        """
        self.threshold_db = threshold_db
    
    def find_peaks(self, readings: List[VoltageReading], top_n: int = 10) -> List[KeyMoment]:
        """Find the loudest moments (local maxima)."""
        if len(readings) < 3:
            return []
        
        peaks = []
        
        for i in range(1, len(readings) - 1):
            prev_db = readings[i - 1].db_level
            curr_db = readings[i].db_level
            next_db = readings[i + 1].db_level
            
            # Local maximum
            if curr_db > prev_db and curr_db > next_db:
                delta = curr_db - prev_db
                
                if delta >= self.threshold_db / 2:  # Lower threshold for peaks
                    peaks.append(KeyMoment(
                        timestamp=readings[i].timestamp,
                        time_seconds=readings[i].time_seconds,
                        moment_type='peak',
                        db_level=curr_db,
                        delta_from_previous=round(delta, 2),
                        description=f"Peak at {curr_db:.1f} dB"
                    ))
        
        # Sort by dB level and return top N
        peaks.sort(key=lambda x: x.db_level, reverse=True)
        return peaks[:top_n]
    
    def find_drops(self, readings: List[VoltageReading]) -> List[KeyMoment]:
        """Find sudden drops in volume."""
        drops = []
        
        for i in range(1, len(readings)):
            prev_db = readings[i - 1].db_level
            curr_db = readings[i].db_level
            delta = prev_db - curr_db  # Positive = drop
            
            if delta >= self.threshold_db:
                drops.append(KeyMoment(
                    timestamp=readings[i].timestamp,
                    time_seconds=readings[i].time_seconds,
                    moment_type='drop',
                    db_level=curr_db,
                    delta_from_previous=round(-delta, 2),
                    description=f"Drop of {delta:.1f} dB"
                ))
        
        return drops
    
    def find_builds(self, readings: List[VoltageReading], min_duration_sec: float = 3.0) -> List[KeyMoment]:
        """Find building sections (rising volume trend)."""
        if len(readings) < 5:
            return []
        
        builds = []
        sample_interval = readings[1].time_seconds - readings[0].time_seconds if len(readings) > 1 else 0.1
        min_samples = max(3, int(min_duration_sec / sample_interval))
        
        i = 0
        while i < len(readings) - min_samples:
            # Check for rising trend
            start_db = readings[i].db_level
            rising_count = 0
            
            for j in range(i + 1, min(i + min_samples * 2, len(readings))):
                if readings[j].db_level > readings[j - 1].db_level:
                    rising_count += 1
                else:
                    break
            
            if rising_count >= min_samples - 1:
                end_idx = i + rising_count
                total_rise = readings[end_idx].db_level - start_db
                
                if total_rise >= self.threshold_db:
                    builds.append(KeyMoment(
                        timestamp=readings[i].timestamp,
                        time_seconds=readings[i].time_seconds,
                        moment_type='build_start',
                        db_level=start_db,
                        delta_from_previous=round(total_rise, 2),
                        description=f"Build starts, rises {total_rise:.1f} dB over {rising_count * sample_interval:.1f}s"
                    ))
                    i = end_idx
            
            i += 1
        
        return builds
    
    def find_climaxes(self, readings: List[VoltageReading], peaks: List[KeyMoment]) -> List[KeyMoment]:
        """Find climax moments (highest peaks with context)."""
        if not peaks:
            return []
        
        # The loudest peak is likely the climax
        climaxes = []
        
        if peaks:
            loudest = peaks[0]  # Already sorted by dB
            climaxes.append(KeyMoment(
                timestamp=loudest.timestamp,
                time_seconds=loudest.time_seconds,
                moment_type='climax',
                db_level=loudest.db_level,
                delta_from_previous=loudest.delta_from_previous,
                description=f"Main climax at {loudest.db_level:.1f} dB"
            ))
        
        return climaxes
    
    def get_activity_timeline(
        self,
        readings: List[VoltageReading],
        bucket_seconds: int = 10
    ) -> List[ActivityBucket]:
        """Generate activity levels over time (like VideoAnalysis)."""
        if not readings:
            return []
        
        timeline = []
        sample_interval = readings[1].time_seconds - readings[0].time_seconds if len(readings) > 1 else 0.1
        samples_per_bucket = max(1, int(bucket_seconds / sample_interval))
        
        for i in range(0, len(readings), samples_per_bucket):
            bucket_readings = readings[i:i + samples_per_bucket]
            
            if not bucket_readings:
                continue
            
            db_values = [r.db_level for r in bucket_readings]
            avg_db = sum(db_values) / len(db_values)
            max_db = max(db_values)
            
            # Classify activity level
            if avg_db > -10:
                activity = 'high'
            elif avg_db > -20:
                activity = 'medium'
            elif avg_db > -40:
                activity = 'low'
            else:
                activity = 'silent'
            
            timeline.append(ActivityBucket(
                start_time=bucket_readings[0].timestamp,
                end_time=bucket_readings[-1].timestamp,
                avg_db=round(avg_db, 2),
                max_db=round(max_db, 2),
                activity_level=activity,
                bucket_index=len(timeline)
            ))
        
        return timeline
    
    def get_all_key_moments(
        self,
        readings: List[VoltageReading],
        top_peaks: int = 5
    ) -> List[KeyMoment]:
        """Get all key moments sorted by time."""
        peaks = self.find_peaks(readings, top_n=top_peaks)
        drops = self.find_drops(readings)
        builds = self.find_builds(readings)
        climaxes = self.find_climaxes(readings, peaks)
        
        all_moments = peaks + drops + builds + climaxes
        all_moments.sort(key=lambda x: x.time_seconds)
        
        return all_moments


# ============================================================================
# TEMPO DETECTOR (Optional - Uses librosa if available)
# ============================================================================

class TempoDetector:
    """Detect tempo (BPM) from audio."""
    
    def __init__(self):
        self._librosa_available = self._check_librosa()
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            return True
        except ImportError:
            return False
    
    def detect(self, audio_path: str) -> Optional[TempoInfo]:
        """Detect tempo from audio file."""
        if not self._librosa_available:
            logger.warning("Librosa not available - skipping tempo detection")
            return None
        
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Detect tempo
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Handle different librosa versions
            if hasattr(tempo, '__iter__'):
                tempo = float(tempo[0])
            else:
                tempo = float(tempo)
            
            return TempoInfo(
                bpm=round(tempo, 1),
                confidence=0.8,  # librosa doesn't provide confidence
                beat_count=len(beat_frames),
                description=self._describe_tempo(tempo)
            )
            
        except Exception as e:
            logger.warning(f"Tempo detection failed: {e}")
            return None
    
    def _describe_tempo(self, bpm: float) -> str:
        """Describe tempo in human terms."""
        if bpm < 60:
            return "Very slow (largo)"
        elif bpm < 80:
            return "Slow (adagio)"
        elif bpm < 100:
            return "Moderate (andante)"
        elif bpm < 120:
            return "Walking pace (moderato)"
        elif bpm < 140:
            return "Fast (allegro)"
        elif bpm < 180:
            return "Very fast (vivace)"
        else:
            return "Extremely fast (presto)"


# ============================================================================
# MOOD CLASSIFIER (Heuristic-based)
# ============================================================================

class MoodClassifier:
    """
    Classify mood based on audio characteristics.
    Uses heuristics rather than ML for simplicity (Delta philosophy).
    """
    
    def classify(
        self,
        readings: List[VoltageReading],
        tempo: Optional[TempoInfo],
        key_moments: List[KeyMoment]
    ) -> MoodAnalysis:
        """Classify mood from audio characteristics."""
        if not readings:
            return MoodAnalysis()
        
        # Calculate basic statistics
        db_values = [r.db_level for r in readings if r.db_level > -100]
        if not db_values:
            return MoodAnalysis()
        
        avg_db = sum(db_values) / len(db_values)
        dynamic_range = max(db_values) - min(db_values)
        
        # Analyze trend (rising = building, falling = fading)
        first_quarter = db_values[:len(db_values)//4] if len(db_values) > 4 else db_values
        last_quarter = db_values[-(len(db_values)//4):] if len(db_values) > 4 else db_values
        trend = (sum(last_quarter)/len(last_quarter)) - (sum(first_quarter)/len(first_quarter))
        
        # Count builds and peaks
        builds = [m for m in key_moments if m.moment_type == 'build_start']
        peaks = [m for m in key_moments if m.moment_type == 'peak']
        climaxes = [m for m in key_moments if m.moment_type == 'climax']
        
        # Determine energy level
        if avg_db > -15:
            energy = "High"
        elif avg_db > -25:
            energy = "Medium"
        else:
            energy = "Low"
        
        # Determine mood based on characteristics
        characteristics = []
        mood = "Neutral"
        arc = ""
        
        # Fast + Loud = Energetic
        if tempo and tempo.bpm > 120 and avg_db > -20:
            mood = "Energetic"
            characteristics.append("Fast tempo")
            characteristics.append("High energy")
        
        # Slow + Quiet = Calm
        elif (not tempo or tempo.bpm < 80) and avg_db < -25:
            mood = "Calm"
            characteristics.append("Relaxed pace")
            characteristics.append("Soft dynamics")
        
        # Large dynamic range = Dramatic
        if dynamic_range > 30:
            if mood == "Neutral":
                mood = "Dramatic"
            characteristics.append("Wide dynamic range")
        
        # Has builds and climax = Uplifting/Triumphant
        if builds and climaxes:
            mood = "Uplifting"
            characteristics.append("Building intensity")
            characteristics.append("Clear climax")
            arc = "Builds from calm to powerful climax"
        
        # Rising trend = Building
        if trend > 5:
            if not arc:
                arc = "Gradually building intensity"
            characteristics.append("Rising energy")
        
        # Falling trend = Fading/Resolving
        elif trend < -5:
            if not arc:
                arc = "Gradually fading/resolving"
            characteristics.append("Decreasing energy")
        
        # Mostly consistent = Steady
        if dynamic_range < 15:
            characteristics.append("Consistent dynamics")
            if mood == "Neutral":
                mood = "Steady"
        
        return MoodAnalysis(
            primary_mood=mood,
            energy_level=energy,
            emotional_arc=arc if arc else "Relatively consistent throughout",
            confidence=0.7,  # Heuristic-based
            characteristics=characteristics
        )


# ============================================================================
# SPEECH DETECTOR (Optional)
# ============================================================================

class SpeechDetector:
    """Detect and transcribe speech in audio."""
    
    def __init__(self):
        self._sr_available = self._check_speech_recognition()
    
    def _check_speech_recognition(self) -> bool:
        """Check if speech_recognition is available."""
        try:
            import speech_recognition
            return True
        except ImportError:
            return False
    
    def detect(self, audio_path: str) -> SpeechAnalysis:
        """Detect and transcribe speech."""
        if not self._sr_available:
            logger.warning("SpeechRecognition not available - skipping speech detection")
            return SpeechAnalysis()
        
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            wav_path = self._ensure_wav(audio_path)
            
            try:
                with sr.AudioFile(wav_path) as source:
                    audio = recognizer.record(source)
                
                # Try to transcribe
                try:
                    text = recognizer.recognize_google(audio)
                    
                    return SpeechAnalysis(
                        has_speech=True,
                        speech_ratio=0.5,  # Rough estimate
                        full_transcript=text
                    )
                except sr.UnknownValueError:
                    # No speech detected
                    return SpeechAnalysis(has_speech=False)
                except sr.RequestError as e:
                    logger.warning(f"Speech recognition API error: {e}")
                    return SpeechAnalysis()
                    
            finally:
                if wav_path != audio_path and Path(wav_path).exists():
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Speech detection failed: {e}")
            return SpeechAnalysis()
    
    def _ensure_wav(self, audio_path: str) -> str:
        """Ensure audio is in WAV format."""
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        temp_wav = tempfile.mktemp(suffix='.wav')
        
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', temp_wav],
            capture_output=True
        )
        
        return temp_wav


# ============================================================================
# AUDIO ANALYZER (Main Orchestrator)
# ============================================================================

class AudioAnalyzer:
    """
    Main audio analysis orchestrator.
    Coordinates all analysis components.
    """
    
    def __init__(
        self,
        samples_per_second: int = 10,
        delta_threshold_db: float = 6.0,
        detect_tempo: bool = True,
        detect_speech: bool = False,  # Off by default (slow)
        cleanup_temp: bool = True
    ):
        """
        Initialize audio analyzer.
        
        Args:
            samples_per_second: Voltage gauge sample rate
            delta_threshold_db: Minimum dB change for key moments
            detect_tempo: Whether to detect tempo (requires librosa)
            detect_speech: Whether to detect/transcribe speech
            cleanup_temp: Whether to cleanup temp files
        """
        self.samples_per_second = samples_per_second
        self.delta_threshold_db = delta_threshold_db
        self.detect_tempo = detect_tempo
        self.detect_speech = detect_speech
        self.cleanup_temp = cleanup_temp
        
        # Initialize components
        self._voltage_gauge = VoltageGauge(samples_per_second)
        self._delta_detector = DeltaDetector(delta_threshold_db)
        self._tempo_detector = TempoDetector() if detect_tempo else None
        self._mood_classifier = MoodClassifier()
        self._speech_detector = SpeechDetector() if detect_speech else None
    
    def analyze(
        self,
        audio_path: str,
        analysis_type: str = "comprehensive"
    ) -> AnalysisResult:
        """
        Analyze audio file.
        
        Args:
            audio_path: Path to audio file
            analysis_type: Type of analysis
                - comprehensive: Full analysis
                - quick: Voltage/dB and key moments only
                - voltage_only: Just voltage readings
                
        Returns:
            AnalysisResult with all analysis data
        """
        start_time = datetime.now()
        path = Path(audio_path)
        
        if not path.exists():
            raise AudioNotFoundError(f"Audio not found: {audio_path}")
        
        # Verify dependencies
        DependencyChecker.verify_required()
        
        logger.info(f"Starting {analysis_type} analysis of {path.name}")
        
        # Initialize result
        result = AnalysisResult(
            file_path=str(path.absolute()),
            file_name=path.name,
            analysis_type=analysis_type,
            analysis_timestamp=start_time.isoformat()
        )
        
        # Extract metadata
        logger.info("Extracting metadata...")
        metadata = MetadataExtractor.extract(audio_path)
        result.duration = metadata.duration_formatted
        result.duration_seconds = metadata.duration_seconds
        result.format_name = metadata.format_name
        result.codec = metadata.codec
        result.sample_rate = metadata.sample_rate
        result.channels = metadata.channels
        result.bitrate = metadata.bitrate
        result.file_size_mb = round(metadata.file_size / (1024 * 1024), 2)
        
        # Voltage gauge analysis (Core)
        logger.info("Reading voltage levels...")
        result.voltage_readings = self._voltage_gauge.read_voltages(audio_path)
        
        stats = self._voltage_gauge.get_statistics(result.voltage_readings)
        result.peak_db = stats['peak_db']
        result.average_db = stats['average_db']
        result.dynamic_range_db = stats['dynamic_range_db']
        
        # Delta detection (Key moments)
        logger.info("Detecting key moments...")
        result.key_moments = self._delta_detector.get_all_key_moments(
            result.voltage_readings,
            top_peaks=10
        )
        result.activity_timeline = self._delta_detector.get_activity_timeline(
            result.voltage_readings,
            bucket_seconds=10
        )
        
        if analysis_type == "voltage_only":
            result.summary = self._generate_summary(result)
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
        
        # Tempo detection (Optional)
        if self._tempo_detector and analysis_type == "comprehensive":
            logger.info("Detecting tempo...")
            result.tempo = self._tempo_detector.detect(audio_path)
        
        # Mood classification
        logger.info("Classifying mood...")
        result.mood = self._mood_classifier.classify(
            result.voltage_readings,
            result.tempo,
            result.key_moments
        )
        
        # Speech detection (Optional, comprehensive only)
        if self._speech_detector and analysis_type == "comprehensive":
            logger.info("Detecting speech...")
            result.speech = self._speech_detector.detect(audio_path)
        
        # Generate summary
        result.summary = self._generate_summary(result)
        result.processing_time_seconds = round(
            (datetime.now() - start_time).total_seconds(), 2
        )
        
        logger.info(f"Analysis complete in {result.processing_time_seconds}s")
        
        return result
    
    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate human-readable summary."""
        parts = []
        
        # Basic info
        parts.append(
            f"{result.duration} audio ({result.codec}, "
            f"{result.sample_rate}Hz, {result.bitrate//1000}kbps)"
        )
        
        # Voltage/dB stats
        parts.append(
            f"Peak: {result.peak_db}dB, Avg: {result.average_db}dB, "
            f"Dynamic range: {result.dynamic_range_db}dB"
        )
        
        # Key moments
        if result.key_moments:
            parts.append(f"{len(result.key_moments)} key moments detected")
        
        # Tempo
        if result.tempo:
            parts.append(f"Tempo: {result.tempo.bpm} BPM ({result.tempo.description})")
        
        # Mood
        if result.mood:
            parts.append(
                f"Mood: {result.mood.primary_mood} ({result.mood.energy_level} energy)"
            )
            if result.mood.emotional_arc:
                parts.append(f"Arc: {result.mood.emotional_arc}")
        
        # Speech
        if result.speech and result.speech.has_speech:
            parts.append("Speech detected")
        
        return ". ".join(parts) + "."


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='audioanalysis',
        description='AudioAnalysis - Enable AI agents to listen to audio content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audioanalysis analyze music.mp3
  audioanalysis analyze music.mp3 -o results.json
  audioanalysis analyze music.mp3 --type quick
  audioanalysis voltage music.mp3
  audioanalysis moments music.mp3 --top 10
  audioanalysis check-deps

Built by ATLAS for Team Brain
Voltage Gauge Concept: Logan Smith
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Full audio analysis')
    analyze_parser.add_argument('audio', help='Path to audio file')
    analyze_parser.add_argument('-o', '--output', help='Output JSON file')
    analyze_parser.add_argument(
        '-t', '--type',
        choices=['comprehensive', 'quick', 'voltage_only'],
        default='comprehensive',
        help='Analysis type (default: comprehensive)'
    )
    analyze_parser.add_argument(
        '-s', '--samples-per-second',
        type=int, default=10,
        help='Voltage samples per second (default: 10)'
    )
    analyze_parser.add_argument(
        '-d', '--delta-threshold',
        type=float, default=6.0,
        help='Delta threshold in dB (default: 6.0)'
    )
    analyze_parser.add_argument(
        '--no-tempo', action='store_true',
        help='Skip tempo detection'
    )
    analyze_parser.add_argument(
        '--speech', action='store_true',
        help='Enable speech detection/transcription'
    )
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Voltage command
    voltage_parser = subparsers.add_parser('voltage', help='Get voltage/dB readings')
    voltage_parser.add_argument('audio', help='Path to audio file')
    voltage_parser.add_argument('-o', '--output', help='Output JSON file')
    voltage_parser.add_argument(
        '-s', '--samples-per-second',
        type=int, default=10,
        help='Samples per second (default: 10)'
    )
    
    # Moments command
    moments_parser = subparsers.add_parser('moments', help='Find key moments')
    moments_parser.add_argument('audio', help='Path to audio file')
    moments_parser.add_argument('-o', '--output', help='Output JSON file')
    moments_parser.add_argument(
        '--top', type=int, default=10,
        help='Number of top moments (default: 10)'
    )
    moments_parser.add_argument(
        '-d', '--delta-threshold',
        type=float, default=6.0,
        help='Delta threshold in dB (default: 6.0)'
    )
    
    # Timeline command
    timeline_parser = subparsers.add_parser('timeline', help='Activity timeline')
    timeline_parser.add_argument('audio', help='Path to audio file')
    timeline_parser.add_argument('-o', '--output', help='Output JSON file')
    timeline_parser.add_argument(
        '-b', '--bucket-seconds',
        type=int, default=10,
        help='Seconds per bucket (default: 10)'
    )
    
    # Check dependencies
    subparsers.add_parser('check-deps', help='Check dependencies')
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'check-deps':
            print(DependencyChecker.get_status_report())
            return 0
        
        elif args.command == 'analyze':
            if args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            
            analyzer = AudioAnalyzer(
                samples_per_second=args.samples_per_second,
                delta_threshold_db=args.delta_threshold,
                detect_tempo=not args.no_tempo,
                detect_speech=args.speech
            )
            
            result = analyzer.analyze(args.audio, analysis_type=args.type)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(result), f, indent=2)
                print(f"Saved analysis to {args.output}")
            else:
                # Print summary
                print("=" * 60)
                print("AUDIO ANALYSIS COMPLETE")
                print("=" * 60)
                print(f"File: {result.file_name}")
                print(f"Duration: {result.duration}")
                print(f"Format: {result.format_name} ({result.codec})")
                print(f"Sample rate: {result.sample_rate} Hz")
                print(f"Bitrate: {result.bitrate // 1000} kbps")
                print(f"Size: {result.file_size_mb} MB")
                print()
                print("Voltage/dB Analysis:")
                print(f"  Peak: {result.peak_db} dB")
                print(f"  Average: {result.average_db} dB")
                print(f"  Dynamic range: {result.dynamic_range_db} dB")
                print()
                
                if result.key_moments:
                    print(f"Key Moments ({len(result.key_moments)}):")
                    for moment in result.key_moments[:5]:
                        print(f"  [{moment.timestamp}] {moment.moment_type}: {moment.description}")
                    if len(result.key_moments) > 5:
                        print(f"  ... and {len(result.key_moments) - 5} more")
                    print()
                
                if result.tempo:
                    print(f"Tempo: {result.tempo.bpm} BPM - {result.tempo.description}")
                    print()
                
                if result.mood:
                    print(f"Mood: {result.mood.primary_mood}")
                    print(f"Energy: {result.mood.energy_level}")
                    if result.mood.emotional_arc:
                        print(f"Arc: {result.mood.emotional_arc}")
                    if result.mood.characteristics:
                        print(f"Characteristics: {', '.join(result.mood.characteristics)}")
                    print()
                
                if result.speech and result.speech.has_speech:
                    print("Speech detected!")
                    if result.speech.full_transcript:
                        print(f"Transcript: {result.speech.full_transcript[:200]}...")
                    print()
                
                if result.activity_timeline:
                    print("Activity Timeline:")
                    for bucket in result.activity_timeline[:10]:
                        bar = "#" * int(max(0, (bucket.avg_db + 50) / 5))
                        print(f"  {bucket.start_time}-{bucket.end_time}: "
                              f"{bucket.activity_level:8s} [{bar:<10}] "
                              f"avg:{bucket.avg_db:.1f}dB")
                    if len(result.activity_timeline) > 10:
                        print(f"  ... and {len(result.activity_timeline) - 10} more buckets")
                
                print()
                print(f"Processing time: {result.processing_time_seconds}s")
            
            return 0
        
        elif args.command == 'voltage':
            gauge = VoltageGauge(samples_per_second=args.samples_per_second)
            readings = gauge.read_voltages(args.audio)
            stats = gauge.get_statistics(readings)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump({
                        'readings': [asdict(r) for r in readings],
                        'statistics': stats
                    }, f, indent=2)
                print(f"Saved voltage readings to {args.output}")
            else:
                print(f"Voltage Analysis for: {args.audio}")
                print("=" * 50)
                print(f"Samples: {len(readings)}")
                print(f"Peak: {stats['peak_db']} dB")
                print(f"Average: {stats['average_db']} dB")
                print(f"Dynamic range: {stats['dynamic_range_db']} dB")
                print()
                print("Sample readings:")
                for r in readings[:20]:
                    bar = "#" * int(max(0, (r.db_level + 50) / 5))
                    print(f"  [{r.timestamp}] {r.db_level:6.1f} dB [{bar}]")
                if len(readings) > 20:
                    print(f"  ... and {len(readings) - 20} more")
            
            return 0
        
        elif args.command == 'moments':
            gauge = VoltageGauge()
            detector = DeltaDetector(threshold_db=args.delta_threshold)
            
            readings = gauge.read_voltages(args.audio)
            moments = detector.get_all_key_moments(readings, top_peaks=args.top)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(m) for m in moments], f, indent=2)
                print(f"Saved key moments to {args.output}")
            else:
                print(f"Key Moments for: {args.audio}")
                print("=" * 50)
                for i, m in enumerate(moments, 1):
                    print(f"{i}. [{m.timestamp}] {m.moment_type}")
                    print(f"   {m.description}")
                    print(f"   Level: {m.db_level:.1f} dB, Delta: {m.delta_from_previous:+.1f} dB")
                    print()
            
            return 0
        
        elif args.command == 'timeline':
            gauge = VoltageGauge()
            detector = DeltaDetector()
            
            readings = gauge.read_voltages(args.audio)
            timeline = detector.get_activity_timeline(readings, bucket_seconds=args.bucket_seconds)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(b) for b in timeline], f, indent=2)
                print(f"Saved timeline to {args.output}")
            else:
                print(f"Activity Timeline for: {args.audio}")
                print("=" * 50)
                for bucket in timeline:
                    bar = "#" * int(max(0, (bucket.avg_db + 50) / 5))
                    print(f"  {bucket.start_time}-{bucket.end_time}: "
                          f"{bucket.activity_level:8s} [{bar:<10}] "
                          f"avg:{bucket.avg_db:.1f}dB max:{bucket.max_db:.1f}dB")
            
            return 0
        
        else:
            parser.print_help()
            return 1
    
    except AudioAnalysisError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
