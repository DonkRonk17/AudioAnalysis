#!/usr/bin/env python3
"""
Test Suite for AudioAnalysis
Built using BUILD_PROTOCOL_V1.md + Bug Hunt Protocol

Components Tested:
- DependencyChecker: Verifying external tools
- MetadataExtractor: FFprobe integration
- VoltageGauge: Logan's core insight - amplitude as voltage
- DeltaDetector: Key moment detection
- TempoDetector: BPM detection
- MoodClassifier: Heuristic mood classification
- AudioAnalyzer: Main orchestrator

Built by ATLAS for Team Brain
Voltage Gauge Concept: Logan Smith
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from audioanalysis import (
    # Data classes
    AudioMetadata,
    VoltageReading,
    KeyMoment,
    ActivityBucket,
    TempoInfo,
    MoodAnalysis,
    SpeechAnalysis,
    AnalysisResult,
    
    # Exceptions
    AudioAnalysisError,
    DependencyError,
    AudioNotFoundError,
    ProcessingError,
    UnsupportedFormatError,
    
    # Components
    DependencyChecker,
    MetadataExtractor,
    VoltageGauge,
    DeltaDetector,
    TempoDetector,
    MoodClassifier,
    AudioAnalyzer,
    
    # CLI
    create_parser,
    main,
    
    # Version
    __version__,
)


class TestAudioMetadataDataclass(unittest.TestCase):
    """Test AudioMetadata dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        meta = AudioMetadata()
        self.assertEqual(meta.duration_seconds, 0.0)
        self.assertEqual(meta.duration_formatted, "00:00:00")
        self.assertEqual(meta.sample_rate, 0)
        self.assertEqual(meta.channels, 0)
    
    def test_custom_values(self):
        """Test custom values are stored correctly."""
        meta = AudioMetadata(
            duration_seconds=180.5,
            duration_formatted="00:03:00",
            sample_rate=44100,
            channels=2,
            bitrate=320000,
            codec="mp3",
            format_name="mp3",
            file_size=5000000
        )
        self.assertEqual(meta.duration_seconds, 180.5)
        self.assertEqual(meta.sample_rate, 44100)
        self.assertEqual(meta.codec, "mp3")


class TestVoltageReadingDataclass(unittest.TestCase):
    """Test VoltageReading dataclass - Logan's core concept."""
    
    def test_create_voltage_reading(self):
        """Test creating a voltage reading."""
        reading = VoltageReading(
            timestamp="01:30",
            time_seconds=90.0,
            amplitude_rms=0.5,
            db_level=-6.02
        )
        self.assertEqual(reading.timestamp, "01:30")
        self.assertEqual(reading.time_seconds, 90.0)
        self.assertEqual(reading.amplitude_rms, 0.5)
        self.assertAlmostEqual(reading.db_level, -6.02, places=2)


class TestKeyMomentDataclass(unittest.TestCase):
    """Test KeyMoment dataclass."""
    
    def test_create_key_moment(self):
        """Test creating a key moment."""
        moment = KeyMoment(
            timestamp="02:10",
            time_seconds=130.0,
            moment_type="climax",
            db_level=-3.5,
            delta_from_previous=8.2,
            description="Main climax"
        )
        self.assertEqual(moment.moment_type, "climax")
        self.assertEqual(moment.db_level, -3.5)


class TestActivityBucketDataclass(unittest.TestCase):
    """Test ActivityBucket dataclass."""
    
    def test_create_activity_bucket(self):
        """Test creating an activity bucket."""
        bucket = ActivityBucket(
            start_time="00:00",
            end_time="00:10",
            avg_db=-15.5,
            max_db=-8.2,
            activity_level="medium"
        )
        self.assertEqual(bucket.activity_level, "medium")
        self.assertEqual(bucket.avg_db, -15.5)


class TestAnalysisResultDataclass(unittest.TestCase):
    """Test AnalysisResult dataclass."""
    
    def test_create_result(self):
        """Test creating analysis result."""
        result = AnalysisResult(
            file_path="/path/to/audio.mp3",
            file_name="audio.mp3"
        )
        self.assertEqual(result.file_name, "audio.mp3")
        self.assertEqual(result.tool_version, __version__)
    
    def test_default_lists(self):
        """Test default empty lists."""
        result = AnalysisResult(
            file_path="/path/to/audio.mp3",
            file_name="audio.mp3"
        )
        self.assertEqual(result.voltage_readings, [])
        self.assertEqual(result.key_moments, [])
        self.assertEqual(result.activity_timeline, [])


class TestDependencyChecker(unittest.TestCase):
    """Test DependencyChecker class."""
    
    def test_check_command_success(self):
        """Test checking a command that exists (python)."""
        result = DependencyChecker._check_command(['python', '--version'])
        self.assertTrue(result)
    
    def test_check_command_failure(self):
        """Test checking a command that doesn't exist."""
        result = DependencyChecker._check_command(['nonexistent_command_xyz'])
        self.assertFalse(result)
    
    def test_check_python_module_success(self):
        """Test checking a module that exists."""
        result = DependencyChecker._check_python_module('os')
        self.assertTrue(result)
    
    def test_check_python_module_failure(self):
        """Test checking a module that doesn't exist."""
        result = DependencyChecker._check_python_module('nonexistent_module_xyz')
        self.assertFalse(result)
    
    def test_get_status_report(self):
        """Test status report generation."""
        report = DependencyChecker.get_status_report()
        self.assertIn("AudioAnalysis Dependency Status", report)
        self.assertIn("Required:", report)
        self.assertIn("Optional:", report)
    
    def test_check_all_returns_dict(self):
        """Test check_all returns dictionary."""
        results = DependencyChecker.check_all()
        self.assertIsInstance(results, dict)
        self.assertIn('ffmpeg', results)
        self.assertIn('ffprobe', results)


class TestVoltageGauge(unittest.TestCase):
    """Test VoltageGauge class - Logan's core insight."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gauge = VoltageGauge(samples_per_second=10)
    
    def test_init_with_samples_per_second(self):
        """Test initialization with custom sample rate."""
        gauge = VoltageGauge(samples_per_second=20)
        self.assertEqual(gauge.samples_per_second, 20)
    
    def test_amplitude_to_db_max(self):
        """Test dB conversion at maximum amplitude."""
        db = self.gauge._amplitude_to_db(1.0)
        self.assertEqual(db, 0.0)  # 1.0 amplitude = 0 dB
    
    def test_amplitude_to_db_half(self):
        """Test dB conversion at half amplitude."""
        db = self.gauge._amplitude_to_db(0.5)
        self.assertAlmostEqual(db, -6.02, places=1)  # Half = -6 dB
    
    def test_amplitude_to_db_silence(self):
        """Test dB conversion at silence."""
        db = self.gauge._amplitude_to_db(0.0)
        self.assertEqual(db, -100.0)  # Practical minimum
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        self.assertEqual(self.gauge._format_timestamp(0), "00:00")
        self.assertEqual(self.gauge._format_timestamp(61), "01:01")
        self.assertEqual(self.gauge._format_timestamp(125), "02:05")
    
    def test_get_statistics_empty(self):
        """Test statistics with empty readings."""
        stats = self.gauge.get_statistics([])
        self.assertEqual(stats['peak_db'], -100.0)
        self.assertEqual(stats['average_db'], -100.0)
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        readings = [
            VoltageReading("00:00", 0.0, 0.5, -6.0),
            VoltageReading("00:01", 1.0, 0.25, -12.0),
            VoltageReading("00:02", 2.0, 1.0, 0.0),
        ]
        stats = self.gauge.get_statistics(readings)
        self.assertEqual(stats['peak_db'], 0.0)
        self.assertEqual(stats['min_db'], -12.0)


class TestDeltaDetector(unittest.TestCase):
    """Test DeltaDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = DeltaDetector(threshold_db=6.0)
    
    def test_init_with_threshold(self):
        """Test initialization with threshold."""
        detector = DeltaDetector(threshold_db=10.0)
        self.assertEqual(detector.threshold_db, 10.0)
    
    def test_find_peaks_empty(self):
        """Test peak finding with empty list."""
        peaks = self.detector.find_peaks([])
        self.assertEqual(peaks, [])
    
    def test_find_peaks_too_short(self):
        """Test peak finding with list too short."""
        readings = [
            VoltageReading("00:00", 0.0, 0.5, -6.0),
            VoltageReading("00:01", 1.0, 0.25, -12.0),
        ]
        peaks = self.detector.find_peaks(readings)
        self.assertEqual(peaks, [])
    
    def test_find_peaks_with_peak(self):
        """Test peak finding with actual peak."""
        readings = [
            VoltageReading("00:00", 0.0, 0.1, -20.0),
            VoltageReading("00:01", 1.0, 0.5, -6.0),   # Peak
            VoltageReading("00:02", 2.0, 0.1, -20.0),
        ]
        peaks = self.detector.find_peaks(readings)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0].moment_type, 'peak')
        self.assertEqual(peaks[0].db_level, -6.0)
    
    def test_find_drops(self):
        """Test drop detection."""
        readings = [
            VoltageReading("00:00", 0.0, 1.0, 0.0),     # Loud
            VoltageReading("00:01", 1.0, 0.1, -20.0),   # Drop!
        ]
        drops = self.detector.find_drops(readings)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].moment_type, 'drop')
    
    def test_get_activity_timeline_empty(self):
        """Test timeline with empty readings."""
        timeline = self.detector.get_activity_timeline([])
        self.assertEqual(timeline, [])
    
    def test_get_activity_timeline(self):
        """Test activity timeline generation."""
        readings = [
            VoltageReading(f"00:{i:02d}", float(i), 0.5, -6.0)
            for i in range(20)
        ]
        timeline = self.detector.get_activity_timeline(readings, bucket_seconds=5)
        self.assertGreater(len(timeline), 0)
        for bucket in timeline:
            self.assertIn(bucket.activity_level, ['high', 'medium', 'low', 'silent'])


class TestTempoDetector(unittest.TestCase):
    """Test TempoDetector class."""
    
    def test_describe_tempo_slow(self):
        """Test tempo description for slow tempo."""
        detector = TempoDetector()
        desc = detector._describe_tempo(60)
        self.assertIn("Slow", desc)
    
    def test_describe_tempo_moderate(self):
        """Test tempo description for moderate tempo."""
        detector = TempoDetector()
        desc = detector._describe_tempo(100)
        self.assertIn("moderato", desc.lower())  # "Walking pace (moderato)"
    
    def test_describe_tempo_fast(self):
        """Test tempo description for fast tempo."""
        detector = TempoDetector()
        desc = detector._describe_tempo(150)
        self.assertIn("fast", desc.lower())


class TestMoodClassifier(unittest.TestCase):
    """Test MoodClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MoodClassifier()
    
    def test_classify_empty_readings(self):
        """Test classification with empty readings."""
        result = self.classifier.classify([], None, [])
        self.assertEqual(result.primary_mood, "Unknown")
    
    def test_classify_high_energy(self):
        """Test classification of high energy content."""
        readings = [
            VoltageReading(f"00:{i:02d}", float(i), 0.8, -5.0)
            for i in range(30)
        ]
        tempo = TempoInfo(bpm=140, confidence=0.8, beat_count=100, description="Fast")
        
        result = self.classifier.classify(readings, tempo, [])
        self.assertEqual(result.energy_level, "High")
    
    def test_classify_low_energy(self):
        """Test classification of low energy content."""
        readings = [
            VoltageReading(f"00:{i:02d}", float(i), 0.05, -30.0)
            for i in range(30)
        ]
        
        result = self.classifier.classify(readings, None, [])
        self.assertEqual(result.energy_level, "Low")


class TestAudioAnalyzer(unittest.TestCase):
    """Test AudioAnalyzer class."""
    
    def test_init_default_values(self):
        """Test default initialization values."""
        analyzer = AudioAnalyzer()
        self.assertEqual(analyzer.samples_per_second, 10)
        self.assertEqual(analyzer.delta_threshold_db, 6.0)
    
    def test_init_custom_values(self):
        """Test custom initialization values."""
        analyzer = AudioAnalyzer(
            samples_per_second=20,
            delta_threshold_db=10.0,
            detect_tempo=False,
            detect_speech=True
        )
        self.assertEqual(analyzer.samples_per_second, 20)
        self.assertEqual(analyzer.delta_threshold_db, 10.0)
    
    def test_analyze_nonexistent_audio(self):
        """Test analysis of non-existent audio."""
        analyzer = AudioAnalyzer()
        with self.assertRaises(AudioNotFoundError):
            analyzer.analyze("/nonexistent/audio.mp3")


class TestCLI(unittest.TestCase):
    """Test CLI interface."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        self.assertIsNotNone(parser)
    
    def test_parse_check_deps(self):
        """Test parsing check-deps command."""
        parser = create_parser()
        args = parser.parse_args(['check-deps'])
        self.assertEqual(args.command, 'check-deps')
    
    def test_parse_analyze(self):
        """Test parsing analyze command."""
        parser = create_parser()
        args = parser.parse_args(['analyze', 'audio.mp3'])
        self.assertEqual(args.command, 'analyze')
        self.assertEqual(args.audio, 'audio.mp3')
    
    def test_parse_analyze_with_options(self):
        """Test parsing analyze with options."""
        parser = create_parser()
        args = parser.parse_args([
            'analyze', 'audio.mp3',
            '-o', 'output.json',
            '-t', 'quick',
            '-s', '20',
            '-d', '10.0'
        ])
        self.assertEqual(args.output, 'output.json')
        self.assertEqual(args.type, 'quick')
        self.assertEqual(args.samples_per_second, 20)
        self.assertEqual(args.delta_threshold, 10.0)
    
    def test_parse_voltage_command(self):
        """Test parsing voltage command."""
        parser = create_parser()
        args = parser.parse_args([
            'voltage', 'audio.mp3',
            '-s', '30'
        ])
        self.assertEqual(args.command, 'voltage')
        self.assertEqual(args.samples_per_second, 30)
    
    def test_parse_moments_command(self):
        """Test parsing moments command."""
        parser = create_parser()
        args = parser.parse_args([
            'moments', 'audio.mp3',
            '--top', '15',
            '-d', '8.0'
        ])
        self.assertEqual(args.command, 'moments')
        self.assertEqual(args.top, 15)
        self.assertEqual(args.delta_threshold, 8.0)
    
    def test_parse_timeline_command(self):
        """Test parsing timeline command."""
        parser = create_parser()
        args = parser.parse_args([
            'timeline', 'audio.mp3',
            '-b', '15'
        ])
        self.assertEqual(args.command, 'timeline')
        self.assertEqual(args.bucket_seconds, 15)


class TestExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_audio_analysis_error(self):
        """Test base exception."""
        with self.assertRaises(AudioAnalysisError):
            raise AudioAnalysisError("Test error")
    
    def test_dependency_error(self):
        """Test dependency error."""
        with self.assertRaises(DependencyError):
            raise DependencyError("Missing FFmpeg")
    
    def test_audio_not_found_error(self):
        """Test audio not found error."""
        with self.assertRaises(AudioNotFoundError):
            raise AudioNotFoundError("audio.mp3 not found")
    
    def test_processing_error(self):
        """Test processing error."""
        with self.assertRaises(ProcessingError):
            raise ProcessingError("Processing failed")
    
    def test_unsupported_format_error(self):
        """Test unsupported format error."""
        with self.assertRaises(UnsupportedFormatError):
            raise UnsupportedFormatError("Format not supported")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_audio_path(self):
        """Test handling of empty audio path raises appropriate error."""
        analyzer = AudioAnalyzer()
        with self.assertRaises(AudioAnalysisError):
            analyzer.analyze("")
    
    def test_voltage_gauge_zero_samples(self):
        """Test voltage gauge with zero samples per second."""
        gauge = VoltageGauge(samples_per_second=0)
        self.assertEqual(gauge.samples_per_second, 0)
    
    def test_delta_detector_extreme_thresholds(self):
        """Test delta detector with extreme thresholds."""
        # Very high threshold - should detect nothing
        detector_high = DeltaDetector(threshold_db=100.0)
        self.assertEqual(detector_high.threshold_db, 100.0)
        
        # Very low threshold - should detect everything
        detector_low = DeltaDetector(threshold_db=0.0)
        self.assertEqual(detector_low.threshold_db, 0.0)
    
    def test_mood_classifier_all_silence(self):
        """Test mood classifier with all silence."""
        readings = [
            VoltageReading(f"00:{i:02d}", float(i), 0.0, -100.0)
            for i in range(10)
        ]
        classifier = MoodClassifier()
        result = classifier.classify(readings, None, [])
        # Should not crash, may return unknown or low energy
        self.assertIsNotNone(result)


class TestDataclassSerialization(unittest.TestCase):
    """Test dataclass serialization."""
    
    def test_audio_metadata_to_dict(self):
        """Test AudioMetadata serialization."""
        meta = AudioMetadata(
            duration_seconds=180.0,
            sample_rate=44100,
            channels=2
        )
        data = asdict(meta)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['sample_rate'], 44100)
    
    def test_analysis_result_to_dict(self):
        """Test AnalysisResult serialization."""
        result = AnalysisResult(
            file_path="/path/audio.mp3",
            file_name="audio.mp3"
        )
        data = asdict(result)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['file_name'], "audio.mp3")
    
    def test_voltage_reading_to_dict(self):
        """Test VoltageReading serialization."""
        reading = VoltageReading(
            timestamp="01:30",
            time_seconds=90.0,
            amplitude_rms=0.5,
            db_level=-6.0
        )
        data = asdict(reading)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['db_level'], -6.0)
    
    def test_key_moment_to_dict(self):
        """Test KeyMoment serialization."""
        moment = KeyMoment(
            timestamp="02:00",
            time_seconds=120.0,
            moment_type="peak",
            db_level=-3.0,
            delta_from_previous=8.0
        )
        data = asdict(moment)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['moment_type'], "peak")


class TestVoltageGaugeConversions(unittest.TestCase):
    """Test VoltageGauge dB conversions - critical for accuracy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gauge = VoltageGauge()
    
    def test_db_conversion_reference_points(self):
        """Test known dB conversion reference points."""
        # 1.0 amplitude = 0 dB (maximum)
        self.assertEqual(self.gauge._amplitude_to_db(1.0), 0.0)
        
        # 0.5 amplitude ≈ -6.02 dB (half voltage)
        db_half = self.gauge._amplitude_to_db(0.5)
        self.assertAlmostEqual(db_half, -6.02, places=1)
        
        # 0.1 amplitude = -20 dB
        db_tenth = self.gauge._amplitude_to_db(0.1)
        self.assertAlmostEqual(db_tenth, -20.0, places=1)
        
        # 0.01 amplitude = -40 dB
        db_hundredth = self.gauge._amplitude_to_db(0.01)
        self.assertAlmostEqual(db_hundredth, -40.0, places=1)
    
    def test_db_conversion_silence_handling(self):
        """Test dB conversion handles silence correctly."""
        # Zero amplitude should return practical minimum
        self.assertEqual(self.gauge._amplitude_to_db(0.0), -100.0)
        
        # Negative amplitude (shouldn't happen but handle it)
        self.assertEqual(self.gauge._amplitude_to_db(-0.5), -100.0)


class TestIntegration(unittest.TestCase):
    """Integration tests (require FFmpeg)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if FFmpeg is available for integration tests."""
        cls.ffmpeg_available = DependencyChecker._check_command(['ffmpeg', '-version'])
    
    def test_dependency_check_integration(self):
        """Test full dependency check."""
        results = DependencyChecker.check_all(include_optional=True)
        self.assertIn('ffmpeg', results)
        self.assertIn('librosa', results)
    
    @unittest.skipUnless(
        DependencyChecker._check_command(['ffmpeg', '-version']),
        "FFmpeg not available"
    )
    def test_metadata_extractor_requires_valid_file(self):
        """Test metadata extractor requires valid file."""
        with self.assertRaises(AudioNotFoundError):
            MetadataExtractor.extract("/definitely/not/a/real/audio.mp3")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
