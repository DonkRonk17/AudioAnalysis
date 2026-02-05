#!/usr/bin/env python3
"""
AudioAnalysis - Setup Configuration

Enable AI agents to listen and analyze audio content.

Requested by: Logan Smith (via WSL_CLIO)
Voltage Gauge Concept: Logan Smith
Built by: ATLAS (Team Brain)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="audioanalysis",
    version="1.0.0",
    author="ATLAS (Team Brain)",
    author_email="contact@metaphysicsandcomputing.com",
    description="Enable AI agents to listen and analyze audio content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DonkRonk17/AudioAnalysis",
    py_modules=["audioanalysis"],
    python_requires=">=3.9",
    
    # No required pip dependencies - just FFmpeg!
    install_requires=[],
    
    # Optional features
    extras_require={
        "tempo": ["librosa>=0.10.0"],
        "speech": ["SpeechRecognition>=3.10.0"],
        "full": ["librosa>=0.10.0", "SpeechRecognition>=3.10.0"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
    
    entry_points={
        "console_scripts": [
            "audioanalysis=audioanalysis:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    keywords="audio analysis voltage gauge dB decibel tempo mood AI agents",
    
    project_urls={
        "Bug Reports": "https://github.com/DonkRonk17/AudioAnalysis/issues",
        "Source": "https://github.com/DonkRonk17/AudioAnalysis",
        "Team Brain": "https://metaphysicsandcomputing.com",
    },
)
