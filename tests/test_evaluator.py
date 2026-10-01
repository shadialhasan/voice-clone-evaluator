# -*- coding: utf-8 -*-
"""
Automated Unit Tests for Voice Clone Evaluator
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
"""

import math
import json
import csv
import wave
import struct
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import (
    read_wav_samples,
    calculate_rms_energy,
    calculate_dbfs,
    calculate_peak_amplitude,
    calculate_crest_factor,
    calculate_zero_crossing_rate,
    calculate_spectral_centroid_sim,
    profile_audio,
    evaluate_pair,
    export_metrics_json,
    export_metrics_csv,
    parse_args,
)


def create_synthetic_wav(
    file_path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
    amplitude: int = 16000,
    freq: float = 440.0
):
    """Generates a test WAV file with specified acoustic properties."""
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_frames):
            t = i / float(sample_rate)
            val = int(amplitude * math.sin(2 * math.pi * freq * t))
            for _ in range(channels):
                frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)


class TestAcousticMath:
    """Tests for core acoustic mathematical operations."""

    def test_spectral_centroid_sim_identical(self):
        samples = [1000, -2000, 3000, -1500, 500]
        score = calculate_spectral_centroid_sim(samples, samples)
        assert score == 100.0

    def test_spectral_centroid_sim_silence(self):
        silence1 = [0, 0, 0, 0]
        silence2 = [0, 0, 0, 0]
        assert calculate_spectral_centroid_sim(silence1, silence2) == 100.0

        active = [100, 200, 300]
        assert calculate_spectral_centroid_sim(silence1, active) == 0.0
        assert calculate_spectral_centroid_sim(active, silence1) == 0.0

    def test_spectral_centroid_sim_energy_ratio(self):
        # When energy of s2 is exactly 50% of s1
        # Energy = sum(s^2) / N
        s1 = [100, 100]  # energy = (10000 + 10000)/2 = 10000
        s2 = [100, 0]    # energy = (10000 + 0)/2 = 5000
        score = calculate_spectral_centroid_sim(s1, s2)
        assert score == 50.0

    def test_rms_energy_calculation(self):
        assert calculate_rms_energy([]) == 0.0
        # DC signal: all samples equal 500
        dc_samples = [500, 500, 500, 500]
        assert calculate_rms_energy(dc_samples) == 500.0

    def test_dbfs_calculation(self):
        assert calculate_dbfs(0.0) == -100.0
        # Full scale 32768 RMS -> 0.0 dBFS
        assert calculate_dbfs(32768.0) == 0.0
        # Half scale amplitude -> approx -6.02 dBFS
        half_scale_db = calculate_dbfs(16384.0)
        assert abs(half_scale_db - (-6.02)) < 0.1

    def test_peak_and_crest_factor(self):
        assert calculate_peak_amplitude([]) == 0
        samples = [120, -950, 400, -100]
        assert calculate_peak_amplitude(samples) == 950

        # Crest factor: peak / rms
        rms = calculate_rms_energy([500, 500])
        crest = calculate_crest_factor(500, rms)
        assert crest == 1.0
        assert calculate_crest_factor(500, 0.0) == 0.0

    def test_zero_crossing_rate(self):
        assert calculate_zero_crossing_rate([]) == 0.0
        assert calculate_zero_crossing_rate([100]) == 0.0
        # Constant positive -> 0 crossings
        assert calculate_zero_crossing_rate([100, 200, 300, 400]) == 0.0
        # Alternating sign every sample -> 3 crossings over 3 intervals = 1.0
        assert calculate_zero_crossing_rate([100, -100, 100, -100]) == 1.0


class TestWavIOAndProfiling:
    """Tests for WAV file decoding and profiling."""

    def test_read_wav_samples_mono(self, tmp_path):
        wav_file = tmp_path / "mono_test.wav"
        create_synthetic_wav(wav_file, duration_seconds=0.5, sample_rate=22050, channels=1)

        samples, rate = read_wav_samples(wav_file)
        assert rate == 22050
        assert len(samples) == int(0.5 * 22050)

    def test_read_wav_samples_stereo_downmix(self, tmp_path):
        wav_file = tmp_path / "stereo_test.wav"
        create_synthetic_wav(wav_file, duration_seconds=0.5, sample_rate=44100, channels=2)

        samples, rate = read_wav_samples(wav_file)
        assert rate == 44100
        # When downmixed, length should equal n_frames
        assert len(samples) == int(0.5 * 44100)

    def test_read_wav_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_wav_samples(tmp_path / "missing.wav")

    def test_profile_audio(self):
        samples = [1000, -1000, 1000, -1000]
        profile = profile_audio(samples, 4)
        assert profile["sample_count"] == 4
        assert profile["duration_seconds"] == 1.0
        assert profile["peak_amplitude"] == 1000
        assert profile["rms_energy"] == 1000.0


class TestEvaluationPipeline:
    """Tests for full evaluation and export workflow."""

    def test_evaluate_pair_matching_audio(self, tmp_path):
        ref_wav = tmp_path / "ref.wav"
        syn_wav = tmp_path / "syn.wav"
        create_synthetic_wav(ref_wav, duration_seconds=1.0, amplitude=12000, freq=440.0)
        create_synthetic_wav(syn_wav, duration_seconds=1.0, amplitude=12000, freq=440.0)

        results = evaluate_pair(ref_wav, syn_wav, threshold=80.0)
        comp = results["comparisons"]

        assert comp["energy_similarity_score"] == 100.0
        assert comp["duration_match_percentage"] == 100.0
        assert comp["overall_similarity_score"] == 100.0
        assert comp["benchmark_passed"] is True
        assert comp["quality_tier"] == "EXCELLENT"

    def test_evaluate_pair_mismatch(self, tmp_path):
        ref_wav = tmp_path / "ref.wav"
        syn_wav = tmp_path / "syn.wav"
        create_synthetic_wav(ref_wav, duration_seconds=2.0, amplitude=25000, freq=440.0)
        # Much lower energy and shorter duration
        create_synthetic_wav(syn_wav, duration_seconds=0.5, amplitude=2000, freq=220.0)

        results = evaluate_pair(ref_wav, syn_wav, threshold=80.0)
        comp = results["comparisons"]

        assert comp["benchmark_passed"] is False
        assert comp["overall_similarity_score"] < 50.0

    def test_export_metrics_json_and_csv(self, tmp_path):
        ref_wav = tmp_path / "ref.wav"
        syn_wav = tmp_path / "syn.wav"
        json_out = tmp_path / "metrics.json"
        csv_out = tmp_path / "metrics.csv"

        create_synthetic_wav(ref_wav, duration_seconds=0.5, amplitude=8000)
        create_synthetic_wav(syn_wav, duration_seconds=0.5, amplitude=8000)

        results = evaluate_pair(ref_wav, syn_wav)

        export_metrics_json(results, json_out)
        assert json_out.exists()
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["reference_file"] == "ref.wav"

        export_metrics_csv(results, csv_out)
        assert csv_out.exists()
        with open(csv_out, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["reference_file"] == "ref.wav"


class TestCliArgs:
    """Tests for evaluator CLI argument parsing."""

    def test_cli_requires_ref_and_synth(self):
        with pytest.raises(SystemExit):
            parse_args([])

        with pytest.raises(SystemExit):
            parse_args(["--ref", "ref.wav"])

    def test_cli_custom_arguments(self):
        args = parse_args([
            "-r", "truth.wav",
            "-s", "clone.wav",
            "-t", "85.5",
            "--verbose",
            "-j", "out.json",
            "-c", "out.csv"
        ])
        assert args.ref == "truth.wav"
        assert args.synth == "clone.wav"
        assert args.threshold == 85.5
        assert args.verbose is True
        assert args.export_json == "out.json"
        assert args.export_csv == "out.csv"
