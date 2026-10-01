# -*- coding: utf-8 -*-
"""
Voice Clone Quality & Similarity Evaluator
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
Phone: +963934005922
Copyright (c) 2026 MHD. Shadi AL-Hasan
"""

import os
import sys
import math
import csv
import json
import wave
import struct
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def get_env_or_default(key: str, default: str) -> str:
    """Reads environment variable with fallback default."""
    return os.environ.get(key, default)


def read_wav_samples(path: Path) -> Tuple[List[int], int]:
    """Reads uncompressed PCM WAV audio and returns 16-bit signed sample list and sample rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(n_frames)

        if sampwidth != 2:
            raise ValueError(f"Unsupported sample width: {sampwidth * 8}-bit. Only 16-bit PCM WAV supported.")

        samples = list(struct.unpack(f"<{n_frames * channels}h", raw))
        if channels == 2:
            # Downmix stereo to mono by taking left channel
            samples = [samples[i] for i in range(0, len(samples), 2)]
        return samples, framerate


def calculate_rms_energy(samples: List[int]) -> float:
    """Calculates Root Mean Square (RMS) energy across samples."""
    if not samples:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / len(samples))


def calculate_dbfs(rms: float, max_amplitude: float = 32768.0) -> float:
    """Converts RMS amplitude to decibels relative to Full Scale (dBFS)."""
    if rms <= 0.0:
        return -100.0
    ratio = max(rms / max_amplitude, 1e-5)
    return round(20 * math.log10(ratio), 2)


def calculate_peak_amplitude(samples: List[int]) -> int:
    """Calculates peak absolute amplitude in sample list."""
    if not samples:
        return 0
    return max(abs(s) for s in samples)


def calculate_crest_factor(peak: int, rms: float) -> float:
    """Calculates Crest Factor (Peak to RMS ratio)."""
    if rms <= 0.0:
        return 0.0
    return round(peak / rms, 3)


def calculate_zero_crossing_rate(samples: List[int]) -> float:
    """Calculates Zero-Crossing Rate (ZCR) representing high-frequency spectral components."""
    if len(samples) < 2:
        return 0.0
    crossings = 0
    for i in range(1, len(samples)):
        if (samples[i] >= 0 and samples[i - 1] < 0) or (samples[i] < 0 and samples[i - 1] >= 0):
            crossings += 1
    return round(crossings / (len(samples) - 1), 5)


def calculate_spectral_centroid_sim(samples1: List[int], samples2: List[int]) -> float:
    """Calculates approximate spectral energy similarity ratio between two audio samples."""
    energy1 = sum(s * s for s in samples1) / max(len(samples1), 1)
    energy2 = sum(s * s for s in samples2) / max(len(samples2), 1)

    if energy1 == 0 and energy2 == 0:
        return 100.0
    if energy1 == 0 or energy2 == 0:
        return 0.0

    ratio = min(energy1, energy2) / max(energy1, energy2)
    return round(ratio * 100, 2)


def profile_audio(samples: List[int], sample_rate: int) -> Dict[str, Any]:
    """Computes comprehensive acoustic profile for audio sample."""
    duration = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    rms = calculate_rms_energy(samples)
    peak = calculate_peak_amplitude(samples)
    dbfs = calculate_dbfs(rms)
    crest = calculate_crest_factor(peak, rms)
    zcr = calculate_zero_crossing_rate(samples)

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate_hz": sample_rate,
        "sample_count": len(samples),
        "rms_energy": round(rms, 2),
        "dbfs": dbfs,
        "peak_amplitude": peak,
        "crest_factor": crest,
        "zero_crossing_rate": zcr,
    }


def evaluate_pair(
    ref_path: Path,
    syn_path: Path,
    threshold: float = 80.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """Performs acoustic evaluation comparing reference voice with synthesized clone."""
    ref_path = Path(ref_path)
    syn_path = Path(syn_path)

    if verbose:
        print(f"[*] Reading Reference Audio: {ref_path.name}")
        print(f"[*] Reading Synthesized Audio: {syn_path.name}")

    s1, r1 = read_wav_samples(ref_path)
    s2, r2 = read_wav_samples(syn_path)

    ref_profile = profile_audio(s1, r1)
    syn_profile = profile_audio(s2, r2)

    dur1 = ref_profile["duration_seconds"]
    dur2 = syn_profile["duration_seconds"]
    dur_delta = round(abs(dur2 - dur1), 3)
    dur_ratio = round(min(dur1, dur2) / max(dur1, dur2) * 100, 2) if max(dur1, dur2) > 0 else 100.0

    energy_sim = calculate_spectral_centroid_sim(s1, s2)

    # Combined composite similarity score (70% energy similarity + 30% duration consistency)
    overall_score = round((0.70 * energy_sim) + (0.30 * dur_ratio), 2)

    if overall_score >= 90.0:
        quality_tier = "EXCELLENT"
    elif overall_score >= 80.0:
        quality_tier = "GOOD"
    elif overall_score >= 65.0:
        quality_tier = "ACCEPTABLE"
    else:
        quality_tier = "POOR"

    passed = overall_score >= threshold

    metrics = {
        "reference_file": ref_path.name,
        "synthesized_file": syn_path.name,
        "reference_profile": ref_profile,
        "synthesized_profile": syn_profile,
        "comparisons": {
            "duration_delta_seconds": dur_delta,
            "duration_match_percentage": dur_ratio,
            "energy_similarity_score": energy_sim,
            "overall_similarity_score": overall_score,
            "threshold_required": threshold,
            "quality_tier": quality_tier,
            "benchmark_passed": passed,
        }
    }

    print("\n" + "=" * 55)
    print("       VOICE CLONE ACOUSTIC EVALUATION REPORT       ")
    print("=" * 55)
    print(f"- Reference File:         {ref_path.name}")
    print(f"- Synthesized File:       {syn_path.name}")
    print(f"- Reference Duration:     {dur1:.2f}s (RMS: {ref_profile['rms_energy']:.1f}, {ref_profile['dbfs']} dBFS)")
    print(f"- Synthesized Duration:   {dur2:.2f}s (RMS: {syn_profile['rms_energy']:.1f}, {syn_profile['dbfs']} dBFS)")
    print(f"- Duration Delta:         {dur_delta:.2f}s (Match: {dur_ratio:.1f}%)")
    print(f"- Energy Match Score:     {energy_sim:.2f}%")
    print(f"- Composite Score:        {overall_score:.2f}% [Threshold: {threshold:.1f}%]")
    print(f"- Quality Assessment:     {quality_tier} ({'PASSED' if passed else 'FAILED'})")

    if verbose:
        print("-" * 55)
        print("DETAILED SPECTRAL & DYNAMIC BREAKDOWN:")
        print(f"  • Ref Peak: {ref_profile['peak_amplitude']} | Synth Peak: {syn_profile['peak_amplitude']}")
        print(f"  • Ref Crest Factor: {ref_profile['crest_factor']} | Synth Crest: {syn_profile['crest_factor']}")
        print(f"  • Ref ZCR: {ref_profile['zero_crossing_rate']} | Synth ZCR: {syn_profile['zero_crossing_rate']}")
    print("=" * 55 + "\n")

    return metrics


def export_metrics_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Exports acoustic evaluation report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[+] Exported JSON acoustic metrics to: {output_path}")
    return output_path


def export_metrics_csv(data: Dict[str, Any], output_path: Path) -> Path:
    """Exports acoustic evaluation summary row to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    row = {
        "reference_file": data["reference_file"],
        "synthesized_file": data["synthesized_file"],
        "ref_duration": data["reference_profile"]["duration_seconds"],
        "syn_duration": data["synthesized_profile"]["duration_seconds"],
        "ref_rms": data["reference_profile"]["rms_energy"],
        "syn_rms": data["synthesized_profile"]["rms_energy"],
        "ref_dbfs": data["reference_profile"]["dbfs"],
        "syn_dbfs": data["synthesized_profile"]["dbfs"],
        "energy_score": data["comparisons"]["energy_similarity_score"],
        "composite_score": data["comparisons"]["overall_similarity_score"],
        "quality_tier": data["comparisons"]["quality_tier"],
        "passed": data["comparisons"]["benchmark_passed"],
    }

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[+] Exported CSV acoustic metrics to: {output_path}")
    return output_path


def parse_args(args=None):
    default_thresh = float(get_env_or_default("SIMILARITY_THRESHOLD", "80.0"))
    parser = argparse.ArgumentParser(
        description="Evaluate acoustic quality & similarity between reference voice and cloned voice."
    )
    parser.add_argument("--ref", "-r", required=True, help="Reference ground truth audio (.wav)")
    parser.add_argument("--synth", "-s", required=True, help="Synthesized/cloned audio (.wav)")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=default_thresh,
        help=f"Acceptance similarity threshold percentage (default: {default_thresh})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose acoustic metrics breakdown (crest factor, ZCR, peak)",
    )
    parser.add_argument(
        "--export-json", "-j",
        default=None,
        help="Export comprehensive metrics to JSON file",
    )
    parser.add_argument(
        "--export-csv", "-c",
        default=None,
        help="Export summary metrics row to CSV file",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    metrics = evaluate_pair(
        ref_path=Path(args.ref),
        syn_path=Path(args.synth),
        threshold=args.threshold,
        verbose=args.verbose or (get_env_or_default("VERBOSE_OUTPUT", "false").lower() == "true"),
    )

    if args.export_json:
        export_metrics_json(metrics, Path(args.export_json))

    if args.export_csv:
        export_metrics_csv(metrics, Path(args.export_csv))

    # Exit code based on benchmark outcome for CI pipelines
    if not metrics["comparisons"]["benchmark_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
