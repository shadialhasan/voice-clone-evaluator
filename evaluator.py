# -*- coding: utf-8 -*-
"""
Voice Clone Quality & Similarity Evaluator
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
Phone: +963934005922
Copyright (c) 2026 MHD. Shadi AL-Hasan
"""

import math
import wave
import struct
import argparse
from pathlib import Path

def read_wav_samples(path: Path):
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
        samples = struct.unpack(f"<{n_frames * channels}h", raw)
        if channels == 2:
            samples = [samples[i] for i in range(0, len(samples), 2)]
        return samples, wf.getframerate()

def calculate_spectral_centroid_sim(samples1, samples2):
    """Calculates approximate spectral energy similarity ratio between two audio samples."""
    energy1 = sum(s*s for s in samples1) / max(len(samples1), 1)
    energy2 = sum(s*s for s in samples2) / max(len(samples2), 1)
    if energy1 == 0 or energy2 == 0:
        return 0.0
    ratio = min(energy1, energy2) / max(energy1, energy2)
    return round(ratio * 100, 2)

def evaluate_pair(ref_path: Path, syn_path: Path):
    print(f"[*] Evaluating Reference: {ref_path.name}")
    print(f"[*] Evaluating Synthesized: {syn_path.name}")
    
    s1, r1 = read_wav_samples(ref_path)
    s2, r2 = read_wav_samples(syn_path)

    dur1 = len(s1) / r1
    dur2 = len(s2) / r2
    sim_score = calculate_spectral_centroid_sim(s1, s2)

    print("\n" + "=" * 40)
    print("VOICE EVALUATION METRICS:")
    print(f"- Reference Duration:   {dur1:.2f}s")
    print(f"- Synthesized Duration: {dur2:.2f}s")
    print(f"- Energy Match Score:   {sim_score}%")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio similarity between reference voice and cloned voice.")
    parser.add_argument("--ref", required=True, help="Reference ground truth audio (.wav)")
    parser.add_argument("--synth", required=True, help="Synthesized/cloned audio (.wav)")
    args = parser.parse_args()

    evaluate_pair(Path(args.ref), Path(args.synth))

if __name__ == "__main__":
    main()
