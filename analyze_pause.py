#!/usr/bin/env python3
"""
Analyze audio file to detect and measure silence/pause durations.
Usage: python3 analyze_pause.py <audio_file.wav>

Requires: scipy
Install with: pip install scipy
"""

import sys
import numpy as np

try:
    from scipy.io import wavfile
except ImportError:
    print("Error: scipy is required. Install with: pip install scipy")
    sys.exit(1)

def analyze_audio_pauses(filename, silence_threshold=0.01, min_silence_duration_ms=100):
    """
    Analyze an audio file to detect and measure pauses/silence.

    Args:
        filename: Path to WAV file
        silence_threshold: Amplitude threshold below which audio is considered silent (0.0-1.0)
        min_silence_duration_ms: Minimum duration in ms to consider as a pause

    Returns:
        List of (start_time, end_time, duration) tuples for detected pauses
    """
    # Read the WAV file using scipy (supports IEEE float format)
    sample_rate, audio_array = wavfile.read(filename)

    # Get audio info
    n_channels = 1 if audio_array.ndim == 1 else audio_array.shape[1]
    n_frames = len(audio_array)
    duration_sec = n_frames / sample_rate

    print(f"Audio file: {filename}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {n_channels}")
    print(f"Data type: {audio_array.dtype}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Total frames: {n_frames}")
    print()

    # If stereo, convert to mono by averaging channels
    if n_channels == 2:
        audio_array = audio_array.mean(axis=1)

    # Normalize to [-1.0, 1.0] range based on data type
    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0
    elif audio_array.dtype == np.uint8:
        audio_array = (audio_array.astype(np.float32) - 128) / 128.0
    elif audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_array = audio_array.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio data type: {audio_array.dtype}")

    # Take absolute value for amplitude detection
    amplitude = np.abs(audio_array)

    # Detect silence: frames where amplitude is below threshold
    is_silent = amplitude < silence_threshold

    # Find silence regions
    pauses = []
    in_silence = False
    silence_start = 0

    min_silence_frames = int(min_silence_duration_ms * sample_rate / 1000)

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            # Start of silence
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            # End of silence
            silence_duration_frames = i - silence_start
            if silence_duration_frames >= min_silence_frames:
                start_time_ms = (silence_start / sample_rate) * 1000
                end_time_ms = (i / sample_rate) * 1000
                duration_ms = end_time_ms - start_time_ms
                pauses.append((start_time_ms, end_time_ms, duration_ms))
            in_silence = False

    # Handle case where audio ends in silence
    if in_silence:
        silence_duration_frames = len(is_silent) - silence_start
        if silence_duration_frames >= min_silence_frames:
            start_time_ms = (silence_start / sample_rate) * 1000
            end_time_ms = (len(is_silent) / sample_rate) * 1000
            duration_ms = end_time_ms - start_time_ms
            pauses.append((start_time_ms, end_time_ms, duration_ms))

    return pauses, sample_rate

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pause.py <audio_file.wav>")
        print("Example: python3 analyze_pause.py test_pause.wav")
        sys.exit(1)

    filename = sys.argv[1]

    # Adjust these parameters if needed
    silence_threshold = 0.01  # Amplitude below 1% is considered silent
    min_pause_duration_ms = 100  # Only report pauses longer than 100ms

    try:
        pauses, sample_rate = analyze_audio_pauses(filename, silence_threshold, min_pause_duration_ms)

        if pauses:
            print(f"Detected {len(pauses)} pause(s) (threshold: {silence_threshold}, min duration: {min_pause_duration_ms}ms):")
            print()
            for i, (start, end, duration) in enumerate(pauses, 1):
                print(f"Pause #{i}:")
                print(f"  Start:    {start:7.0f} ms ({start/1000:.2f} sec)")
                print(f"  End:      {end:7.0f} ms ({end/1000:.2f} sec)")
                print(f"  Duration: {duration:7.0f} ms ({duration/1000:.2f} sec)")
                print()

            # Summary
            total_pause_duration = sum(d for _, _, d in pauses)
            print(f"Total pause duration: {total_pause_duration:.0f} ms ({total_pause_duration/1000:.2f} sec)")
        else:
            print("No pauses detected (try adjusting silence_threshold or min_pause_duration_ms)")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
