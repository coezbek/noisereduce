\
import argparse
import numpy as np
from scipy.io import wavfile
from noisereduce.noisereduce import reduce_noise

def main():
    parser = argparse.ArgumentParser(description="Reduce noise from an audio file using spectral gating.")
    parser.add_argument("input_file", help="Path to the input WAV file.")
    parser.add_argument("output_file", help="Path to save the processed WAV file.")

    parser.add_argument("--stationary", action="store_true", help="Whether to perform stationary or non-stationary noise reduction.")
    parser.add_argument("--noise-file", help="Path to a WAV file containing example noise (for stationary reduction).", default=None, dest="noise_file")
    parser.add_argument("--prop-decrease", type=float, default=1.0, help="Proportion to reduce the noise by (0.0 to 1.0).", dest="prop_decrease")
    parser.add_argument("--time-constant-s", type=float, default=2.0, help="Time constant in seconds for non-stationary algorithm.", dest="time_constant_s")
    parser.add_argument("--freq-mask-smooth-hz", type=int, default=500, help="Frequency range (Hz) to smooth the mask over.", dest="freq_mask_smooth_hz")
    parser.add_argument("--time-mask-smooth-ms", type=int, default=50, help="Time range (ms) to smooth the mask over.", dest="time_mask_smooth_ms")
    parser.add_argument("--thresh-n-mult-nonstationary", type=float, default=3.0, help="Multiplier for threshold in non-stationary mode.", dest="thresh_n_mult_nonstationary") # Docstring says int, default 1. Code default is 2. README says int.
    parser.add_argument("--sigmoid-slope-nonstationary", type=float, default=10.0, help="Slope for sigmoid in non-stationary mode.", dest="sigmoid_slope_nonstationary") # Docstring says int. README says int.
    parser.add_argument("--n-std-thresh-stationary", type=float, default=1.5, help="Number of standard deviations above mean for stationary threshold.", dest="n_std_thresh_stationary")
    parser.add_argument("--tmp-folder", help="Temporary folder for parallel processing.", default=None, dest="tmp_folder")
    parser.add_argument("--chunk-size", type=int, default=600000, help="Size of signal chunks for processing.", dest="chunk_size")
    parser.add_argument("--padding", type=int, default=30000, help="Padding for signal chunks.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size.", dest="n_fft")
    parser.add_argument("--win-length", type=int, default=None, help="Window length for STFT. Defaults to n_fft.", dest="win_length")
    parser.add_argument("--hop-length", type=int, default=None, help="Hop length for STFT. Defaults to win_length // 4.", dest="hop_length")
    parser.add_argument("--clip-noise-stationary", action="store_true", default=True, help="Clip noise to signal length in stationary mode.", dest="clip_noise_stationary")
    parser.add_argument("--no-clip-noise-stationary", action="store_false", dest="clip_noise_stationary", help="Do not clip noise to signal length in stationary mode.")
    parser.add_argument("--no-progress", action="store_true", help="Don't show a progress bar.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel jobs. Set to -1 to use all CPU cores (not for torch). Default 4.", dest="n_jobs")
    parser.add_argument("--use-torch", action="store_true", help="Use PyTorch version of spectral gating.", dest="use_torch")
    parser.add_argument("--device", default="cuda", help="Device for PyTorch version (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    # Read input WAV file
    try:
        sr, y = wavfile.read(args.input_file)
    except Exception as e:
        print(f"Error reading input file {args.input_file}: {e}")
        return
    
    # Warn if not mono
    if len(y.shape) > 1 and y.shape[1] != 1:
        print(f"Warning: Input file {args.input_file} is not mono. Only the first channel will be used.")
        y = y[:, 0]  # Use only the first channel

    # Ensure y is float
    # if y.dtype != np.float32 and y.dtype != np.float64:
    #     print(f"Warning: Input data type {y.dtype} is not float32 or float64. Converting to float32.")
    #     if np.issubdtype(y.dtype, np.integer):
    #         # Scale integer types to [-1.0, 1.0]
    #         max_val = np.iinfo(y.dtype).max
    #         y = y.astype(np.float32) / max_val
    #     else:
    #         # For other types, just convert
    #         y = y.astype(np.float32)
# 

    y_noise = None
    if args.noise_file:
        try:
            sr_noise, y_noise_data = wavfile.read(args.noise_file)
            if sr_noise != sr:
                print(f"Warning: Sample rate of noise file ({sr_noise} Hz) does not match input signal ({sr} Hz). Resampling not implemented.")
                # Potentially add resampling here if desired, or raise error
            
            if y_noise_data.dtype != np.float32 and y_noise_data.dtype != np.float64:
                if np.issubdtype(y_noise_data.dtype, np.integer):
                    max_val_noise = np.iinfo(y_noise_data.dtype).max
                    y_noise = y_noise_data.astype(np.float32) / max_val_noise
                else:
                    y_noise = y_noise_data.astype(np.float32)

        except Exception as e:
            print(f"Error reading noise file {args.noise_file}: {e}")
            # Decide if to proceed without y_noise or exit
            # For now, proceed without y_noise if file read fails
            y_noise = None


    # Call reduce_noise function
    reduced_y = reduce_noise(
        y=y,
        sr=sr,
        stationary=args.stationary,
        y_noise=y_noise,
        prop_decrease=args.prop_decrease,
        time_constant_s=args.time_constant_s,
        freq_mask_smooth_hz=args.freq_mask_smooth_hz,
        time_mask_smooth_ms=args.time_mask_smooth_ms,
        thresh_n_mult_nonstationary=args.thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary=args.sigmoid_slope_nonstationary,
        n_std_thresh_stationary=args.n_std_thresh_stationary,
        tmp_folder=args.tmp_folder,
        chunk_size=args.chunk_size,
        padding=args.padding,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        clip_noise_stationary=args.clip_noise_stationary,
        use_tqdm=not args.no_progress,
        n_jobs=args.n_jobs,
        use_torch=args.use_torch,
        device=args.device,
    )

    # Write output WAV file
    # Ensure data is in a writable format for wavfile (e.g., int16 or float32)
    # If original was int, scale back. For simplicity, always write float32 for now
    # or scale to int16 if that's preferred for output.
    # For now, assume reduced_y is float and wavfile can handle it.
    # If it was scaled from int, it's now in [-1, 1] range.
    # To save as int16, multiply by 32767 and convert type.
    
    try:
        # If the output should be int16, scale and convert
        # For now, let's try to save as float32 if possible, or scale to int16
        if reduced_y.dtype == np.float32 or reduced_y.dtype == np.float64:
             # If values are in [-1, 1], scale to int16 for broader compatibility
            if np.max(np.abs(reduced_y)) <= 1.0:
                output_y = (reduced_y * 32767).astype(np.int16)
            else: # if it's already scaled e.g. from librosa load
                output_y = reduced_y.astype(np.int16) # Or handle as error/warning
        elif np.issubdtype(reduced_y.dtype, np.integer):
            output_y = reduced_y # Already integer
        else:
            print(f"Warning: Output data type {reduced_y.dtype} may not be directly writable. Attempting conversion to int16.")
            output_y = (reduced_y * 32767).astype(np.int16)

        wavfile.write(args.output_file, sr, output_y)
        print(f"Processed audio saved to {args.output_file}")
    except Exception as e:
        print(f"Error writing output file {args.output_file}: {e}")

if __name__ == "__main__":
    main()
