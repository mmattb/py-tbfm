"""
Myriad reusable functions go here.
"""
import torch

def zscore(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    demeaned = data - mean
    zscored = demeaned / std
    return zscored

def fft_circular_shift_per_basis(bases, shift):
    """
    Args:
        bases: Tensor of shape (batch_size, num_bases, trial_len):
        shift: Tensor of shape (batch_size, in_dim, num_bases).
    
    Returns:
        shifted_bases: Tensor of shape (batch_size, in_dim, num_bases, trial_len):
            circularly shifted vectors
    """
    _, _, trial_len = bases.shape
    in_dim = shift.shape[1]

    # (batch_size, in_dim, num_bases, trial_len)
    bases = bases.unsqueeze(1).expand(-1, in_dim, -1, -1)

    # Run the shifts through a sigmoid to map to (0, 1), then multiply
    # to map onto (0, trial_len-1).
    shift = torch.sigmoid(shift) * (trial_len - 1)

    # FFT frequency vector
    freqs = torch.fft.fftfreq(trial_len, device=bases.device)  # shape (trial_len,)

    # Forward FFT over last dim
    spectrum = torch.fft.fft(bases, dim=-1)  # shape (batch_size, in_dim, num_bases, trial_len)

    # Phase shift: shape (batch_size, in_dim, num_bases, trial_len)
    phase = torch.exp(-2j * torch.pi * freqs.view(1, 1, 1, -1) * shift[..., None])

    # Apply phase shift
    shifted_spectrum = spectrum * phase

    # Inverse FFT
    shifted = torch.fft.ifft(shifted_spectrum, dim=-1).real  # (batch_size, in_dim, num_bases, trial_len)

    return shifted

def spectrum_shift_ifft(spectrum, shift, L):
    """
    Applies a differentiable circular (wrap-around) shift to each basis using
    the Fourier shift theorem. Assumes spectrum was produced by rfft.

    Inputs:
        spectrum: Tensor of shape (B, N, H) -- complex-valued, frequency domain
                  where H = L // 2 + 1 (output of torch.fft.rfft)
        shift:    Tensor of shape (B, C, N) -- float, amount to shift each basis
        L:        int, original time-domain length (needed for irfft)

    Output:
        shifted_bases: Tensor of shape (B, C, N, L) -- real-valued, time domain
    """
    num_ch = shift.shape[1]
    # (B, C, N, H)
    spectrum = spectrum.unsqueeze(1).expand(-1, num_ch, -1, -1).permute(0, 1, 3, 2)

    B, C, N, H = spectrum.shape
    device = spectrum.device

    # Get frequencies corresponding to rfft bins
    freqs = torch.fft.rfftfreq(L, device=device)  # shape: (H,)

    # Reshape shift and freqs to broadcast
    shift = shift.unsqueeze(-1)  # (B, C, N, 1)
    freqs = freqs.view(1, 1, 1, -1)  # (1, 1, 1, H)

    # Compute complex exponential for phase shift
    phase = torch.exp(-2j * torch.pi * shift * freqs)  # (B, C, N, H)

    # Apply shift
    shifted_spectrum = spectrum * phase  # (B, C, N, H)

    # Inverse FFT to get real-valued time signal
    shifted_bases = torch.fft.irfft(shifted_spectrum, n=L, dim=-1)  # (B, C, N, L)

    return shifted_bases