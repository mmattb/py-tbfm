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