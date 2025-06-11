"""
Some functions for testing/demo.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn


def generate_ou_moving_mean(
    mean, trial_len=200, batch_size=1000, kappa=0.5, sigma=0.5, x0_mean=None
):
    _, dt = np.linspace(0, trial_len, trial_len, retstep=True)

    if x0_mean is None:
        X0 = np.random.normal(size=(batch_size,)) + mean[0].item()
    else:
        X0 = np.random.normal(size=(batch_size,), scale=0.2) + x0_mean

    X = np.zeros((trial_len, batch_size))
    X[0, :] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(trial_len - 1, batch_size))

    # Uncomment for Euler Maruyama
    # for t in range(0,trial_len-1):
    #    X[t + 1, :] = X[t, :] + kappa*(mean[t] - X[t, :])*dt + sigma * np.sqrt(dt) * W[t, :]

    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for t in range(0, trial_len - 1):
        X[t + 1, :] = (
            mean[t].item()
            + np.exp(-kappa * dt) * (X[t, :] - mean[t].item())
            + std_dt * W[t, :]
        )

    X = torch.tensor(X)
    X = X.permute(1, 0).unsqueeze(-1)

    return X


def generate_ou_sinusoidal_moving_mean(
    trial_len=200,
    batch_size=1000,
    kappa=0.05,
    # Measured in time steps
    wavelength=40,
    # Shift the phase so all channels aren't the same mean
    # Measured in time steps
    phase_shift=0,
    sigma=0.5,
    x0_mean=None,
):
    lspace = torch.linspace(0, trial_len, trial_len)
    mean = torch.sin(lspace * 2 * math.pi / wavelength + phase_shift)

    # Cool; now OU on top of that:
    X = generate_ou_moving_mean(
        mean,
        trial_len=trial_len,
        batch_size=batch_size,
        kappa=kappa,
        sigma=sigma,
        x0_mean=x0_mean,
    )

    return X


def bin_state_percentiles(y, yhat, ch=30, runway_length=20, bin_count=5):
    initial_states = y[:, runway_length, ch]
    recs = [(initial_states[idx], idx) for idx in range(initial_states.shape[0])]
    recs.sort()
    rec_count = len(recs)
    bin_size = int(rec_count / bin_count)

    idxs = []
    counts = []
    for ii in range(bin_count):
        rstart = ii * bin_size
        rend = rstart + bin_size

        cidxs = [r[1] for r in recs[rstart:rend]]

        idxs.append(cidxs)
        counts.append(len(cidxs))

    means = []
    meanshat = []
    minval = 1e99
    maxval = -1e99
    for cidxs in idxs:
        m = torch.mean(y[cidxs, :, ch], axis=0).detach().cpu().numpy()
        means.append(m)
        minval = min(minval, min(m))
        maxval = max(maxval, max(m))

        if yhat is not None:
            mhat = torch.mean(yhat[cidxs, :, ch], axis=0).detach().cpu().numpy()
            meanshat.append(mhat)
            minval = min(minval, min(m))
            maxval = max(maxval, max(m))

    return means, meanshat, minval, maxval


def graph_state_dependency(
    y,
    yhat,
    ch=0,
    runway_length=20,
    bin_count=5,
    title="Train",
    colormap=None,
    colormap_offset=0.0,
    savepath=None,
):
    means, meanshat, minval, maxval = bin_state_percentiles(
        y, yhat, ch=ch, runway_length=runway_length, bin_count=bin_count
    )

    if colormap:
        colors = getattr(plt.cm, colormap)(
            np.linspace(colormap_offset, 1, bin_count + 1)
        )
    else:
        colors = plt.cm.Paired(np.linspace(colormap_offset, 1, bin_count + 1))

    linewidth = 6.0
    plt.figure(figsize=(12, 8))
    for idx in range(bin_count):
        m = means[idx]
        color = colors[idx]
        plt.plot(
            m,
            color=color,
            linewidth=linewidth,
        )

        if yhat is not None:
            mhat = meanshat[idx]
            plt.plot(
                range(runway_length, mhat.shape[0] + runway_length),
                mhat,
                "--",
                color=color,
                linewidth=6.0,
            )

    plt.plot(
        [runway_length, runway_length],
        [1.1 * minval, 1.1 * maxval],
        "k--",
        linewidth=linewidth,
    )

    plt.title(title, fontsize=36)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Time steps", fontsize=32)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)


class WaveDataGenerator:
    """
    Generates a synthetic dataset emulating traveling waves near cortical surface which are
    driven by lower layer (e.g. Layer 5) activity.
    The waves are parameterized by their center position, frequency, speed, and amplitude.
    Those parameters are determined in a nonlinear way from deep cell activity.
    """
    def __init__(
        self,
        grid_size=(8, 8),        # Surface electrode grid dimensions
        n_depth=6,               # Number of depth electrodes
        time_steps=100,          # Length of each trial
        hidden_dim=8,           # Hidden dimension for MLP
        device="cpu",
        delay=20,
        ou_kappa=0.5,  # OU process decay rate
        ou_sigma=0.05,  # OU process noise level
    ):
        self.grid_size = grid_size
        self.n_depth = n_depth
        self.time_steps = time_steps
        self.device = device
        self.delay = delay
        self.ou_kappa = ou_kappa
        self.ou_sigma = ou_sigma
        
        # Total number of electrodes (surface + depth)
        self.n_total = grid_size[0] * grid_size[1] + n_depth
        
        # MLP to map depth electrode activity to wave parameters
        self.param_net = nn.Sequential(
            nn.Linear(n_depth, 7, bias=False),
            nn.Sigmoid()  # Bound parameters between 0 and 1
        ).to(device)
        nn.init.xavier_normal_(self.param_net[0].weight, gain=5.0)
        
    def generate_depth_activity(self, batch_size):
        """Generate depth electrode activity using rotational dynamics
        
        Creates a linear dynamical system with rotational dynamics in a 
        low-dimensional latent space (3D), then projects to the depth electrodes.
        """
        with torch.no_grad():
            # Create rotation matrix for 3D latent space
            theta1 = 0.1  # Rotation speed in first plane
        
            # State transition matrix for latent dynamics
            A = np.array([
                [np.cos(theta1), -np.sin(theta1), 0],
                [np.sin(theta1), np.cos(theta1), 0],
                [0, 0, np.exp(-0.01)]  # Slight decay in third dimension
            ])
        
            # Project from 3D latent to n_depth electrodes
            C = torch.randn(self.n_depth, 3) / np.sqrt(3)
            C = C.to(self.device)
        
            # Initial conditions in latent space
            z0 = torch.randn(batch_size, 3).to(self.device)
        
            # Generate trajectories
            z = torch.zeros(batch_size, self.time_steps, 3).to(self.device)
            z[:, 0, :] = z0
        
            # Convert A to torch tensor
            A = torch.tensor(A, dtype=torch.float32).to(self.device)
        
            # Simulate dynamics
            for t in range(1, self.time_steps):
                z[:, t, :] = z[:, t-1, :] @ A.T
        
            # Project to electrode space
            depth_activity = z @ C.T
        
        return depth_activity
    
    def generate_wave_params(self, depth_activity):
        with torch.no_grad():
            """Convert depth activity to wave parameters"""
            depth_mean = depth_activity[:, :5, :].mean(dim=1)
            params = self.param_net(depth_mean)

            speed_mod = 2.5

            source_x = params[:, 0] * self.grid_size[0]     # Source X coordinate
            source_y = params[:, 1] * self.grid_size[1]     # Source Y coordinate
            direction = params[:, 2] * np.pi * 2            # Wave propagation direction
            radius = params[:, 3] * self.grid_size[0] * 2.5 + 2.5   # Larger radius
            frequency = (params[:, 4] * 0.002 + 0.002) * speed_mod         # 0.02 to 0.07 Hz - slower temporal frequency
            speed = (params[:, 5] * 0.9 + 0.3) * speed_mod                # 0.1 to 0.4 units/timestep - slower speed
            amplitude = params[:, 6] * 2 + 0.5

            # Print the first 10 parameters for debugging
            for i in range(10):
                print(f"Source: ({source_x[i].item():.2f}, {source_y[i].item():.2f}), "
                      f"Direction: {direction[i].item():.2f}, Radius: {radius[i].item():.2f}, "
                      f"Frequency: {frequency[i].item():.4f}, Speed: {speed[i].item():.4f}, "
                      f"Amplitude: {amplitude[i].item():.2f}")

        return source_x, source_y, direction, radius, frequency, speed, amplitude

    def generate_batch(self, batch_size):
        """Generate a batch of data"""
        with torch.no_grad():
            # Generate depth activity and get parameters
            depth_activity = self.generate_depth_activity(batch_size)
            params = self.generate_wave_params(depth_activity)
            source_x, source_y, direction, radius, frequency, speed, amplitude = params
            
            # Create grid coordinates
            x = torch.linspace(0, self.grid_size[0]-1, self.grid_size[0])
            y = torch.linspace(0, self.grid_size[1]-1, self.grid_size[1])
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            grid_x = grid_x.to(self.device)
            grid_y = grid_y.to(self.device)
            
            # Generate surface activity
            t = torch.arange(self.time_steps, dtype=torch.float32).to(self.device)
            surface_activity = torch.zeros(batch_size, self.time_steps, 
                                         self.grid_size[0] * self.grid_size[1]).to(self.device)
    
            dt = 1.0
            std_dt = torch.sqrt(torch.tensor(self.ou_sigma**2 / (2 * self.ou_kappa) * 
                                           (1 - torch.exp(torch.tensor(-2 * self.ou_kappa * dt)))))
     
            for b in range(batch_size):
                # Calculate distances and angles from source
                dx = grid_x - source_x[b]
                dy = grid_y - source_y[b]
                distances = torch.sqrt(dx**2 + dy**2)
                
                # Project distances onto wave direction
                proj_distances = (dx * torch.cos(direction[b]) + 
                                dy * torch.sin(direction[b]))
                
                # Generate wave pattern with space-time coupling
                wave_mean = torch.zeros((self.time_steps,) + distances.shape).to(self.device)
                
                # Time-space coupled wave
                for t_idx, t_val in enumerate(t[self.delay:], self.delay):
                    # Position = initial position - speed * time
                    wave_phase = (2 * np.pi * frequency[b] * t_val - 
                                proj_distances / speed[b])
                    
                    # Attenuate with distance from source
                    attenuation = torch.exp(-distances / radius[b])
                    
                    wave_mean[t_idx] = amplitude[b] * attenuation * torch.sin(wave_phase)
        
                # Generate OU process with time-varying mean
                ou_process = torch.zeros_like(wave_mean)
                ou_process[0] = torch.randn_like(distances) * std_dt
                
                for tidx in range(1, self.time_steps):
                    ou_process[tidx] = ou_process[tidx-1] + \
                                   self.ou_kappa * (wave_mean[tidx] - ou_process[tidx-1]) * dt + \
                                   std_dt * torch.randn_like(distances)
                
                # Reshape to (time, n_surface_electrodes)
                surface_activity[b] = ou_process.reshape(self.time_steps, -1)
            
            # Combine surface and depth activity
            full_activity = torch.cat([surface_activity, depth_activity], dim=2)
        
        return full_activity, params 
    
def plot_grid_activity(data, batch_idx=0, grid_size=(8, 8), timesteps=(30, 90, 160),
                       figsize=(15, 5)):
    """Plot the surface electrode grid activity as a series of heatmaps.
    
    Args:
        data: tensor of shape (batch_size, time_steps, n_total)
        batch_idx: which batch element to plot
        timesteps: list of timepoints to plot, if None plots [0, delay-1, delay, -1]
        figsize: figure size
    """
    # Get only surface activity (exclude depth electrodes)
    surface_data = data[batch_idx, :, :grid_size[0]*grid_size[1]]
    
    # Create figure
    fig, axes = plt.subplots(1, len(timesteps), figsize=figsize)
    if len(timesteps) == 1:
        axes = [axes]
    
    # Common colorbar range
    vmin = surface_data.min()
    vmax = surface_data.max()
    
    for ax, t in zip(axes, timesteps):
        # Reshape to 2D grid
        grid = surface_data[t].reshape(grid_size)
        
        # Plot
        im = ax.imshow(grid.cpu().numpy(), 
                      origin='lower',
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax)
        ax.set_title(f't = {t}')
        
    # Add colorbar
    return fig
