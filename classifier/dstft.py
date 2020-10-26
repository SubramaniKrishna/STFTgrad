"""
Code for the differentiable STFT front-end
As explained in our paper, we use a Gaussian Window STFT, with N = floor(6\sigma)
"""

# Dependencies
import jax.numpy as jnp
import jax

def diff_stft(xinp,s,hf = 0.5):
    """
    Inputs
    ------
    xinp: jnp.array
        Input audio signal in time domain
    s: jnp.float
        The standard deviation of the Gaussian window to be used
    hf: jnp.float
        The fraction of window size that will be overlapped within consecutive frames
    
    Outputs
    -------
    a: jnp.array
        The computed magnitude spectrogram
    """

    # Effective window length of Gaussian is 6\sigma
    sz = s * 6
    hp = hf*sz

    # Truncating to integers for use in jnp functions
    intsz = int(jnp.round(sz))
    inthp = int(jnp.round(hp))

    m = jnp.arange(0, intsz, dtype=jnp.float32)

    # Obtaining the "differentiable" window function by using the real valued \sigma
    window = jnp.exp(-0.5 * jnp.power((m - sz / 2) / (s + 1e-5), 2))
    window_norm = window/jnp.sum(window)
    
    # Computing the STFT, and taking its magnitude
    stft = jnp.sqrt(1/(2*window_norm.shape[0] + 1))*jnp.stack([jnp.fft.rfft(window_norm * xinp[i:i+intsz]) for i in range(0, len(xinp) - intsz, inthp)],1)
    a = jnp.abs(stft)
    
    return a

