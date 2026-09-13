"""Angular Spectrum Method (ASM): differentiable free-space propagation.

Physical constants match the course's optical setup (532 nm illumination,
6.5 um pixel pitch) -- these are properties of the actual instrument that
recorded the dataset, not free parameters.

The field is padded to 2x size (replicate padding) before the FFT and
cropped back afterward. Without this, the FFT treats the field as if it
wrapped around periodically (circular convolution), which leaks energy from
one edge of the image into the other -- padding gives the field room to
diffract without that artifact.
"""
import torch
import torch.nn.functional as F

WAVELENGTH_M = 532e-9   # green laser, meters
PIXEL_PITCH_M = 6.5e-6  # sensor pixel size, meters


def _transfer_function(height, width, distance_mm, wavelength, pixel_pitch, device):
    """H(fx, fy; z): phase shift applied to each spatial-frequency plane wave
    after propagating distance z. distance_mm: tensor of shape [B]."""
    distance_m = distance_mm.view(-1, 1, 1, 1).to(device) * 1e-3

    fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=pixel_pitch, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=pixel_pitch, device=device))
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Propagating angle term under the sqrt; clamped at 0 because frequencies
    # beyond 1/wavelength correspond to evanescent (non-propagating) waves --
    # physically they decay instantly, so we just zero their contribution.
    under_sqrt = 1 - (wavelength ** 2) * (fx_grid ** 2 + fy_grid ** 2)
    kz = torch.clamp(under_sqrt, min=0).sqrt() / wavelength
    kz = kz.unsqueeze(0).unsqueeze(0)  # broadcast over (batch, channel)

    return torch.exp(1j * 2 * torch.pi * distance_m * kz)


def _center_crop(x, size):
    *_, h, w = x.shape
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top: top + size, left: left + size]


def propagate(field, distance_mm, wavelength=WAVELENGTH_M, pixel_pitch=PIXEL_PITCH_M):
    """Propagate a complex field by `distance_mm` (per-sample, shape [B]).

    field: complex tensor [B, 1, H, W]. Returns a complex tensor of the same shape.
    """
    b, c, h, w = field.shape
    device = field.device

    padded = F.pad(field, (w // 2, w // 2, h // 2, h // 2), mode="replicate")
    h_z = _transfer_function(padded.shape[-2], padded.shape[-1], distance_mm,
                              wavelength, pixel_pitch, device)

    spectrum = torch.fft.fftshift(torch.fft.fft2(padded), dim=(-2, -1))
    propagated = torch.fft.ifft2(torch.fft.ifftshift(spectrum * h_z, dim=(-2, -1)))

    return _center_crop(propagated, h)


def field_to_hologram(amp, phase, distance_mm):
    """amp, phase: real tensors [B, 1, H, W]. distance_mm: tensor [B].
    Returns the simulated, per-sample min-max normalized intensity hologram --
    matching how the ground-truth holograms in the dataset were stored."""
    field = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
    propagated = propagate(field, distance_mm)

    intensity = propagated.abs() ** 2
    i_min = intensity.amin(dim=(-2, -1), keepdim=True)
    i_max = intensity.amax(dim=(-2, -1), keepdim=True)
    return (intensity - i_min) / (i_max - i_min + 1e-8)
