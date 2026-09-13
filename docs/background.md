# Background

## Why phase can't be measured directly

There is no sensor fast enough to track the rapid oscillation of light directly.
Instead, each pixel averages (integrates) the signal over the exposure time, and
this averaging process mathematically cancels out the phase term, leaving only
the intensity — the squared amplitude of the field. As a result, the phase
information is lost at the moment of measurement.

Amplitude and phase carry complementary information: amplitude reflects how much
light is absorbed or scattered by the tissue, while phase encodes the local optical
path length — effectively the tissue's thickness and refractive index structure.
This is why phase reconstruction matters for this task: it recovers structural
detail that is otherwise invisible without staining ("label-free" imaging).

## Why the hologram changes with distance

The complex field just after the sample propagates through free space before
reaching the sensor, and free-space propagation causes diffraction: the field
spreads out, similar to ripples spreading on water after a stone is dropped. The
farther the sensor is from the sample, the more the fine structure overlaps and
smears into an interference pattern that looks nothing like the original tissue.
This is exactly why generalization across distances (Problem 1-4) and training
across many distances (Problem 2-1) matter — a network that only ever saw one
diffraction pattern (one distance) has no guarantee it will handle a differently
diffracted input.

## The Angular Spectrum Method (ASM)

Taking a 2D Fourier transform of the complex field decomposes it into plane waves,
each traveling at a specific angle (higher spatial frequency = finer detail = a
more tilted plane wave). Propagating a plane wave by distance z shifts its phase
by an amount proportional to that angle and z. This is captured by the transfer
function:

H(fx, fy; z) = exp( i * (2*pi*z/lambda) * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2) )

So simulating propagation by distance z reduces to: **FFT the field -> multiply
by H(z) -> inverse FFT**. This is the physical forward model used for
self-supervised training (Problem 2): since holograms can be synthesized from a
ground-truth field at any chosen z, the network can be trained across many
distances without needing real hologram data for each one. Because FFT and
complex multiplication are differentiable, this forward model can sit inside the
training loop and gradients still flow through it normally.
