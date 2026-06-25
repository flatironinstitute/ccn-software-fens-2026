---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
orphan: true
---

```{code-cell} ipython3
:tags: [hide-input, render-all]

%matplotlib inline
%load_ext autoreload
%autoreload 2
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)
```

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`02_02_place_cells_signal_processing.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

# Signal processing

<div class="render-all">
    
In this series of notebooks, we will review more advanced applications of pynapple; tuning curves, signal processing, and decoding; as well as fitting GLMs to the data using NeMoS. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

This series is split into 4 notebooks:
1. Data wrangling, 1D neural tuning, and model fitting
2. (This notebook) Signal processing
3. 2D neural tuning and model fitting
4. Neural decoding

This notebook assumes you have already gone through the first notebook to explore the data. We'll reinitialize variables created in the first notebook that will be used here.

</div>

```{code-cell} ipython3
:tags: [render-all]

import workshop_utils
# imports
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy as sp
import seaborn as sns
import tqdm
import pynapple as nap

# necessary for animation
import nemos as nmo
plt.style.use(nmo.styles.plot_style)

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# code needed from first notebook
path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
forward_ep = data["forward_ep"]
position = data["position"].restrict(forward_ep)
lfp = data["eeg"][:,0]
spikes = data["units"]
speed = np.abs(position.derivative())
ex_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
ex_lfp = lfp.restrict(ex_ep)
ex_position = position.restrict(ex_ep)
```

## Part 3: Signal processing
### Getting the Wavelet Decomposition

<div class="render-all">

In this notebook, we'll use pynapple's signal processing module to analyze LFP and visualize phase precession within hippocampal place cells. We'll start by performing a wavelet decomposition on the LFP trace during the example run saved in the Tsd `ex_lfp` defined above. We can do this in pynapple using the function [`nap.compute_wavelet_transform`](https://pynapple.org/generated/pynapple.process.wavelets.html#pynapple.process.wavelets.compute_wavelet_transform). This function requires the signal input as well as a set of frequencies to use for decomposition.

:::{admonition} Background: Continuout Wavelet Transform
:class: dropdown
A [continuous wavelet transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform) decomposes a signal into a set of [wavelets](https://en.wikipedia.org/wiki/Wavelet), in this case [Morlet wavelets](https://en.wikipedia.org/wiki/Morlet_wavelet), that span both frequency and time. You can think of the wavelet transform as a cross-correlation between the signal and each wavelet, giving the similarity between the signal and various frequency components at each time point of the signal. Similar to a Fourier transform, this gives us an estimate of what frequencies are dominating a signal. Unlike the Fourier tranform, however, the wavelet transform gives us this estimate as a function of time.
:::

First we'll define our set of frequencies. We care more about lower frequencies, so we'll use the numpy function [`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html) for denser coverage at the lower end of our frequency interval.

</div>

#### 1 Define 100 log-spaced samples between 5 and 200 Hz using [`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html)

<div class="render-user">
```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = 
```
</div>

```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
```

Now we can compute the wavelet transform using [`nap.compute_wavelet_transform`](https://pynapple.org/generated/pynapple.process.wavelets.html#pynapple.process.wavelets.compute_wavelet_transform) by passing both `ex_lfp` and `freqs`. We'll also specify the optional argument `fs`, which is known to be 1250Hz from the study methods.

#### 2. Compute the wavelet transform of `ex_lfp` using `freqs` defined above.

<div class="render-all">

- Supply the known sampling rate, 1250 Hz, as the optional argument `fs` 

</div>


<div class="render-user">  
```{code-cell} ipython3
sample_rate = 1250
ex_cwt =
```
</div>

```{code-cell} ipython3
sample_rate = 1250
ex_cwt = nap.compute_wavelet_transform(ex_lfp, freqs, fs=sample_rate)
```

:::{admonition} Note
:class: tip render-all
If `fs` is not provided, it can be inferred from the time series [`rate`](https://pynapple.org/generated/pynapple.Tsd.html#id0) attribute, e.g. `ex_lfp.rate`. However, while inferred rate is close to the true sampling rate, it can introduce a small floating-point error. Therefore, it is better to supply the true sampling rate when it is known.
:::

<div class="render-all">
    
We can visualize the results by plotting a heat map of the calculated wavelet scalogram.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, height_ratios=[1.0, 0.3], sharex=True)
fig.suptitle("Wavelet Decomposition")

amp = np.abs(ex_cwt.values)
cax = axs[0].pcolormesh(ex_cwt.t, freqs, amp.T)
axs[0].set(ylabel="Frequency (Hz)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[0].minorticks_off()
fig.colorbar(cax,label="Amplitude")

p1 = axs[1].plot(ex_lfp)
axs[1].set(ylabel="LFP (a.u.)", xlabel="Time(s)")
axs[1].margins(0)
ax = axs[1].twinx()
p2 = ax.plot(ex_position, color="orange")
ax.set_ylabel("Position (cm)")
ax.legend([p1[0], p2[0]],["raw LFP","animal position"])
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-06.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-06.png)
:::
</div>

<div class="render-all">
    
You should see a high amplitude in the 6-12Hz frequency band, corresponding to theta, while the animal runs down the track, which dampens during rest.

</div>

### Computing theta phase

<div class="render-all">

To capture phase precession, we will need to compute the phase of the theta oscillation present in the LFP. Similar to our analysis of position, we only want to compute theta phase during forward runs down the track, where the theta power will be strongest.

</div>

#### 3. Restrict `lfp` to `forward_ep`.

<div class="render-user">  
```{code-cell} ipython3
lfp = 
```
</div>

```{code-cell} ipython3
lfp = lfp.restrict(forward_ep)
```

<div class="render-all">

We can extract the theta oscillation by applying a bandpass filter on the raw LFP. To do this, we use the pynapple function [`nap.apply_bandpass_filter`](https://pynapple.org/generated/pynapple.process.filtering.html#pynapple.process.filtering.apply_bandpass_filter). This function will handle splits in the data (defined by the time support) by filtering each discontinuous epoch separately.

</div>

#### 4. Using [`nap.apply_bandpass_filter`](https://pynapple.org/generated/pynapple.process.filtering.html#pynapple.process.filtering.apply_bandpass_filter), filter `lfp` for theta within a 6-12 Hz range.

<div class="render-all">

- Same as before, pass the sampling rate of 1250 Hz (`sample_rate`)

</div>


<div class="render-user">   
```{code-cell} ipython3
theta_band = 
```
</div>

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp, (6.0, 12.0), fs=sample_rate)
```

<div class="render-all">

We can visualize the output by plotting the filtered signal with the original signal.

</div>

```{code-cell} ipython3
:tags: [render-all]

ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end)
fig = plt.figure(constrained_layout=True, figsize=(10, 3))
plt.plot(lfp.restrict(ex_run_ep), label="raw")
plt.plot(theta_band.restrict(ex_run_ep), label="filtered")
plt.xlabel("Time (s)")
plt.ylabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-07.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-07.png)
:::
</div>

<div class="render-all">

From the filtered signal, we can extract the theta phase. We can do using the pynapple function [`nap.compute_hilbert_phase`](https://pynapple.org/generated/pynapple.process.signal.html#pynapple.process.signal.compute_hilbert_phase), which will give us the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform), a common method for computing the phase of a periodic signal. This function will return the phase angle wrapped to the $[0, 2\pi]$ range.

</div>

#### 5. Use [`nap.compute_hilbert_phase`](https://pynapple.org/generated/pynapple.process.signal.html#pynapple.process.signal.compute_hilbert_phase) to compute the phase of `theta_band`

<div class="render-user"> 
```{code-cell} ipython3
# compute the phase
theta_phase = 
```
</div>

```{code-cell} ipython3
theta_phase = nap.compute_hilbert_phase(theta_band)
theta_phase
```

<div class="render-all">

Let's plot the phase on top of the filtered LFP signal, zooming in on a few cycles.

</div>

```{code-cell} ipython3
:tags: [render-all]

ex_run_shorter = nap.IntervalSet(ex_run_ep.start[0], ex_run_ep.start[0]+0.5)
fig,axs = plt.subplots(2,1,figsize=(10,4), constrained_layout=True, sharex=True)#, height_ratios=[2,1])
ax = axs[0]
ax.plot(lfp.restrict(ex_run_shorter))
ax.set_ylabel("LFP (a.u.)")
ax = axs[1]
p1 = ax.plot(theta_phase.restrict(ex_run_shorter), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_shorter))
ax.set_ylabel("Filtered LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-08.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-08.png)
:::
</div>

### Visualizing phase precession within a single unit

<div class="render-all">
    
We can vizualize phase precession within a place cell by looking at how its spike times line up with theta while the animal runs through its place field. To demonstrate this, we'll use unit 177, whose place field is cenetered on the linear track. To plot spike times and theta amplitude together, we can use the pynapple method [`value_from`](https://pynapple.org/generated/pynapple.TsGroup.value_from.html), which will find the theta amplitude closest in time to each spike. 

#### 6. Use the pynapple object method [`value_from`](https://pynapple.org/generated/pynapple.TsGroup.value_from.html) to find the value of `theta_band` corresponding to each spike time from unit 177.

</div>

<div class="render-user">  
```{code-cell} ipython3
unit = 177
spike_theta = 
```
</div>

```{code-cell} ipython3
unit = 177
spike_theta = spikes[unit].value_from(theta_band)
```

<div class="render-all">

Let's plot `spike_theta` on top of the LFP and filtered theta, as well as visualize the animal's position along the track.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, sharex=True)
axs[0].plot(lfp.restrict(ex_run_ep), alpha=0.5, label="raw LFP")
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)")
axs[0].legend()

axs[1].plot(ex_position, '--', color="green", label="animal position")
axs[1].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)")
axs[1].legend()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-09.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-09.png)
:::
</div>

<div class="render-all">
    
As the animal runs through unit 177's place field (thick green), the cell spikes (orange dots) at specific points along the theta cycle dependent on position: starting at the rising edge, moving towards the trough, and ending at the falling edge.

We see this pattern more clearly by plotting the spike times aligned to the phase of theta. 

</div>

#### 7. Compute the value of `theta_phase` corresponding to each spike time from unit 177.

<div class="render-user">  
```{code-cell} ipython3
spike_phase = 
```
</div>

```{code-cell} ipython3
spike_phase = spikes[unit].value_from(theta_phase)
```

<div class="render-all">

To visualize the results, we'll recreate the plot above and include the theta phase.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(3, 1, figsize=(10,6), constrained_layout=True, sharex=True)
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)", title="Spike times relative to filtered theta")
axs[0].legend()

axs[1].plot(theta_phase.restrict(ex_run_ep), color="slateblue", label="theta phase")
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[1].set(ylabel="Phase (rad)", title="Spike times relative to theta phase")
axs[1].legend()

axs[2].plot(ex_position, '--', color="green", label="animal position")
axs[2].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[2].set(ylabel="Position (cm)", xlabel="Time (s)", title="Animal position")
axs[2].legend()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-10.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-10.png)
:::
</div>

<div class="render-all">
    
We now see a negative slope in the spike phase as the animal moves through the unit's place field. This phemomena is known as phase precession: the phase at which a unit spikes *precesses* (gets earlier) as the animal runs through that unit's place field. Explicitly, that unit will spike at *late* phases of theta (higher radians) in *earlier* positions in the field, and fire at *early* phases of theta (lower radians) in *late* positions in the field.

We can observe this phenomena on average across the session by relating the spike phase to the spike position. 

</div>

#### 8. Compute the position corresponding to each spike for example unit 177.

<div class="render-user">
```{code-cell} ipython3
spike_position = 
```
</div>

```{code-cell} ipython3
spike_position = spikes[unit].value_from(position)
```

<div class="render-all">

Now we can plot the spike phase against the spike position in a scatter plot.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(figsize=(5,3))
axs.plot(spike_position, spike_phase, 'o')
axs.set_ylabel("Phase (rad)")
axs.set_xlabel("Position (cm)")
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-11.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-11.png)
:::
</div>


<div class="render-all">
    
Similar to what we saw in a single run, there is a negative relationship between theta phase and field position, characteristic of phase precession.

</div>
