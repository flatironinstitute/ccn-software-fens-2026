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

This notebook can be downloaded as **{nb-download}`02_04_place_cells_decoding.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

# Neural decoding

<div class="render-all">
    
In this series of notebooks, we will review more advanced applications of pynapple; tuning curves, signal processing, and decoding; as well as fitting GLMs to the data using NeMoS. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

This series is split into 4 notebooks:
1. Data wrangling, 1D neural tuning, and model fitting
2. Signal processing
3. 2D neural tuning and model fitting
4. (This notebook) Neural decoding

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
lfp = data["eeg"][:,0].restrict(forward_ep)
spikes = data["units"]
ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end)
ex_position = position.restrict(ex_run_ep)
```

## Part 5: Neural decoding
### Decoding position from spiking activity

<div class="render-all">

In this notebook we'll do a popular analysis in the rat hippocampus sphere: Bayesian decoding. This analysis is an elegent application of Bayes' rule in predicting the animal's location (or other behavioral variables) given neural activity at some point in time. Refer to the dropdown box below for a more in-depth explanation.

:::{admonition} Background: Bayesian decoding
:class: dropdown
Recall Bayes' rule, written here in terms of our relevant variables:

$$P(position|spikes) = \frac{P(position)P(spikes|position)}{P(spikes)}$$

Our goal is to compute the unknown posterior $P(position|spikes)$ given known prior $P(position)$ and known likelihood $P(spikes|position)$. 

$P(position)$, also known as the *occupancy*, is the probability that the animal is occupying some position. This can be computed exactly by the proportion of the total time spent at each position, but in many cases it is sufficient to estimate the occupancy as a uniform distribution, i.e. it is equally likely for the animal to occupy any location.

The next term, $P(spikes|position)$, which is the probability of seeing some sequence of spikes across all neurons at some position. Computing this relys on the following assumptions:
1. Neurons fire according to a Poisson process (i.e. their spiking activity follows a Poisson distribution)
2. Neurons fire independently from one another.

While neither of these assumptions are strictly true, they are generally reasonable for pyramidal cells in hippocampus and allow us to simplify our computation of $P(spikes|position)$

The first assumption gives us an equation for $P(spikes|position)$ for a single neuron, which we'll call $P(spikes_i|position)$ to differentiate it from $P(spikes|position) = P(spikes_1,spikes_2,...,spikes_N|position)$, or the total probability across all $N$ neurons. The equation we get is that of the Poisson distribution:

$$
P(spikes_i|position) = \frac{(\tau f_i(position))^n e^{-\tau f_i(position)}}{n!}
$$

where $f_i(position)$ is the firing rate of the neuron at position $(position)$ (i.e. the tuning curve), $\tau$ is the width of the time window over which we're computing the probability, and $n$ is the total number of times the neuron spiked in the time window of interest.

The second assumptions allows us to simply combine the probabilities of individual neurons. Recall the product rule for independent events: $P(A,B) = P(A)P(B)$ if $A$ and $B$ are independent. Treating neurons as independent, then, gives us the following:

$$
P(spikes|position) = \prod_i P(spikes_i|position)
$$

The final term, $P(spikes)$, is inferred indirectly using the law of total probability:

$$P(spikes) = \sum_{position}P(position,spikes) = \sum_{position}P(position)P(spikes|position)$$

Another way of putting it is $P(spikes)$ is the normalization factor such that $\sum_{position} P(position|spikes) = 1$, which is achived by dividing the numerator by its sum.

If this method looks daunting, we have some good news: pynapple has it implemented already in the function `nap.decode_bayes`. All we'll need are the spikes, the tuning curves, and the width of the time window $\tau$.
:::

(phase-precess-cv-full)=
:::{admonition} Aside: Cross-validation
:class: tip 
    
Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves. 

If we want to decode an example run down the track, our training set should omit this run before computing the tuning curves. We can do this by using the IntervalSet method `set_diff`, to take out the example run epoch from all run epochs. Next, we'll restrict our data to these training epochs and re-compute the place fields using `nap.compute_tuning_curves`. We'll also apply a Gaussian smoothing filter to the place fields, which will smooth our decoding results down the line.
:::

The code below will redefine the position tuning curves that we originally computed in the first notebook with two changes: 1) they leave out the period we will decode (see above), and 2) they're computed over *all* units, which can help with decoder accuracy.

</div>

```{code-cell} ipython3
:tags: [render-all]

from scipy.ndimage import gaussian_filter1d
# hold out trial from place field computation
run_train = forward_ep.set_diff(ex_run_ep)
# get position of training set
position_train = position.restrict(run_train)
# compute place fields using training set
place_fields = nap.compute_tuning_curves(spikes, position_train, bins=50, feature_names=["position"])
# smooth place fields
place_fields.data = gaussian_filter1d(place_fields.data, 1, axis=-1)

# plot sorted, normalized tuning curves
idx = place_fields.argmax(axis=1)
place_fields_sorted = place_fields.sortby(idx)
place_fields_sorted["unit"] = np.arange(place_fields_sorted.shape[0])
p = (place_fields_sorted / place_fields_sorted.max(axis=1)).plot()
```

<div class="render-all">

We can decode any number of features using the function [`nap.decode_bayes`](https://pynapple.org/generated/pynapple.process.decoding.html#pynapple.process.decoding.decode_bayes), which decodes any feature used in the input `tuning_curves` computed by [`nap.compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves).

#### 1. Use [`nap.decode_bayes`](https://pynapple.org/generated/pynapple.process.decoding.html#pynapple.process.decoding.decode_bayes) to decode position during `ex_run_ep` using all units in `spikes`.

- Use 40 ms time bins
- Note: `ex_run_ep` was originally computed in the first notebook and redefined above.

</div>

<div class="render-user">
```{code-cell} ipython3
decoded_position, decoded_prob = 
```
</div>

```{code-cell} ipython3
decoded_position, decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, 0.04)
```

<div class="render-all">

Let's plot decoded position with the animal's true position. We'll overlay them on a heat map of the decoded probability to visualize the confidence of the decoder.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(decoded_position.index,place_fields.position,np.transpose(decoded_prob))
ax.plot(decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-15.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-15.png)
:::
</div>


<div class="render-all">
    
While the decoder generally follows the animal's true position, there is still a lot of error in the decoder, especially later in the run. We can improve the decoder error by smoothing the spike counts. [`nap.decode_bayes`](https://pynapple.org/generated/pynapple.process.decoding.html#pynapple.process.decoding.decode_bayes) provides the option to do this for you by specifying `sliding_window_size`, which specifies the width, in number of bins, of a uniform (all ones) kernel to convolve with the spike counts. This is equivalent to applying a moving sum to adjacent bins, where the width of the kernel is the number of adjacent bins being added together. This can also be referred to as using a *sliding window* to count spikes that shifts in shorter increments than the window's width, which combines the accuracy of using a wider time bin with the temporal resolution of a shorter time bin.

For example, let's say we want a sliding window of $200 ms$ that shifts by $40 ms$. This is equivalent to summing together 5 adjacent $40 ms$ bins, or convolving spike counts in $40 ms$ bins with a length-5 array of ones ($[1, 1, 1, 1, 1]$). Let's visualize this convolution.

</div>

```{code-cell} ipython3
:tags: [render-all]

unit = 177
ex_counts = spikes[unit].restrict(ex_run_ep).count(0.04)
workshop_utils.animate_1d_convolution(ex_counts, np.ones(5), tsd_label="original counts", kernel_label="moving sum", conv_label="convolved counts")
```

<div class="render-all">
    
The count at each time point is computed by convolving the kernel (yellow), centered at that time point, with the original spike counts (blue). For a length-5 kernel of ones, this amounts to summing the counts in the center bin with two bins before and two bins after (shaded green, top). The result is an array of counts smoothed out in time (green, bottom).

</div>

#### 2. Decode the same run as above, now using sliding window size of 5 bins.

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, bin_size=0.04, sliding_window_size=5)
```

<div class="render-all">

Let's plot the results.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(smth_decoded_position.index,place_fields.position,np.transpose(smth_decoded_prob))
ax.plot(smth_decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-16.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-16.png)
:::
</div>

<div class="render-all">
    
This gives us a much closer approximation of the animal's true position.

Units phase precessing together creates fast, spatial sequences around the animal's true position (see notebook 2 for a deeper dive into phase precession). We can demonstrate this by decoding at an even shorter time scale, which will appear as smooth errors in the decoder.

</div>

#### 3. Decode again using a smaller bin size of $10 ms$ and sliding window size of 5 bins.

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, bin_size=0.01, sliding_window_size=5)
```

<div class="render-all">
    
We'll make the same plot as before to visualize the results, but plot it alongside the raw and filtered LFP.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True, height_ratios=[3,1], sharex=True)
c = axs[0].pcolormesh(smth_decoded_prob.index, smth_decoded_prob.columns, np.transpose(smth_decoded_prob))
p1 = axs[0].plot(smth_decoded_position, "--", color="r")
p2 = axs[0].plot(ex_position, color="r")
axs[0].set_ylabel("Position (cm)")
axs[0].legend([p1[0],p2[0]],["decoded position","true position"])
fig.colorbar(c, label = "predicted probability")

sample_rate = 1250
theta_band = nap.apply_bandpass_filter(lfp, (6.0, 12.0), fs=sample_rate)
axs[1].plot(lfp.restrict(ex_run_ep))
axs[1].plot(theta_band.restrict(ex_run_ep))
axs[1].set_ylabel("LFP (a.u.)")

fig.supxlabel("Time (s)");
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/pc-17.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/pc-17.png)
:::
</div>

<div class="render-all">
    
The estimated position oscillates with cycles of theta, where each "sweep" is referred to as a "theta sequence". Fully understanding the properties of theta sequences and their role in learning, memory, and planning is an active topic of research in Neuroscience!

</div>

### Bonus Exercise 1

<div class="render-all">
    
Pynapple has another decoding method, [`nap.decode_template`](https://pynapple.org/generated/pynapple.process.decoding.html#pynapple.process.decoding.decode_template), that is agnostic to the underlying noise model of the data. In other words, where the above implementation of Bayesian decoding is specific to spiking data (Poisson distributed data), template decoding can be applied to any data modality. As a bonus exercise, you can try decoding position using this method and compare the results to the Bayesian decoder used above!

</div>

```{code-cell} ipython3
# template decoding
```

### Bonus Exercise 2

<div class="render-all">
    
Instead of using place fields computed from the data, what if we used the predicted tuning curves by our GLM in Part 2 to do decoding? As a second bonus exercise, you can try Bayesian decoding using GLM-predicted tuning curves and compare the results to the decoding above.

</div>

```{code-cell} ipython3
# GLM decoding
```
