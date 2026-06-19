---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
no-search: true
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

This notebook can be downloaded as **{nb-download}`part1_extracellular.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

# Head-direction cells: Part 1 — Extracellular recordings

<div class="render-all">

This is **Part 1** of a two-part tutorial comparing two recording modalities — extracellular electrophysiology and calcium imaging — using the same head-direction system in the mouse as a common reference. Both datasets contain head-direction cells, but the signal properties differ: spikes are discrete and fast, while calcium transients are continuous and slow. This gives us a natural testbed to see how the same analysis workflow adapts to different data types.

This part uses spike trains from the anterodorsal thalamic nucleus (ADn) recorded with a silicon probe ([Peyrache et al., 2015](https://www.nature.com/articles/nn.3968)).

With **pynapple** we will:
1. Load a NWB file and extract spike times and head-direction
2. Compute head-direction tuning curves
3. Compute cross-correlograms during wake and sleep

With **nemos** we will fit a population GLM to characterize functional connectivity:
1. Build spike-history features with a raw history window and with a `RaisedCosineLogConv` basis
2. Fit a single-neuron GLM and compare the two feature representations
3. Fit a `PopulationGLM` to all neurons simultaneously and visualize the coupling filters

**Part 2 — Calcium imaging** uses deconvolved fluorescence traces from head-direction cells in the postsubiculum, and continues in a separate notebook.

The pynapple documentation can be found [here](https://pynapple.org) and the nemos documentation [here](https://nemos.readthedocs.io/en/latest/).

Let's start by importing all the packages.

</div>

```{code-cell} ipython3
:tags: [render-all]

import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import jax
import workshop_utils
from nemos import _documentation_utils as doc_plots

# LBFGS works better with float64 precision
jax.config.update("jax_enable_x64", True)

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure some plots
plt.style.use(nmo.styles.plot_style)
```

## Analyzing head-direction cells with Pynapple

## Fetch and load data

<div class="render-all">

The dataset comes from [Peyrache et al., 2015](https://www.nature.com/articles/nn.3968). If you ran the workshop setup script the file is already on disk; otherwise `fetch_data` from `workshop_utils` will download it and return the local path.

</div>

```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Mouse32-140822.nwb")

print(path)
```

<div class="render-all">

Pynapple provides the convenience function `nap.load_file` for loading a NWB file.

**Question:** Can you open the NWB file by passing the variable `path` to the function `load_file` and call the output `data`?

</div>

<div class="render-user">
```{code-cell} ipython3
data =
print(data)
```
</div>

```{code-cell} ipython3
data = nap.load_file(path)

print(data)
```

<div class="render-all">

The content of the NWB file is not loaded yet. The object `data` behaves like a dictionary.
It contains multiple entries corresponding to different data types stored in the NWB file.
In NWB files, spike times are stored in the `units` entry.

**Question:** Can you load the spike times from the NWB and call the variable `spikes`?

</div>

<div class="render-user">
```{code-cell} ipython3
spikes =   # Get spike timings
print(spikes)
```
</div>

```{code-cell} ipython3
spikes = data["units"]  # Get spike timings
print(spikes)
```

<div class="render-all">

The recording contains neurons from several brain areas. We want only those labeled `adn` (anterodorsal nucleus) with a firing rate above 2 Hz.

**Question:** Can you filter `spikes` to keep only ADn neurons above 2 Hz?

Metadata can be accessed as an attribute or like a dictionary key; you can also use the helper methods below:

1. `spikes.location` or `spikes['location']` — returns a pandas Series of brain-area labels.
2. [`spikes.getby_category`](https://pynapple.org/generated/pynapple.TsGroup.getby_category.html#pynapple.TsGroup.getby_category) — filter by a categorical metadata field.
3. [`spikes.getby_threshold`](https://pynapple.org/generated/pynapple.TsGroup.getby_threshold.html#pynapple.TsGroup.getby_threshold) — filter by a numeric threshold.

</div>

<div class="render-user">
```{code-cell} ipython3
spikes =   # Select only ADN neurons with rate > 2.0 Hz
print(len(spikes))
```
</div>

```{code-cell} ipython3
spikes = spikes[(spikes.location=='adn') & (spikes.rate>2.0)]

print(len(spikes))
```

<div class="render-all">

The NWB file contains other information about the recording. `ry` contains the values of the head-direction of the animal over time. 

**Question:** Can you extract the angle of the animal in a variable called `angle` and print it?

</div>

<div class="render-user">
```{code-cell} ipython3
angle =   # Get head-direction data from NWB object
print(angle)
```
</div>

```{code-cell} ipython3
angle = data["ry"]
print(angle)
```

<div class="render-all">

The data are not fully loaded into memory yet. While `angle` is a `Tsd` object, its underlying array (accessible via `.d`) is still an `h5py` dataset — pynapple lazy-loads NWB data by default, which is efficient for large recordings that would not fit in RAM.

</div>

```{code-cell} ipython3
print(angle.d)
```

<div class="render-all">

The animal was recorded during wakefulness and sleep. 

**Question:** Can you extract the behavioral intervals in a variable called `epochs`?

</div>

<div class="render-user">
```{code-cell} ipython3
epochs =   # Get behavioral epochs from NWB object
print(epochs)
```
</div>

```{code-cell} ipython3
epochs = data["epochs"]

print(epochs)
```

<div class="render-all">

NWB files can store intervals with multiple labels. The `IntervalSet` object exposes those labels as metadata.

**Question:** Using the `tags` column, can you split `epochs` into two `IntervalSet` objects — one for `wake` and one for `sleep`?

</div>

<div class="render-user">
```{code-cell} ipython3
wake_ep =  # Get wake intervals from epochs
sleep_ep =  # Get sleep intervals from epochs
```
</div>

```{code-cell} ipython3
wake_ep = epochs[epochs.tags=="wake"]
sleep_ep = epochs[epochs.tags=="sleep"]
```

## Compute tuning curves

<div class="render-all">

We now have spikes, a behavioral feature (head-direction), and the epochs during which that feature was recorded. We can compute **tuning curves** — the mean firing rate of each neuron as a function of head-direction during wakefulness — with a single pynapple call.

**Question:** Can you compute the head-direction tuning curves of the ADn units using `nap.compute_tuning_curves` and store the result in `tuning_curves`?

</div>

<div class="render-user">
```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(
    data=, # The neural activity as a TsGroup
    features=, # Which feature? Here the head-direction of the animal
    bins=, # How many bins of feature space? Here 61 angular bins is a good numbers
    epochs = angle.time_support, # The epochs should correspond to when the features are defined. Here we use the time support directly
    range= (0, 2*np.pi), # The min and max of the bin array
    feature_names = ["angle"] # Let's give a name to our feature for better labelling of the output.
    ) 
tuning_curves
```
</div>

```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(
    data=spikes,
    features=angle, 
    bins=61, 
    epochs = angle.time_support,
    range=(0, 2 * np.pi),
    feature_names = ["angle"]
    )

tuning_curves
```

<div class="render-all">

The output is an xarray object with neurons along one dimension and angular bins along the other.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = plt.figure()
plt.subplot(221)
tuning_curves[0].plot()
# plt.plot(tuning_curves[0])
plt.ylabel("Firing rate (Hz)")
plt.subplot(222,projection='polar')
plt.plot(tuning_curves.angle, tuning_curves[0].values)
plt.subplot(223)
tuning_curves[1].plot()
plt.ylabel("Firing rate (Hz)")
plt.subplot(224,projection='polar')
plt.plot(tuning_curves.angle, tuning_curves[1].values)
plt.tight_layout()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/01-00.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/01-00.png)
:::
</div>


<div class="render-all">

Most of these neurons are sharply tuned to a preferred direction.

The preferred direction is simply the angle at which firing rate peaks (`idxmax` over the angle dimension).

</div>

```{code-cell} ipython3
:tags: [render-all]

pref_ang = tuning_curves.idxmax(dim="angle")

print(pref_ang)
```

<div class="render-all">

The variable `pref_ang` maps each neuron to its preferred direction. Attaching it to the `spikes` object as metadata lets us use it for sorting and labelling later.

**Question:** Can you add `pref_ang` to the metadata of `spikes` under the field name `pref_ang`?

</div>


:::{admonition} Hint
:class: render-all

There are multiple ways of doing this:
```
tsgroup['label'] = metadata
tsgroup.label = metadata
tsgroup.set_info(label=metadata)
```
:::


```{code-cell} ipython3
# spikes['pref_ang'] = pref_ang
spikes.set_info(pref_ang = pref_ang)

spikes
```

<div class="render-all">

Let's visualize the spiking activity of each neuron alongside the animal's head-direction, with neurons ordered by their preferred direction. We restrict to a short window for clarity.

</div>

```{code-cell} ipython3
:tags: [render-all]

ex_ep = nap.IntervalSet(start=8910, end=8960)

fig = plt.figure()
plt.subplot(211)
plt.ylabel("Head direction (rad)")
plt.plot(angle.restrict(ex_ep))
plt.ylim(0, 2*np.pi)
plt.subplot(212)
plt.plot(spikes.restrict(ex_ep).to_tsd("pref_ang"), '|')
plt.ylabel("Neuron (pref. dir.)")
plt.xlabel("Time (s)")
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/01-01.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/01-01.png)
:::
</div>

## Compute correlograms

<div class="render-all">

The raster shows that some neurons tend to fire together, while others fire at opposite phases. A **cross-correlogram** quantifies this: it measures how often spikes from one neuron occur at each time lag relative to spikes from another, revealing whether pairs are co-active or anti-correlated.

**Question:** Can you compute cross-correlograms during wake for all pairs of neurons using `nap.compute_crosscorrelogram` and call the result `cc_wake`?

</div>

<div class="render-user">
```{code-cell} ipython3
cc_wake = nap.compute_crosscorrelogram(
    group=, # The neural activity as a TsGroup
    binsize=, # I suggest 200 ms bin
    windowsize=, # Let's do a 20 s window
    ep= # Which epoch to restrict the cross-correlograms. Here is it should be wakefulness.
    )
```
</div>

```{code-cell} ipython3
cc_wake = nap.compute_crosscorrelogram(spikes, binsize=0.2, windowsize=20.0, ep=wake_ep)
```

<div class="render-all">

The output is a pandas DataFrame where each column is a neuron pair, computed automatically for all combinations, and the index gives the time lag. Let's focus on two pairs: neurons 7 and 20 share a preferred direction, while neurons 7 and 26 prefer opposite directions.

Indexing: `cc[(7, 20)]` selects a correlogram column; `tuning_curves.sel(unit=[7, 20])` selects the corresponding tuning curves from the xarray.

</div>

```{code-cell} ipython3
:tags: [render-all]

index = spikes.keys()


fig = plt.figure()
plt.subplot(221)
tuning_curves.sel(unit=[7,20]).plot(x='angle', hue='unit')
plt.title("Tuning curves")
plt.ylabel("Firing rate (Hz)")
plt.subplot(222)
plt.plot(cc_wake[(7, 20)])
plt.xlabel("Time lag (s)")
plt.title("Cross-corr.")
plt.ylabel("Firing rate (Hz)")
plt.subplot(223)
tuning_curves.sel(unit=[7,26]).plot(x='angle', hue='unit')
plt.title("Tuning curves")
plt.ylabel("Firing rate (Hz)")
plt.subplot(224)
plt.plot(cc_wake[(7, 26)])
plt.xlabel("Time lag (s)")
plt.title("Cross-corr.")
plt.ylabel("Firing rate (Hz)")
plt.tight_layout()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/01-02.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/01-02.png)
:::
</div>


<div class="render-all">

Neurons with similar preferred directions are positively correlated at lag 0 (they co-fire), while neurons with opposite preferred directions are negatively correlated. The same neurons were also recorded during sleep.

**Question:** Can you compute cross-correlograms during sleep? Use a shorter bin (20 ms) and window (1 s) to match the faster timescales typical of sleep activity.

</div>

<div class="render-user">
```{code-cell} ipython3
cc_sleep = nap.compute_crosscorrelogram(
    group=, # The neural activity as a TsGroup
    binsize=, # I suggest 20 ms bin
    windowsize=, # Let's do a 1 s window
    ep= # Which epoch to restrict the cross-correlograms. Here is it should be sleep.
    )
```
</div>

```{code-cell} ipython3
cc_sleep = nap.compute_crosscorrelogram(spikes, 0.02, 1.0, ep=sleep_ep)
```

<div class="render-all">

Let's compare the wake and sleep correlograms for the same-direction pair and the opposite-direction pair.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = plt.figure()
plt.subplot(231)
tuning_curves.sel(unit=[7,20]).plot(x='angle', hue='unit')
plt.title("Tuning curves")
plt.ylabel("Firing rate (Hz)")
plt.subplot(232)
plt.plot(cc_wake[(7, 20)])
plt.xlabel("Time lag (s)")
plt.ylabel("Firing rate (Hz)")
plt.title("Wake")
plt.subplot(233)
plt.plot(cc_sleep[(7, 20)])
plt.xlabel("Time lag (s)")
plt.ylabel("Firing rate (Hz)")
plt.title("Sleep")
plt.subplot(234)
tuning_curves.sel(unit=[7,26]).plot(x='angle', hue='unit')
plt.subplot(235)
plt.plot(cc_wake[(7, 26)])
plt.xlabel("Time lag (s)")
plt.ylabel("Firing rate (Hz)")
plt.subplot(236)
plt.plot(cc_sleep[(7, 26)])
plt.xlabel("Time lag (s)")
plt.ylabel("Firing rate (Hz)")
plt.tight_layout()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/01-03.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/01-03.png)
:::
</div>

<div class="render-all">

The pairwise relationships are preserved during sleep, even though the animal is not moving and head-direction is undefined. Cells that co-fired during wake continue to do so during sleep, suggesting the coordination is not driven purely by a shared sensory input.

</div>

## Modelling extracellular spike history effects with GLM

<div class="render-all">

The correlograms showed that pairwise coordination is preserved across brain states: co-active pairs during wake remain co-active during sleep, and anti-correlated pairs remain anti-correlated. The goal now is to quantify these functional interactions with a generalized linear model (GLM). Because cells influence each other, the recent spike history of one cell should help predict the activity of another.

</div>

<div class="render-all">

**Question:** Can we predict each neuron's spiking from the recent spike history of the population alone — without using the head-direction signal?

To keep fitting times short we restrict the analysis to the first 3 minutes of the wake epoch.

</div>

```{code-cell} ipython3
:tags: [render-all]

# restrict wake epoch to first 3 minutes
wake_ep = nap.IntervalSet(
    start=wake_ep.start[0], end=wake_ep.start[0] + 3 * 60
)
```

<div class="render-all">
To use the GLM we first need to discretize the spike trains into time bins using pynapple's `count` method.

**Question:** Can you bin the spike trains into 10 ms bins during `wake_ep` and call the result `count`?

</div>

<div class="render-user">
```{code-cell} ipython3
bin_size = 0.01
count =   # Bin spike trains during wake_ep
print(count.shape)
```
</div>

```{code-cell} ipython3
bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

print(count.shape)
```

<div class="render-all">

We sort the neuron columns by preferred direction so that the population activity plots are easier to interpret.

</div>

```{code-cell} ipython3
:tags: [render-all]

count = count[:, np.argsort(pref_ang.values)]
```

<div class="render-all">

Our goal is to estimate pairwise functional interactions: the recent spike history of all neurons is used to predict each neuron's current activity. We start with a single neuron to build intuition before scaling to the full population.

The simplest formulation uses counts in a fixed-length window $y_{t-i}, \dots, y_{t-1}$ to predict the next count $y_t$.

Select neuron 0 from `count` (call it `neuron_count`) and define `epoch_one_spk` as the first 1.2 s of the recording for visualization.

</div>

```{code-cell} ipython3
:tags: [render-all]

# select a neuron's spike count time series
neuron_count = count[:, 0]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support.start[0], end=count.time_support.start[0] + 1.2
)
```

## Features Construction

<div class="render-all">

Let's set the spike history window — how far back in time we look to predict the current firing rate.

- Set the window to 800 ms (0.8 s).
- Visualize it with `doc_plots.plot_history_window`.

</div>

```{code-cell} ipython3
:tags: [render-all]

# set the size of the spike history window in seconds
window_size_sec = 0.8

fig = doc_plots.plot_history_window(neuron_count, epoch_one_spk, window_size_sec);
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-01.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-01.png)
:::
</div>

<div class="render-all">

At each time point we slide the window by one bin and stack the spike counts into a matrix. Each row becomes the predictor for the rate in the next bin (the red rectangle in the figure).

</div>

```{code-cell} ipython3
:tags: [render-all]

doc_plots.run_animation(neuron_count, epoch_one_spk.start[0])
```

<div class="render-all">

For time points $t$ smaller than the window size there is no complete history available. Zero-padding would introduce spurious edge artifacts, so instead nemos restricts predictions to $t \geq$ window size and fills earlier entries with NaN.

You can construct this feature matrix with the [`HistoryConv`](https://nemos.readthedocs.io/en/latest/generated/basis/nemos.basis.HistoryConv.html#nemos.basis.HistoryConv) basis.

**Question: Can you:**
    - Convert the window size in number of bins (call it `window_size`)
    - Define an `HistoryConv` basis covering this window size (call it `history_basis`).
    - Create the feature matrix with `history_basis.compute_features` (call it `input_feature`).

</div>

<div class="render-user">
```{code-cell} ipython3
# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)
# define the history bases
history_basis = # Parameter indicate the window size in bins
# create the feature matrix
input_feature =  # Using history_basis compute features on neuron_count
```
</div>

```{code-cell} ipython3
# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)
# define the history bases
history_basis = nmo.basis.HistoryConv(window_size)
# create the feature matrix
input_feature = history_basis.compute_features(neuron_count)
```

<div class="render-all">

NeMoS NaN-pads the first `window_size` time points where a full history is not available.
 
</div>

```{code-cell} ipython3
:tags: [render-all]

# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0]) 
```

<div class="render-all">

Let's verify that the feature matrix dimensions match our expectation.

</div>

```{code-cell} ipython3
:tags: [render-all]

print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")
print(f"Feature shape: {input_feature.shape}")
```

<div class="render-all">

Let's visualize the feature matrix over a few time bins to confirm it looks as expected.

</div>

```{code-cell} ipython3
:tags: [render-all]

suptitle = "Input feature: Count History"
neuron_id = 0
fig = workshop_utils.plot_features(input_feature, count.rate, suptitle)
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-02.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-02.png)
:::
</div>


<div class="render-all">

The time axis runs backward because convolution reverses it — this is equivalent, and we can interpret each weight as the influence of a spike at that lag on the future firing rate. With 10 ms bins and a 0.8 s window the feature has 80 dimensions. We will learn these 80 weights by maximum likelihood.

</div>


## Fitting a single neuron model

<div class="render-all">

When working with real data it is good practice to train on one portion and evaluate on another — a process known as cross-validation. The optimal strategy depends on data structure (continuous time series vs. independent trials). Here we use a simple first-half/second-half split, which is reasonable provided the firing statistics are stationary across the recording.

</div>

```{code-cell} ipython3
:tags: [render-all]

# construct the train and test epochs
duration = input_feature.time_support.tot_length("s")
start = input_feature.time_support["start"]
end = input_feature.time_support["end"]

# define the interval sets
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)
```

<div class="render-all">

**Question: Can you fit the glm to the first half of the recording and visualize the maximum likelihood weights?**

The model used should be a `nmo.glm.GLM` with the solver `LBFGS`.

</div>

<div class="render-user">
```{code-cell} ipython3
# define the GLM object
model = nmo.glm.GLM() # Parameter is the solver name
# Fit over the training epochs
model.fit(
    input_feature.restrict(), # Parameter is the feature matrix restricted to the first half
    neuron_count.restrict() # Parameter is the binned spike count time series restricted to the first half
)
```
</div>

```{code-cell} ipython3
# define the GLM object
model = nmo.glm.GLM(solver_name="LBFGS")

# Fit over the training epochs
model.fit(
    input_feature.restrict(first_half),
    neuron_count.restrict(first_half)
)
```

<div class="render-all">

Each weight represents the influence of a spike at time lag $i$ on the log-rate at time $t$. Let's plot the learned filter.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_), lw=2, label="GLM raw history 1st Half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-03.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-03.png)
:::
</div>

<div class="render-all">

The filter looks like a decaying exponential buried in noise, suggesting it can be described with far fewer degrees of freedom than 80 raw weights. If the noise reflects over-fitting, the weights should differ substantially between the two halves of the data.

</div>


<div class="render-all">

**Question: Can you fit a new model on the second half of the data and call it `model_second_half`?**

</div>

<div class="render-user">
```{code-cell} ipython3
# fit on the other half of the data
model_second_half =  # Parameter is the solver name
model_second_half.fit(
    , # Parameter is the feature matrix restricted to the second half
     # Parameter is the binned spike count time series restricted to the second half
)
```
</div>

```{code-cell} ipython3
# fit on the other half of the data

model_second_half = nmo.glm.GLM(solver_name="LBFGS")
model_second_half.fit(
    input_feature.restrict(second_half),
    neuron_count.restrict(second_half)
)
```

<div class="render-all">

Let's plot the weights learned on the second half of the data and compare them to those learned on the first half.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_),
         label="GLM raw history 1st Half", lw=2)
plt.plot(np.arange(window_size) / count.rate,  np.squeeze(model_second_half.coef_),
         color="orange", label="GLM raw history 2nd Half", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-04.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-04.png)
:::
</div>

<div class="render-all">

The high-frequency fluctuations are inconsistent between the two fits, confirming they reflect noise rather than signal — a hallmark of over-fitting. The underlying decaying trend is consistent, but masked by noise. At 1 ms resolution this would be far worse: 800 coefficients instead of 80. We need a way to reduce dimensionality while preserving the smooth structure of the filter.

</div>

## Reducing feature dimensionality

<div class="render-all">
NeMoS' `basis` module lets us reparametrize the filter in terms of a small set of smooth basis functions, greatly reducing dimensionality. For spike-history inputs the standard choice is the raised cosine log-stretched basis ([Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003)), shown below.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = doc_plots.plot_basis()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-05.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-05.png)
:::
</div>

:::{note}

Choosing the right basis is an important modelling decision. NeMoS provides several options. The raised cosine log-stretched basis is a well-established choice for spike history; the log-stretch gives finer resolution at short lags where neural effects are strongest.
:::

<div class="render-all">

We initialize `RaisedCosineLogConv` by specifying the number of basis functions and the convolution window size. More basis functions give finer temporal resolution at the cost of additional parameters to estimate.

**Question: Can you define the basis `RaisedCosineLogConv`and name it `basis`?**

Basis parameters:
- 8 basis functions.
- Window size of 0.8sec.

</div>


<div class="render-user">
```{code-cell} ipython3
# a basis object can be instantiated in "conv" mode for convolving the input.
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=, # Number of basis functions
    window_size= # Window size in bins
)
```
</div>

```{code-cell} ipython3
# a basis object can be instantiated in "conv" mode for convolving  the input.
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size
)
```

<div class="render-all">

The raw history predictor had 80 values per time point — an order of magnitude more than the 8 basis coefficients we use instead. Beyond the memory and speed gain, the compression directly reduces over-fitting by constraining the filter to a smooth subspace. At 1 ms resolution the saving would be two orders of magnitude.

To apply the basis, call `compute_features` — this convolves the counts with the basis kernels without materializing the large raw-history matrix.

**Question: Can you:**
- Convolve the counts with the basis functions. (Call the output `conv_spk`)
- Print the shape of `conv_spk` and compare it to `input_feature`.

</div>

<div class="render-user">
```{code-cell} ipython3
# equivalent to
# `nmo.convolve.create_convolutional_predictor(basis_kernels, neuron_count)`
conv_spk = basis.compute_features() # Parameter is the binned spike count time series
print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")
```
</div>

```{code-cell} ipython3
# equivalent to
# `nmo.convolve.create_convolutional_predictor(basis_kernels, neuron_count)`
conv_spk = basis.compute_features(neuron_count)

print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")
```

<div class="render-all">

Let's visualize the convolved features over two short windows — one with a single spike and one with multiple spikes — to see how the basis captures the history.

</div>

```{code-cell} ipython3
:tags: [render-all]

# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

fig = doc_plots.plot_convolved_counts(neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk)
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-06.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-06.png)
:::
</div>

## Fit a GLM with basis features with reduced dimensionality

<div class="render-all">

Now that we have our "compressed" history feature matrix, we can fit the parameters for a new GLM model using these features.

**Question: Can you fit the model using the compressed features? Call it `model_basis`.**

</div>

<div class="render-user">
```{code-cell} ipython3
# use restrict on interval set training
model_basis = nmo.glm.GLM() # Parameter is the solver name
model_basis.fit(
    , # Parameter is the convolved feature matrix restricted to the first half
     # Parameter is the binned spike count time series restricted to the first half
)
```
</div>

```{code-cell} ipython3
# use restrict on interval set training
model_basis = nmo.glm.GLM(solver_name="LBFGS")
model_basis.fit(conv_spk.restrict(first_half), neuron_count.restrict(first_half))
```

<div class="render-all">

(head-direction-basis-full)=

The model learned 8 coefficients — one per basis function. To recover the full history filter in the original time domain we project them back through the basis kernels.

</div>

```{code-cell} ipython3
:tags: [render-all]

print(model_basis.coef_)
```

<div class="render-all">

We reconstruct the history filter by multiplying the basis kernels (from `evaluate_on_grid`) by the learned coefficients using `np.matmul`.

</div>

```{code-cell} ipython3
:tags: [render-all]

# get the basis function kernels
_, basis_kernels = basis.evaluate_on_grid(window_size)

# multiply with the weights
self_connection = np.matmul(basis_kernels, model_basis.coef_)

print(self_connection.shape)
```

<div class="render-all">

To check whether the basis model is more stable than the raw-history model, we refit on the second half and compare the two filter estimates.

</div>

```{code-cell} ipython3
:tags: [render-all]

# fit on the other half of the data
model_basis_second_half = nmo.glm.GLM(solver_name="LBFGS").fit(
    conv_spk.restrict(second_half), neuron_count.restrict(second_half)
)
self_connection_second_half = np.matmul(basis_kernels, model_basis_second_half.coef_)
```

<div class="render-all">

Let's plot the weights learned on the second half of the data and compare them to those learned on the first half.

</div>

```{code-cell} ipython3
:tags: [render-all]

time = np.arange(window_size) / count.rate
fig = plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), "k", alpha=0.3, label="GLM raw history 1st half")
plt.plot(time, np.squeeze(model_second_half.coef_), alpha=0.3, color="orange", label="GLM raw history 2nd half")
plt.plot(time, self_connection, "--k", lw=2, label="GLM basis 1st half")
plt.plot(time, self_connection_second_half, color="orange", lw=2, ls="--", label="GLM basis 2nd half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-07.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-07.png)
:::
</div>

<div class="render-all">

**Question:** Can you predict the firing rate from both `model` (raw history) and `model_basis` (basis)? Call the outputs `rate_history` and `rate_basis`, converting from spikes/bin to spikes/s.

</div>

<div class="render-user">
```{code-cell} ipython3
rate_basis = model_basis.predict() # Parameter is the convolved feature matrix
rate_history = model.predict() # Parameter is the original feature
# convert the rate from spike/bin to spike/sec by multiplying with neuron_count.rate
rate_basis = rate_basis * conv_spk.rate
rate_history = rate_history * conv_spk.rate
```
</div>

```{code-cell} ipython3
rate_basis = model_basis.predict(conv_spk) * conv_spk.rate
rate_history = model.predict(input_feature) * conv_spk.rate
```

<div class="render-all">

Let's compare predicted and smoothed observed rates over a short test window.

</div>

```{code-cell} ipython3
:tags: [render-all]

# plot the rates
fig = doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection raw history":rate_history, "Self-connection basis": rate_basis}
);
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-08.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-08.png)
:::
</div>

## All-to-all Connectivity

<div class="render-all">

We now extend the model to the full population: each neuron's firing rate is predicted from the recent spike history of all simultaneously recorded neurons. Convolving the basis with each neuron's counts gives a predictor array of shape `(num_time_points, num_neurons * num_basis_funcs)`.

</div>

## Preparing the features

<div class="render-all">

**Question:** Can you convolve all neurons and store the result in `convolved_count`?

Because we are now passing a multi-neuron array, call `basis.set_input_shape(count)` first so the basis knows the expected input dimensionality.

</div>

<div class="render-user">
```{code-cell} ipython3
# reset the input shape by passing the pop. count
print(count.shape)
print(152/8)
basis.set_input_shape(count)
# convolve all the neurons
convolved_count = basis.compute_features() # Parameter is the binned spike count time series
```
</div>

```{code-cell} ipython3
# reset the input shape by passing the pop. count
print(count.shape)
print(152/8)

basis.set_input_shape(count)

# convolve all the neurons
convolved_count = basis.compute_features(count)
```

<div class="render-all">

Verify that the output shape is `(n_samples, n_basis_funcs × n_neurons)`.

</div>

```{code-cell} ipython3
:tags: [render-all]

print(f"Convolved count shape: {convolved_count.shape}")
```

(head-direction-fit-full)=
## Fitting the Model

<div class="render-all">

We use [`PopulationGLM`](https://nemos.readthedocs.io/en/latest/generated/glm/nemos.glm.PopulationGLM.html) to fit all neurons at once. Conditioned on past activity, the population log-likelihood is the sum of individual neuron log-likelihoods, so joint fitting is equivalent to fitting each neuron independently — but more convenient.

**Question: Can you:**
- Fit a `PopulationGLM`? Call the object `model`. Solver should be `LBFGS`.
- Use Ridge regularization with a `regularizer_strength=0.1`?
- Print the shape of the estimated coefficients.

</div>

<div class="render-user">
```{code-cell} ipython3
model = nmo.glm.PopulationGLM(
    regularizer=, # Regularizer type
    solver_name=, # Solver name
    regularizer_strength= # Regularization strength
    ).fit( , ) # Parameters are the convolved feature matrix and the binned spike count time series
print(f"Model coefficients shape: {model.coef_.shape}")
```
</div>

```{code-cell} ipython3
model = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    ).fit(convolved_count, count)

print(f"Model coefficients shape: {model.coef_.shape}")
```

## Comparing model predictions.

<div class="render-all">

The neuron columns are sorted by preferred direction.

**Question:** Can you predict the firing rate of each neuron from `convolved_count` (call it `predicted_firing_rate`) and convert from spikes/bin to spikes/s?

</div>

<div class="render-user">
```{code-cell} ipython3
predicted_firing_rate = model.predict() # Parameter is the convolved feature matrix
# convert the rate from spike/bin to spike/sec by multiplying with conv_spk.rate
predicted_firing_rate = predicted_firing_rate * conv_spk.rate
```
</div>

```{code-cell} ipython3
predicted_firing_rate = model.predict(convolved_count) * conv_spk.rate
```

<div class="render-all">

Let's compare the model-predicted tuning curves against the empirical ones and inspect the predicted firing rate over time.

</div>

```{code-cell} ipython3
:tags: [render-all]

# use pynapple for time axis for all variables plotted for tick labels in imshow
fig = workshop_utils.plot_head_direction_tuning_model(tuning_curves, spikes, angle, 
                                                predicted_firing_rate, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv");
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-09.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-09.png)
:::
</div>

<div class="render-all">

Let's overlay all three model predictions — raw history, single-neuron basis, and all-to-all basis — on the same neuron to assess the improvement.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: basis": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-10.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-10.png)
:::
</div>

## Visualizing the connectivity

<div class="render-all">

Finally, we can extract and visualize the pairwise coupling filters.

**Question:** Can you reshape the coefficient matrix into a `(n_neurons, n_neurons, n_basis_funcs)` array using `basis.split_by_feature`?

</div>

```{code-cell} ipython3
:tags: [render-all]

# original shape of the weights
print(f"GLM coeff: {model.coef_.shape}")
```

<div class="render-all">

You can use the `split_by_feature` method of `basis` for this. It will reshape the coefficient vector into a 3D array.

![Reshape coefficients](../../../_static/coeff_reshape.png)

</div>

<div class="render-user">
```{code-cell} ipython3
# split the coefficient vector along the feature axis (axis=0)
weights_dict = basis.split_by_feature() # Parameter is the model coefficients. Axis is 0
# The output is a dict with key the basis label, 
# and value the reshaped coefficients
weights = weights_dict["RaisedCosineLogConv"]
print(f"Re-shaped coefficients: {weights.shape}")
```
</div>

```{code-cell} ipython3
# split the coefficient vector along the feature axis (axis=0)
weights_dict = basis.split_by_feature(model.coef_, axis=0)

# the output is a dict with key the basis label, 
# and value the reshaped coefficients
weights = weights_dict["RaisedCosineLogConv"]
print(f"Re-shaped coeff: {weights.shape}")
```

<div class="render-all">

The shape is `(sender_neuron, num_basis, receiver_neuron)`. We project back through the basis kernels with `np.einsum` to get the full coupling filters:

`(sender, num_basis, receiver) × (time_lag, num_basis) → (sender, receiver, time_lag)`

</div>

```{code-cell} ipython3
:tags: [render-all]

responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

print(responses.shape)
```

<div class="render-all">

Each entry in the grid shows how strongly neuron *j*'s past activity drives neuron *i*'s firing rate, with both axes sorted by preferred direction.

</div>

```{code-cell} ipython3
:tags: [render-all]

predicted_tuning_curves = nap.compute_tuning_curves(
    data=predicted_firing_rate,
    features=angle, 
    bins=61, 
    epochs = angle.time_support,
    range=(0, 2 * np.pi),
    feature_names = ["angle"]
    )

                                                 
fig = workshop_utils.plot_coupling_filters(responses, predicted_tuning_curves)
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../../_static/_check_figs/02-11.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../../_static/_check_figs/02-11.png)
:::
</div>

<div class="render-all">

These coupling filters capture the functional influence of each neuron on every other, sorted by preferred direction. Neurons with similar preferences show positive coupling; those with opposing preferences show negative coupling. These are functional connections inferred from shared tuning, not direct synaptic contacts.
</div>
