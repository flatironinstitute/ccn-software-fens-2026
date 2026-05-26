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

This notebook can be downloaded as **{nb-download}`01_head_direction.ipynb`**. See the button at the top right to download as markdown or pdf.
:::


# Analyzing head-direction cells with Pynapple

<div class="render-all">

In this tutorial, we will learn how to use pynapple to analyze electrophysiological data.

We will analyze extracellular recordings of head-direction cells recorded in the 
anterodorsal thalamic nucleus (ADn) of the mouse. We will use a NWB file containing spike times of neurons and the head-direction of the animal over time.
We will study the relationship between neurons during wakefulness and sleep with cross-correlograms.

The pynapple documentation can be found [here](https://pynapple.org).

We will use pynapple to do the following tasks:

1. Loading a NWB file
2. Compute tuning curves
3. Compute cross-correlograms

Let's start by importing all the packages.

</div>

```{code-cell} ipython3
:tags: [render-all]

import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo

# some helper plotting functions
import workshop_utils

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure some plots
plt.style.use(nmo.styles.plot_style)
```

## Fetch and load data

<div class="render-all">

The dataset we will use is from this study : [Peyrache et al., 2015](https://www.nature.com/articles/nn.3968).

If you ran the workshop setup script, you should have this file downloaded already. 
If not, the function we'll use to fetch it will download it for you. 
This function is called `fetch_data`, and can be imported from the `workshop_utils` module. 
This function will give us the file path to where the data is stored. 

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

There are a lot of neurons. The neurons that interest us are the neurons labeled `adn`. 

**Question:** Using the slicing method of your choice, can you select only the neurons in `adn` that are above 2 Hz firing rate?

There are multiple options here. As a reminder, metadata can be accessed like a dictionary or as attributes. There are also
functions that can help you filter neurons based on metadata.

1. `spikes.label` returns a pandas Series with the metadata of the neurons.
2. `spikes['label']` returns a pandas Series with the metadata of the neurons.
3. Functions like [`spikes.getby_category`](https://pynapple.org/generated/pynapple.TsGroup.getby_category.html#pynapple.TsGroup.getby_category)
    or [`spikes.getby_threshold`](https://pynapple.org/generated/pynapple.TsGroup.getby_threshold.html#pynapple.TsGroup.getby_threshold) can help you filter neurons based on metadata.

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

But are the data actually loaded or not?
If you look at the type of `angle`, you will see that it is a `Tsd` object.
But what about the underlying data array?
The underlying data array is stored in the property `d` of the `Tsd` object.
If you print it, you will see that it is an `h5py` array.
By default, data are lazy-loaded. This can be useful when reading larger-than-memory arrays from disk with memory mapping.

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

NWB file can save intervals with multiple labels. The object `IntervalSet` includes the labels as a metadata object.

**Question:** Using the column `tags`, can you create one `IntervalSet` object for intervals labeled `wake` and one `IntervalSet` object for intervals labeled `sleep`?

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

Now we have 
- spikes
- a behavioral feature (i.e. head-direction), 
- epochs corresponding to when the feature is defined (i.e. when the head-direction was recorded).

We can compute tuning curves, i.e. the firing rate of neurons as a function of head-direction. 
We want to know how the firing rate of each neuron changes as a function of the head-direction of the animal during wakefulness.

To do this in pynapple, all you need is the call of a single function : `nap.compute_tuning_curves`!

**Question:** can you compute the firing rate of ADn units as a function of heading direction, i.e. a head-direction tuning curve and call the variable `tuning_curves`?

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

The output is a xarray object indexed by neuron and head\-direction: the first dimension corresponds to neurons, 
the second to angular bins, and additional metadata fields are included.

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

fig.savefig("../../_static/_check_figs/01-00.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/01-00.png)
:::
</div>


<div class="render-all">

Most of those neurons are head-direction neurons.

The next cell allows us to get a quick estimate of the neurons' preferred direction. 
Since this is a lot of xarray wrangling, it is given.

</div>

```{code-cell} ipython3
:tags: [render-all]

pref_ang = tuning_curves.idxmax(dim="angle")

print(pref_ang)
```

<div class="render-all">

The variable `pref_ang` contains the preferred direction of each neuron. 
This information can be useful to add to the metainformation of the `spikes` object since it is neuron-specific information.

**Question:** Can you add it to the metainformation of `spikes`? The metadata field should be called `pref_ang`.

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

This index maps a neuron to a preferred angular direction between 0 and 2pi. 
Let's visualize the spiking activity of the neurons based on their preferred direction 
as well as the head-direction of the animal. To make it easier to see, we will restrict the data to a small epoch.

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

fig.savefig("../../_static/_check_figs/01-01.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/01-01.png)
:::
</div>

## Compute correlograms

<div class="render-all">

We see that some neurons have a correlated activity meaning they tend to fire together, while others have an anti-correlated activity meaning when one neuron fires, the other does not.
Can we quantify this correlation between pairs of neurons? To do this, we can compute cross-correlograms between pairs of neurons.
A cross-correlogram measures the correlation between the spike trains of two neurons as a function of time lag. It counts how often spikes from one neuron occur at different time lags relative to spikes from another neuron.
In pynapple, we use the function `nap.compute_crosscorrelogram` to compute cross-correlograms between pairs of neurons.

**Question:** Can you compute cross-correlograms during wake for all pairs of neurons and call it `cc_wake`?

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

The output is a pandas DataFrame where each column is a pair of neurons. All pairs of neurons are computed automatically.
The index shows the time lag.
Let's visualize some cross-correlograms. 
To make things easier, we will focus on two pairs of neurons: one pair that fires for the same direction and one pair that fires for opposite directions.

The pair (7, 20) fires for the same direction while the pair (7, 26) fires for opposite directions. 

To index pandas columns, you can do `cc[(7, 20)]`.

To index xarray tuning curves, you can do `tuning_curves.sel(unit=[7,20])`

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

fig.savefig("../../_static/_check_figs/01-02.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/01-02.png)
:::
</div>


<div class="render-all">

As you can see, the pair of neurons that fire for the same direction have a positive correlation at time lag 0, meaning they tend to fire together.
The pair of neurons that fire for opposite directions have a negative correlation at time lag 0, meaning when one neuron fires, the other does not.

Pairwise correlations were computed during wakefulness. The activity of the neurons was also recorded during sleep.

**Question:** can you compute the cross-correlograms during sleep?

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

Let's visualize the cross-correlograms during wake and sleep for the pair of neurons that fire for the same direction 
and the pair of neurons that fire for opposite directions.

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

fig.savefig("../../_static/_check_figs/01-03.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/01-03.png)
:::
</div>

<div class="render-all">

What does it mean for the relationship between cells here? Remember that during sleep, the animal is not moving and therefore the head-direction is not defined.

</div>


