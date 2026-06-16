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
---

```{code-cell} ipython3
:tags: [render-all]

%matplotlib inline
```

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`01_fundamentals_of_pynapple.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

:::{admonition} Jupyter Lab tip
:class: important render-all

Newer versions of Jupyter Lab have addressed an issue with skipping around the notebook while scrolling. To make sure this fix is enabled, in the Jupyter Lab GUI, navigate to `Settings > Settings Editor > Notebook` and scroll down to the `Windowing mode` setting and make sure it is set to `contentVisibility`. 

Also reminder to presenter: Go to `View > Appearance`, select `Simple Interface` and turn off everything else to hide as many bars as possible. And maybe activate `Presentation Mode`.

And turn on `View > Render side-by-side` (shortcut `Shift+R`).
:::


<div class="render-all">

# Learning the fundamentals of pynapple

## Learning objectives


- Instantiate pynapple objects
- Make pynapple objects interact
- Use numpy with pynapple
- Slice pynapple objects
- Add metadata to pynapple objects
- Apply core functions of pynapple

**Resources:**
- [Pynapple documentation](https://pynapple.org)
- [API reference for objects and methods](https://pynapple.org/api.html)


Let's start by importing the pynapple package and matplotlib to see if everything is correctly installed.
If an import fails, you can do `!pip install pynapple matplotlib` in a cell to fix it.

</div>

```{code-cell} ipython3
:tags: [render-all]

%matplotlib inline
import workshop_utils
import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import nemos as nmo
plt.style.use(nmo.styles.plot_style)
```

For this notebook we will work with fake data. The following cell generates a set of variables that we will use to create the different pynapple objects.

```{code-cell} ipython3
:tags: [render-all]

cos_ts = np.arange(0,100,0.1)
cos_data = np.cos(1/4*cos_ts)

rng = np.random.default_rng(1)
rand_ts = np.arange(0,100)
rand_data = rng.standard_normal((100,3))
rand_col = ['pineapple', 'banana', 'tomato']

spiral_ts = np.arange(0, 100, 0.5)
d = np.linspace(-10,10,100)
spiral_data = np.zeros((len(spiral_ts),len(d),len(d)))
for i,t in enumerate(spiral_ts):
    rv = stats.multivariate_normal([t/10*np.cos(t),t/10*np.sin(t)])
    pos = np.dstack(np.meshgrid(d,d))
    spiral_data[i] = rv.pdf(pos)

rng = np.random.default_rng(2)
t = np.arange(0,100,0.1)
p = np.cos(1/4*t)/5
p = np.where(p>0,p,0)
burst_times = t[np.where(rng.binomial(n=1, p=p))[0]]
random_times = np.sort(rng.uniform(0, 100, 100))
slow_times = np.arange(0,100,10)
```

## Instantiate pynapple objects 

Pynapple objects can help reduce the size of our workspace by associating relevant data into a single object. Here we will show how to instantiate all the different pynapple objects.

Let's start with objects that combine data points with corresponding timestamps (as well as column names for a `TsdFrame`). Suppose we have the following sets of data.

```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_ts, cos_data)
plt.title("Cosine Wave")

plt.figure()
plt.plot(rand_ts, rand_data, label=rand_col)
plt.title("Random Data")
plt.legend()

anim = workshop_utils.animate_2d_movie(spiral_data)
plt.title("Spiral Data")
anim.run()
```

### Tsd

`Tsd` objects are used to represent 1-dimensional time series data, such as voltage traces.

<div class="render-all">

**Question:** Which dataset belongs as a `Tsd`? Can you instantiate and print the pynapple object using the correct dataset? **HINT**: Name the variable `cos_tsd`.

</div>

```{code-cell} ipython3
cos_tsd = nap.Tsd(t=cos_ts, d=cos_data)
print(cos_tsd)
```

### TsdFrame

`TsdFrame` objects are used to represent 2-dimensional time series data, such as multiple calcium transients, where you can optionally specify the column names.

<div class="render-all">

**Question:** Which dataset belongs as a `TsdFrame`? Can you instantiate and print the pynapple object using the correct dataset? **HINT**: Name the variable `rand_tsd`. Don't forget the column names!

</div>

```{code-cell} ipython3
rand_tsd = nap.TsdFrame(t=rand_ts, d=rand_data, columns=rand_col)
print(rand_tsd)
```

### TsdTensor

`TsdTensors` objects are used to represent 3-dimensional time series data, such as movie frames.

<div class="render-all">

**Question:** Which dataset belongs as a `TsdTensor`? Can you instantiate and print the pynapple object using the correct dataset? **HINT**: Name the variable `spiral_tsd`.

</div>

```{code-cell} ipython3
spiral_tsd = nap.TsdTensor(t=spiral_ts, d=spiral_data)
print(spiral_tsd)
```

### IntervalSet

Pynapple `IntervalSet` objects combine start and end times into a single set of non-overlapping intervals.

<div class="render-all">

**Question:** Can you create and print an `IntervalSet` called `epochs` out of `starts` and `ends`? Be careful, times given above are in `ms`.

</div>

<div class="render-user">
```{code-cell} ipython3
starts = np.array([10000, 60000, 90000]) # starts of an epoch in `ms`
ends = np.array([20000, 80000, 95000])   # ends in `ms`
epochs = 
```
</div>

```{code-cell} ipython3
starts = np.array([10000, 60000, 90000]) # starts of an epoch in `ms`
ends = np.array([20000, 80000, 95000])   # ends in `ms`
epochs = nap.IntervalSet(start=starts, end=ends, time_units='ms')
print(epochs)
```

### Ts

Pynaple `Ts` objects allow us to define time stamps that aren't associated with any particular value or magnitude, such as spike times.

Suppose we record spike times from three different neurons, plotted below.

```{code-cell} ipython3
:tags: [render-all]

plt.figure(figsize=(8,3))
plt.plot(burst_times,np.zeros_like(burst_times),'|',markersize=50)
plt.plot(random_times,np.ones_like(random_times),'|',markersize=50)
plt.plot(slow_times,1+np.ones_like(slow_times),'|',markersize=50)
plt.yticks([0,1,2],labels=['burst_times','random_times','slow_times']);
```

<div class="render-all">

**Question:** Can you instantiate `Ts` objects for each set of spike times above?

</div>

<div class="render-user">
```{code-cell} ipython3
burst_neuron = 
random_neuron = 
slow_neuron = 
```
</div>

```{code-cell} ipython3
burst_neuron = nap.Ts(t=burst_times)
random_neuron = nap.Ts(t=random_times)
slow_neuron = nap.Ts(t=slow_times)
```

### TsGroup

These are a lot of `Ts` objects to have separately. We can use pynapple `TsGroup` objects to combine a group of `Ts` objects together into a single variable.

<div class="render-all">

**Question:** Can you instantiate a `TsGroup` to group together the `Ts` objects defined above and print the result?

</div>

<div class="render-user">
```{code-cell} ipython3
all_neurons =
```
</div>

```{code-cell} ipython3
all_neurons = nap.TsGroup({0:burst_neuron, 1:random_neuron, 2:slow_neuron})
print(all_neurons)
```

## Interaction between pynapple objects 

What started as 12 separate variables (`cos_ts`, `cos_data`, `rand_ts`, `rand_data`, `rand_col`, `spiral_ts`, `spiral_data`, `starts`, `ends`, `burst_times`, `random_times`, `slow_times`) has been reduced to 5 (`cos_tsd`, `rand_tsd`, `spiral_tsd`, `epochs`, `all_neurons`) using pynapple. Now we can see how these objects interact.

### time_support

<div class="render-all">

**Question:** Can you print the `time_support` of `all_neurons`?

</div>

```{code-cell} ipython3
print(all_neurons.time_support)
```

While our simulated experiment ran from 0 to 100 seconds, the `time_support` of `all_neurons` is defined over a slightly shorter interval. Because of this, the rate is inaccurate, since it's computed over the default `time_support`.

<div class="render-all">

**Question:** can you recreate the `tsgroup` object passing the right `time_support` during initialisation?

</div>
<div class="render-user">
```{code-cell} ipython3
all_neurons =
```
</div>

```{code-cell} ipython3
all_neurons = nap.TsGroup({0:burst_neuron, 1:random_neuron, 2:slow_neuron}, time_support = nap.IntervalSet(0, 100))
```

<div class="render-all">

**Question:** Can you print the `time_support` and `rate` to see how they changed?

</div>

```{code-cell} ipython3
print(all_neurons.time_support)
print(all_neurons.rate)
```

### restrict

What if we want to limit our data to intervals of interest? We can restrict any pynapple timeseries object using the object method `restrict`.

<div class="render-all">

**Question:** Can you create an `IntervalSet` object called `ep_signal` and use it to restrict the variable `cos_tsd`? Include two intervals: from 10s to 30s and from 50s to 100s. 

</div>
<div class="render-user">
```{code-cell} ipython3
ep_signal =
cos_tsd_signal =
```
</div>

```{code-cell} ipython3
ep_signal = nap.IntervalSet(start=[10, 50], end=[30, 100])
cos_tsd_signal = cos_tsd.restrict(ep_signal)
```

<div class="render-all">
    
We can print `cos_tsd_signal` to check that the timestamps are within `ep_signal`. Additionally, 
printing the `time_support` shows that it has been updated to match `ep_signal`.

</div>

```{code-cell} ipython3
print(cos_tsd_signal)
print(cos_tsd_signal.time_support)
```

### intersect

Pynapple `IntervalSet` objects can be intersected to create a new `IntervalSet` using the `intersect` method.

```{code-cell} ipython3
:tags: [render-all]

# randomly generated intervals for demonstration
rng = np.random.default_rng(3)
ep_random = nap.IntervalSet(np.sort(rng.uniform(0, 100, 20)))
print(ep_random)
```

<div class="render-all">

**Question:** Can you intersect `ep_signal` with `ep_random`?

</div>

<div class="render-user">
```{code-cell} ipython3
ep_intersect = 
```
</div>

```{code-cell} ipython3
ep_intersect = ep_signal.intersect(ep_random)
ep_intersect
```

<div class="render-all">
    
We can visualize the result using the provided function `workshop_utils.visualize_intervals`

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_intersect])
plt.yticks([0.25,0.5,0.75],["ep_signal","ep_random","ep_intersect"]);
```

### union

Pynapple `IntervalSet` objects can be joined using the `union` method.

<div class="render-all">

**Question:** Can you take the union of `ep_signal` and `ep_random`?

</div>

<div class="render-user">
```{code-cell} ipython3
ep_union = 
```
</div>

```{code-cell} ipython3
ep_union = ep_signal.union(ep_random)
```

<div class="render-all">

Let's visualize the results.

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_union])
plt.yticks([0.25,0.5,0.75],["ep_signal","ep_random","ep_union"]);
```

### set_diff

We can also subtract one `IntervalSet` from another using the `set_diff` method.

<div class="render-all">

**Question:** Can you take the set difference between `ep_signal` and `ep_random`? Do this twice, with each object acting as the base object. Do you expect the results to be the same?

</div>

<div class="render-user">
```{code-cell} ipython3
ep_signal_diff = 
ep_random_diff =
```
</div>

```{code-cell} ipython3
ep_signal_diff = ep_signal.set_diff(ep_random)
ep_random_diff = ep_random.set_diff(ep_signal)
```

<div class="render-all">

Visualizing the results makes it clear that order matters when using `set_diff`.

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_signal_diff, ep_random_diff])
plt.yticks([0.2,0.4,0.6,0.8],["ep_signal","ep_random","ep_signal_diff","ep_random_diff"]);
```

## Numpy & pynapple

<div class="render-all">

Pynapple timeseries objects (`Tsd`, `TsdFrame`, and `TsdTensor`) behave similarly to numpy arrays. They can be sliced using similar syntax, e.g.:

  `tsd[0:10] # First 10 elements`

Arithmetic operations also behave like numpy:

  `tsd = tsd + 1`

Finally, numpy functions are compatible with pynapple objects, and in many cases will return a pynapple object when the time axis is preserved.

**Question:** Can you compute the average of `rand_tsd` across columns and print the result?

</div>

```{code-cell} ipython3
print(np.mean(rand_tsd, 1))
```

<div class="render-all">

**Question:** Can you compute the average frame of `spiral_tsd` along the time axis using `np.mean` and print the result?

</div>

```{code-cell} ipython3
print(np.mean(spiral_tsd, 0))
```

In the first case we still have a pynapple object since the time axis has been preserved. In the second case, we're returned a numpy array.

## Slicing pynapple objects 

Multiple methods exists to slice pynapple object in addition to numpy-like indexing.

<div class="render-all">

**Question:** `IntervalSet` objects also behave similarly to numpy arrays. Using numpy-like indexing, can you extract the first and last epoch of `epochs`?
</div>

```{code-cell} ipython3
print(epochs[[0,2]])
```

### special case of slicing : `TsdFrame`

<div class="render-all">

For `TsdFrame` objects with column labels, the column labels are ignored when using numpy-like indexing.

</div>

```{code-cell} ipython3
:tags: [render-all]

tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3), columns = [12, 0, 1])
print(tsdframe)
```

<div class="render-all">

**Question:** What happens when you do `tsdframe[0]` vs `tsdframe[:,0]` vs `tsdframe[[12,1]]`?

</div>

```{code-cell} ipython3
print(tsdframe[0])
print(tsdframe[:,0])
try:
    print(tsdframe[[12,1]])
except Exception as e:
    print(e)
```

<div class="render-all">

To access `TsdFrame` objects by column names, index using the `loc` method.

**Question:** What happen when you do `tsdframe.loc[0]` and `tsdframe.loc[[0,1]]`?

</div>

```{code-cell} ipython3
print(tsdframe.loc[0])
print(tsdframe[:,0])
```

### get

Sometimes we want to find the nearest data point to a given time stamp. 

<div class="render-all">

**Question:** Using the `get` method, can you get the data point from `spiral_tsd` as close as possible to the time 50.1 seconds?

</div>

```{code-cell} ipython3
print(spiral_tsd.get(50.1))
```

We can also use the `get` method to grab all the data points in some time interval.

<div class="render-all">

**Question:** Using the `get` method, can you get the data point from `spiral_tsd` that occur between 50.1 and 52.1 seconds? **NOTE:** The time support is not updated using `get`.

</div>

```{code-cell} ipython3
print(spiral_tsd.get(50.1, 52.1))
```

### get_slice

If we want the *index* of the data nearest to some time stamp, we can use `get_slice` instead.

<div class="render-all">

**Question:** Using the `get_slice` method, can you get the index of `spiral_tsd` as close as possible to the time 50.1 seconds?

</div>

```{code-cell} ipython3
print(spiral_tsd.get_slice(50.1))
print(spiral_tsd.get_slice(50.1).start)
```

Similarly to `get`, `get_slice` can also be used to get the slice corresponding to some time interval.

+++

## Metadata

<div class="render-all">

Using metadata, we can attach additional info, such as experimental labels, to some of our pynapple objects. Specifically, the following three objects support metadata:

- `TsGroup` : to label each set of time stamps, e.g. neuron region
- `IntervalSet` : to label each interval, e.g. stimulus identity
- `TsdFrame` : to label each column, e.g. neurons in calcium imaging

Metadata can be any data type, and there are a few ways to add/access metadata to/from pynapple objects. 

</div>

### setting metadata
#### item assignment
Metadata can be added to an object using dictionary-like item assignment.

<div class="render-all">

**Question:** Can you add the metadata labels `["burst","random","slow"]` using item assignment to `all_neurons["label"]` and print the result?

</div>

```{code-cell} ipython3
all_neurons["label"] = ["burst", "random", "slow"]
all_neurons
```

#### attribute assignment
Metadata can also be set directly as an attribute to the object.

<div class="render-all">

**Question:** Can you add the values `[1, -1, 1]` to `epochs` as the attribute `epochs.direction`?

</div>

```{code-cell} ipython3
epochs.direction = [1, -1, 1]
epochs
```

#### set_info

<div class="render-all">
    
Each object also has the method `set_info` which allows you to set metadata using keyword arguments to the method.

**Question:** Can you add the rgb colors `[(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)]` as metadata of `rand_tsd` using the `set_info` method?

</div>

```{code-cell} ipython3
rand_tsd.set_info(color=[(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)])
rand_tsd
```

#### at initialization

<div class="render-all">

You can also add metadata at initialization as a dictionary using the keyword argument `metadata`: 

</div>

```{code-cell} ipython3
:tags: [render-all]

rand_tsd = nap.TsdFrame(
    t = rand_ts,d = rand_data,columns=rand_col,
    metadata={'color':['orange','yellow', 'red']}
)
print(rand_tsd)
```

### accessing metadata

<div class="render-all">

Similar to setting metadata, we can retrieve metadata as an attribute (i.e. `all_neurons.label`) or using item access (i.e. `all_neurons['label']`). Additionally we can use `get_info`, a complementary method to `set_info`, to access metadata.

</div>

```{code-cell} ipython3
all_neurons.get_info('label')
```

### slicing with metadata
<div class="render-all">

Metadata can be used to slice pynapple objects.

**Question:** Can you select only the elements of `all_neurons` with rate below 1Hz?

</div>

```{code-cell} ipython3
print(all_neurons[all_neurons.rate<1.0])

print(all_neurons[all_neurons['rate']<1.0])

print(all_neurons.getby_threshold("rate", 1, "<"))
```

<div class="render-all">

**Question:** Can you select the intervals in `epochs` with a direction of 1?
</div>

```{code-cell} ipython3
print(epochs[epochs.direction==1])
```

#### special case of slicing : `TsdFrame`

<div class="render-all">

Where metadata of `TsGroup` and `IntervalSet` objects are associated with each *row*, metadata of `TsdFrame` objects instead is associated with each *column*. This means slicing with metadata must be done on the second axis.

**Question:** Can you select the columns of `rand_tsd` where the color is orange?

</div>

```{code-cell} ipython3
print(rand_tsd[:, rand_tsd.color=="orange"])
```

## Core functions of pynapple 

Pynapple objects give us access to a number of core functions that are widely used in experimental settings. All of the functions can optionally take an `IntervalSet` to restrict the operation to the specified interval.

### count

The `count` methods allows us to count or bin the number of time points that fall within each window of a given bin size. 

<div class="render-all">

**Question:** Using the `count` function, can you count the number of events within 1 second bins for `all_neurons` over the `ep_signal` intervals?

<div class="render-user">
```{code-cell} ipython3
count =
```
</div>

</div>

```{code-cell} ipython3
count = all_neurons.count(1, ep_signal)
print(count)
```

<div class="render-all">

Let's visulize the results. **TIP**: Pynapple works directly with matplotlib. Passing a time series object to `plt.plot` will display the figure with the correct time axis.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure()
ax = plt.subplot(211)
plt.plot(count, 'o-')
plt.subplot(212, sharex=ax)
plt.plot(all_neurons.restrict(ep_signal).to_tsd(), '|')
```

### value_from

<div class="render-all">

We can map a set of timepoints to their nearest value from a timeseries data object by using the method `value_from`.

**Question:** Using the function `value_from`, can you assign values to `burst_neuron` from the `cos_tsd` time series into a new object called `burst_cos`?

</div>

<div class="render-user">
```{code-cell} ipython3
burst_cos = 
```
</div>

```{code-cell} ipython3
burst_cos = burst_neuron.value_from(cos_tsd)
burst_cos
```

<div class="render-all">

Let's plot these objects together.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_tsd)
plt.plot(burst_cos, 'o-')
plt.plot(burst_neuron.fillna(0), 'o')
```

### bin_average

<div class="render-all">

Oftentimes we need to match the sampling rates between different sets of data. Pynapple provides the `bin_average` function to downsample data.

**Question:** Can you downsample `rand_tsd` to one time point every 5 seconds?

</div>
<div class="render-user">
```{code-cell} ipython3
rand_downsamp = 
```
</div>

```{code-cell} ipython3
rand_downsamp = rand_tsd.bin_average(5.0)
```

<div class="render-all">

Let's plot the column for "tomato" and it's downsampled version.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(rand_tsd['tomato'])
plt.plot(rand_downsamp['tomato'], 'o-')
```

### threshold

<div class="render-all">

We may want to find all the intervals where our timeseries data exceeds some value. For 1-dimensional `Tsd` objects, Pynapple provides the `threshold` method to limit the `Tsd` above or below a certain value.

**Question**: Can you threshold `cos_tsd` for values above 0.0? Can you get the intervals of this thresholded data?

</div>
<div class="render-user">
```{code-cell} ipython3
cos_thresh =
ep_above = 
```
</div>

```{code-cell} ipython3
cos_thresh = cos_tsd.threshold(0.0)
ep_above = cos_thresh.time_support
print(ep_above)
```

<div class="render-all">

Let's visualize the resulting `Tsd` and `IntervalSet`.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_tsd)
plt.plot(cos_thresh, 'o-')
[plt.axvspan(s, e, alpha=0.2) for s,e in ep_above.values];
```

## First high level function : `compute_tuning_curves`

<div class="render-all">

Pynapple provides functions for standard analysis in systems neuroscience. The first function we will try is `compute_tuning_curves` that calculates the response of a cell to a particular feature. 

A good practice when using a function for the first time is to check the docstrings to learn how to pass the argument.

**Question**: Can you examine the docstring of `nap.compute_tuning_curves`?

</div>

```{code-cell} ipython3
print(nap.compute_tuning_curves.__doc__)
```

<div class="render-all">

**Question**: Can you compute the response (i.e. firing rate) of the units in `all_neurons` as function of the feature `cos` using the function `nap.compute_tuning_curves`?

</div>

```{code-cell} ipython3
tc = nap.compute_tuning_curves(all_neurons, cos_tsd, bins=5, feature_names=["cos"])
tc
```

<div class="render-all">

The output is an [xarray](https://docs.xarray.dev/en/stable/) object. It is a wrapper of numpy array with extra attributes. It allows to give coordinates to each dimensions as well as attaching attributes. We can make the output look better by labelling the feature we used.

The coordinates can be accessed with the `coords` attribute. The feature position (i.e. center of the bin) can be accessed with the attribute.

**Question**: Can you print the underlying the units number, bin center and bin edges of the tuning curve xarray object?

</div>

```{code-cell} ipython3
print(tc.unit.values)
print(tc.cos.values)
print(tc.occupancy)
print(tc.bin_edges)
print(tc.fs)
```

<div class="render-all">

**Question**: Can you plot the tuning curves for all units?

</div>

```{code-cell} ipython3
# tc.plot()
# tc.plot(row="unit")
# tc.plot(col="unit")
# tc[1].plot()
# plt.plot(tc[1].feat1, tc[1].values)
plt.plot(tc.cos, tc.values.T)
```

## Verify Your Setup

<div class="render-all">

**Question:** Does the following data download work correctly? If not, please ask a TA.

</div>

```{code-cell} ipython3
:tags: [render-all]

import workshop_utils
path = workshop_utils.fetch_data("Mouse32-140822.nwb")
print(path)
```
