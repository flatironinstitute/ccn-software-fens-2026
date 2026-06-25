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


Let's start by importing the pynapple package and other packages we'll need, as well as generate some fake data that we'll use throughout the notebook.

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

Pynapple objects can help reduce the size of our workspace by associating relevant data into a single object. Here we will show how to instantiate all the different pynapple objects. We'll start with objects that combine data points with corresponding timestamps (as well as column names for a `TsdFrame`).

<div class="render-all">

Suppose we have and experiment that generated the following data.

</div>

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

### Tsd, TsdFrame, TsdTensor

Pynapple has three core objects for timeseries data:
- [`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html#pynapple.Tsd) objects are used to represent 1-dimensional time series data, such as voltage traces.
- [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html#pynapple.TsdFrame) objects are used to represent 2-dimensional time series data, such as multiple calcium transients, where you can optionally specify the column names.
- [`TsdTensors`](https://pynapple.org/generated/pynapple.TsdTensor.html#pynapple.TsdTensor) objects are used to represent 3-dimensional time series data, such as movie frames.

<div class="render-all">

**Question:** Can you instantiate the correct pynapple objects ([`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html#pynapple.Tsd), [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html#pynapple.TsdFrame), and [`TsdTensor`](https://pynapple.org/generated/pynapple.TsdTensor.html#pynapple.TsdTensor)) for each of the data sets above? **NOTE**: Make sure to pass column names to the `TsdFrame` data.

</div>

<div class="render-user">
```{code-cell} ipython3
cos_tsd = 
rand_tsd = 
spiral_tsd = 
```
</div>

```{code-cell} ipython3
cos_tsd = nap.Tsd(t=cos_ts, d=cos_data)
print(cos_tsd)
rand_tsd = nap.TsdFrame(t=rand_ts, d=rand_data, columns=rand_col)
print(rand_tsd)
spiral_tsd = nap.TsdTensor(t=spiral_ts, d=spiral_data)
print(spiral_tsd)
```

### IntervalSet

Pynapple [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html#pynapple.IntervalSet) objects combine start and end times into a single set of non-overlapping intervals.

<div class="render-all">

**Question:** Can you create and print an [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html#pynapple.IntervalSet) called `epochs` out of `starts` and `ends`? Be careful, times given above are in `ms`.

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

Pynaple [`Ts`](https://pynapple.org/generated/pynapple.Ts.html#pynapple.Ts) objects allow us to define time stamps that aren't associated with any particular value or magnitude, such as spike times.

<div class="render-all">
    
Suppose we record spike times from three different neurons, plotted below.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure(figsize=(8,3))
plt.plot(burst_times,np.zeros_like(burst_times),'|',markersize=50)
plt.plot(random_times,np.ones_like(random_times),'|',markersize=50)
plt.plot(slow_times,1+np.ones_like(slow_times),'|',markersize=50)
plt.yticks([0,1,2],labels=['burst_times','random_times','slow_times']);
```

<div class="render-all">

**Question:** Can you instantiate [`Ts`](https://pynapple.org/generated/pynapple.Ts.html#pynapple.Ts) objects for each set of spike times above?

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

Instead of keeping all these `Ts` objects separate, we can use pynapple [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup) objects to combine a group of `Ts` objects together into a single variable.

<div class="render-all">

**Question:** Can you instantiate a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup) to group together the `Ts` objects defined above and print the result?

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

The [`time_support`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup.time_support) attribute is an `IntervalSet` associated with every pynapple object that specifys the time interval(s) over which the data is defined. This is inferred from the data or can be set directly during object initialization.

**Question:** Can you print the time support of `all_neurons`?

</div>

```{code-cell} ipython3
print(all_neurons.time_support)
```

While our simulated experiment ran from 0 to 100 seconds, the `time_support` of `all_neurons` is defined over a slightly shorter interval. Because of this, the rate is inaccurate, since it's computed over the default `time_support`.

<div class="render-all">

**Question:** can you recreate the `all_neurons` object passing the true `time_support` during initialisation?

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

<div class="render-all">

What if we want to limit our data to intervals of interest? We can restrict any pynapple timeseries object to a set of intervals defined by an `IntervalSet` using the object method [`restrict`](https://pynapple.org/generated/pynapple.Tsd.restrict.html#pynapple.Tsd.restrict).

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

<div class="render-all">
    
Pynapple `IntervalSet` objects can be intersected to create a new `IntervalSet` using the [`intersect`](https://pynapple.org/generated/pynapple.IntervalSet.intersect.html#pynapple.IntervalSet.intersect) method.

</div>

```{code-cell} ipython3
:tags: [render-all]

# random intervals
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

<div class="render-all">

Pynapple `IntervalSet` objects can be joined using the [`union`](https://pynapple.org/generated/pynapple.IntervalSet.union.html#pynapple.IntervalSet.union) method.

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

<div class="render-all">

We can also subtract one `IntervalSet` from another using the [`set_diff`](https://pynapple.org/generated/pynapple.IntervalSet.set_diff.html#pynapple.IntervalSet.set_diff) method.

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

### value_from

<div class="render-all">

We can map a set of timepoints to their nearest value from a different timeseries data object by using the method [`value_from`](https://pynapple.org/generated/pynapple.Ts.value_from.html#pynapple.Ts.value_from).

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

### interpolate

<div class="render-all">

We can resample or upsample one pynapple object to the timepoints of another pynapple object using the [`interpolate`](https://pynapple.org/generated/pynapple.TsdFrame.interpolate.html#pynapple.TsdFrame.interpolate) method. This is useful for matching the sampling rate between two pynapple objects.

**Question:** Can you upsample `rand_tsd` to the sampling rate of `cos_tsd` using the `interpolate` method?

</div>
<div class="render-user">
```{code-cell} ipython3
rand_tsd_interp = 
```
</div>

```{code-cell} ipython3
rand_tsd_interp = rand_tsd.interpolate(cos_tsd)
```

<div class="render-all">

Let's visualize the results below.

</div>

```{code-cell} ipython3
fig,axs=plt.subplots(2,1, sharex=True, sharey=True)
axs[0].plot(rand_tsd, 'o-')
axs[0].set_title("original points")
axs[1].plot(rand_tsd_interp, 'o-')
axs[1].set_title("interpolated points")
plt.xlim([0,10])
```

## Numpy & pynapple

<div class="render-all">

Pynapple timeseries objects (`Tsd`, `TsdFrame`, and `TsdTensor`) behave similarly to numpy arrays. They can be sliced using similar syntax, e.g.:

  `tsd[0:10] # First 10 elements`

Arithmetic operations also behave like numpy:

  `tsd = tsd + 1`

Finally, numpy functions are compatible with pynapple objects, and in many cases will return a pynapple object when the time axis is preserved.

**Question:** Can you compute the average of `rand_tsd` across columns using `np.mean` and print the result?

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

For `TsdFrame` objects with integer column labels, the column labels are ignored when using numpy-like indexing.

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

To access `TsdFrame` objects by integer column names, index using the `loc` method.

**Question:** What happen when you do `tsdframe.loc[0]` and `tsdframe.loc[[0,1]]`?

</div>

```{code-cell} ipython3
print(tsdframe.loc[0])
print(tsdframe[:,0])
```

### get

Sometimes it may be useful to find the nearest data point to a given time stamp. For this, we can use the [`get`](https://pynapple.org/generated/pynapple.TsdTensor.get.html#pynapple.TsdTensor.get) method.

<div class="render-all">

**Question:** Using the [`get`](https://pynapple.org/generated/pynapple.TsdTensor.get.html#pynapple.TsdTensor.get) method, can you get the data point from `spiral_tsd` as close as possible to the time 50.1 seconds?

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

If we want the *index* of the data nearest to some time stamp, we can use [`get_slice`](https://pynapple.org/generated/pynapple.TsdTensor.get_slice.html#pynapple.TsdTensor.get_slice) instead.

<div class="render-all">

**Question:** Using the `get_slice` method, can you get the index of [`spiral_tsd`](https://pynapple.org/generated/pynapple.TsdTensor.get_slice.html#pynapple.TsdTensor.get_slice) as close as possible to the time 50.1 seconds?

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

<div class="render-all">

Metadata can be added to an object using dictionary-like item assignment.

**Question:** Can you add the metadata labels `["burst","random","slow"]` using item assignment to `all_neurons["label"]` and print the result?

</div>

```{code-cell} ipython3
all_neurons["label"] = ["burst", "random", "slow"]
all_neurons
```

#### attribute assignment

<div class="render-all">

Metadata can also be set directly as an attribute to the object.

**Question:** Can you add the values `[1, -1, 1]` to `epochs` as the attribute `epochs.direction`?

</div>

```{code-cell} ipython3
epochs.direction = ["left", "right", "left"]
epochs
```

#### set_info

<div class="render-all">
    
Each object also has the method [`set_info`](https://pynapple.org/generated/pynapple.TsdFrame.set_info.html#pynapple.TsdFrame.set_info) which allows you to set metadata using keyword arguments to the method.

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

Similar to setting metadata, we can retrieve metadata as an attribute (i.e. `all_neurons.label`) or using item access (i.e. `all_neurons['label']`). Additionally we can use [`get_info`](https://pynapple.org/generated/pynapple.TsGroup.get_info.html#pynapple.TsGroup.get_info), a complementary method to `set_info`, to access metadata.

</div>

```{code-cell} ipython3
:tags: [render-all]

all_neurons.get_info('label')
```

<div class="render-all">

We can also access the metadata as a pandas DataFrame using the `metadata` attribute.

</div>

```{code-cell} ipython3
:tags: [render-all]

all_neurons.metadata
```

### slicing with metadata

<div class="render-all">

Metadata can be used to slice pynapple objects by indexing or, for numeric metadata, by the method [`getby_threshold`](https://pynapple.org/generated/pynapple.TsGroup.getby_threshold.html#pynapple.TsGroup.getby_threshold).

**Question:** Can you select only the elements of `all_neurons` with rate below 1Hz?

</div>

```{code-cell} ipython3
print(all_neurons[all_neurons.rate<1.0])

print(all_neurons[all_neurons['rate']<1.0])

print(all_neurons.getby_threshold("rate", 1, "<"))
```

<div class="render-all">

**Question:** Can you select the intervals in `epochs` with a direction of "left"?
</div>

```{code-cell} ipython3
print(epochs[epochs.direction=="left"])
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

<div class="render-all">

The [`count`](https://pynapple.org/generated/pynapple.TsGroup.count.html#pynapple.TsGroup.count) methods allows us to count or bin the number of time points that fall within each window of a given bin size. 

**Question:** Using the `count` method, can you count the number of events within 1 second bins for `all_neurons` over the `ep_signal` intervals?

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

### bin_average

<div class="render-all">

Oftentimes we need to match the sampling rates between different sets of data. Pynapple provides the [`bin_average`](https://pynapple.org/generated/pynapple.TsdFrame.bin_average.html#pynapple.TsdFrame.bin_average) method to downsample data.

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

We may want to find all the intervals where our timeseries data exceeds some value. For 1-dimensional `Tsd` objects, Pynapple provides the [`threshold`](https://pynapple.org/generated/pynapple.Tsd.threshold.html#pynapple.Tsd.threshold) method to limit the `Tsd` above or below a certain value.

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

### to_trial_tensor

<div class="render-all">

We can reshape a `Tsd`, `TsdFrame`, or `TsdTensor` object into a trial-based tensor using the method [`to_trial_tensor`](https://pynapple.org/generated/pynapple.Tsd.to_trial_tensor.html#pynapple.Tsd.to_trial_tensor), where trials are defined by an `IntervalSet`. The resulting tensor is returned as a numpy array, since pynapple objects do not support 2D time axes.

**Question:** Can you create a trial-based tensor of `cos_tsd` using `trials` provided below? Print the resulting shape.

</div>
<div class="render-user">
```{code-cell} ipython3
trials = nap.IntervalSet(start=np.arange(0,100,20),end=np.arange(19.9,100,20))
# compute trial tensor
```
</div>

```{code-cell} ipython3
trials = nap.IntervalSet(start=np.arange(0,100,20),end=np.arange(19.9,100,20))
cos_tsd.to_trial_tensor(trials).shape
```

### trial_count

<div class="render-all">

A similar function to `to_trial_tensor` exists for `Ts` and `TsGroup` objects: [`trial_count`](https://pynapple.org/generated/pynapple.TsGroup.trial_count.html#pynapple.TsGroup.trial_count). Instead of reshaping the time points, however, it counts the number of time points into a trial-based array, akin to a 2D `count`. Similar to `count`, this method requires you to specify a `bin_size`.

**Question:** Can you compute a trial-based count of the spikes in `all_neurons` using `trials` set above and `bin_size=0.1`? Print the resulting shape.

</div>

```{code-cell} ipython3
all_neurons.trial_count(trials, 0.1).shape
```

## First high-level function: compute_tuning_curves

<div class="render-all">

Pynapple provides functions for standard analysis in systems neuroscience. The first function we will try is [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves) that calculates the response of a cell to a particular feature. 

A good practice when using a function for the first time is to check the docstrings to learn how to pass the arguments.

**Question**: Can you examine the docstring of `nap.compute_tuning_curves`?

</div>

```{code-cell} ipython3
print(nap.compute_tuning_curves.__doc__)
```

<div class="render-all">

**Question**: Can you compute the response (i.e. firing rate) of the units in `all_neurons` as function of the feature `cos_tsd` using the function `nap.compute_tuning_curves`? Label the feature as `"cosine"`

</div>
<div class="render-user">
```{code-cell} ipython3
tuning_curves =
```
</div>

```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(all_neurons, cos_tsd, bins=5, feature_names=["cosine"])
tuning_curves
```

<div class="render-all">

The output is an [xarray](https://docs.xarray.dev/en/stable/) object, which acts as a wrapper to numpy arrays with extra utilities. It allows us to specify the coordinates of each dimension as well as attach additional attributes. By labeling our feature(s), we can make the output more readable.

All coordinates can be accessed with the `coords` attribute, which contains the unit number and any feature's values (i.e. center of the feature bins). Each coordinate can also be accessed individually as an attribute.

**Question**: Can you print the underlying the units number, feature value, and bin edges from `tuning_curves`?

</div>

```{code-cell} ipython3
print(tuning_curves.unit.values)
print(tuning_curves.cosine.values)
print(tuning_curves.occupancy)
print(tuning_curves.bin_edges)
print(tuning_curves.fs)
```

<div class="render-all">
    
Xarray objects also supply convenient methods for quickly plotting data.

**Question**: Can you plot the tuning curve of each unit using the `plot` method?

</div>

```{code-cell} ipython3
tuning_curves.plot(hue="unit")
plt.ylabel("firing rate");
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
