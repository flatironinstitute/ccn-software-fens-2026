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

This notebook can be downloaded as **{nb-download}`02_population_analysis_with_nemos.ipynb`**. See the button at the top right to download as markdown or pdf.
:::


# Fitting a population GLM model with Nemos

<div class="render-all">

This group project is focused on fitting a population GLM model to characterize the functional connectivity.

In the first part, we will use the same dataset as in the previous tutorial, which contains head-direction cells recorded from the antero-dorsal nucleus of the thalamus (ADN) of a mouse during wake and sleep.
In the previous notebook, we characterized the relationship between head-direction cells during wake and sleep. 
Cells that fire together during wake also fire together during sleep and cells that don't fire together during wake don't fire 
together during sleep. The goal here is to characterize this relationship with generalized linear model. 
Since cells have a functional relationship to each other, the activity of one cell should predict the activity of another cell.

In this group project, we will use nemos to do the following tasks:
1. Create spike history features
2. Fit a GLM model to a single neuron
3. Fit a GLM model with basis functions to reduce over-fitting
4. Fit a GLM model to all neurons to learn functional connectivity

In the second part, we will try to apply the same type of analysis to the calcium imaging dataset used in the first tutorial.

Let's start by importing all the packages.

</div>


```{code-cell} ipython3
:tags: [render-all]

import workshop_utils
import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plot style
plt.style.use(nmo.styles.plot_style)
```


## Modelling extracellular spike history effects with GLM 

### Fetching the data

<div class="render-all">

We will use the same dataset as in the previous tutorial, which can be downloaded with the helper function `fetch_data`.
To speed up the session, the following code cell will download the data, load it with pynapple and extract the relevant 
variables:

</div>


```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Mouse32-140822.nwb")
data = nap.load_file(path)
spikes = data["units"]  # Get spike timings
spikes = spikes[(spikes.location=='adn') & (spikes.rate>2.0)]  # Keep only ADN neurons with firing rate > 2Hz
angle = data["ry"] # Get head-direction signal
epochs = data["epochs"] # Get epochs
wake_ep = epochs[epochs.tags=="wake"]
sleep_ep = epochs[epochs.tags=="sleep"]
tuning_curves = nap.compute_tuning_curves(
    data=spikes,
    features=angle, 
    bins=61, 
    epochs = angle.time_support,
    range=(0, 2 * np.pi),
    feature_names = ["angle"]
    )
pref_ang = tuning_curves.idxmax(dim="angle")
spikes.set_info(pref_ang = pref_ang)

```

### Fitting a GLM to a single neuron

<div class="render-all">

**Question : are neurons constantly tuned to head-direction and can we use it to predict the spiking activity of each neuron 
based only on the activity of other neurons?**

To fit the GLM faster, we will use only the first 3 min of wake.

</div>

```{code-cell} ipython3
:tags: [render-all]

# restrict wake epoch to first 3 minutes
wake_ep = nap.IntervalSet(
    start=wake_ep.start[0], end=wake_ep.start[0] + 3 * 60
)
```

<div class="render-all">
To use the GLM, we need first to bin the spike trains. Here we use pynapple and the function `count`.

**Question: can you bin the spike trains in 10 ms bins during the `wake_ep` and call the variable `count`?**

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

Above we defined `pref_ang` as the preferred direction of each neuron. `np.argsort(pref_ang.values)` gives you the order to sort the columns of count.
This is useful to visualize the activity of neurons based on their preferred direction.

</div>

```{code-cell} ipython3
:tags: [render-all]

count = count[:, np.argsort(pref_ang.values)]
```

<div class="render-all">

It's time to use NeMoS. Our end goal is to estimate the pairwise interaction between neurons.
This can be quantified with a GLM if we use the recent population spike history to predict the current time step.

To simplify our life, let's see first how we can model spike history effects in a single neuron.
The simplest approach is to use counts in fixed length window $i$, $y_{t-i}, \dots, y_{t-1}$ to predict the next
count $y_{t}$. 

Before starting the analysis, let's 

- **select a neuron (first column is good) from the `count` object (call the variable `neuron_count`)** 
- **Select the first 1.2 seconds of wake_ep for visualization. (call the epoch `epoch_one_spk`).**

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

#### Features Construction

<div class="render-all">

Let's fix the spike history window size that we will use as predictor meaning how far back in time we want to look to predict the current rate.

Let's :
- Fix a history window of 800ms (0.8 seconds).
- Plot the result using `doc_plots.plot_history_window`

</div>

```{code-cell} ipython3
:tags: [render-all]

# set the size of the spike history window in seconds
window_size_sec = 0.8

fig = doc_plots.plot_history_window(neuron_count, epoch_one_spk, window_size_sec);
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../_static/_check_figs/02-01.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-01.png)
:::
</div>

<div class="render-all">

For each time point, we shift our window one bin at the time and vertically stack the spike count history in a matrix.
Each row of the matrix will be used as the predictors for the rate in the next bin (red narrow rectangle in
the figure).

</div>

```{code-cell} ipython3
:tags: [render-all]

doc_plots.run_animation(neuron_count, epoch_one_spk.start[0])
```

<div class="render-all">

If $t$ is smaller than the window size, we won't have a full window of spike history for estimating the rate.
One may think of padding the window (with zeros for example) but this may generate weird border artifacts.
To avoid that, we can simply restrict our analysis to times $t$ larger than the window and NaN-pad earlier
time-points;

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

NeMoS NaN pads if there aren't enough samples to predict the counts.
 
</div>

```{code-cell} ipython3
:tags: [render-all]

# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0]) 
```

<div class="render-all">

The binned counts originally have shape "number of samples", we should check that the
dimension are matching our expectation

</div>

```{code-cell} ipython3
:tags: [render-all]

print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")
print(f"Feature shape: {input_feature.shape}")
```

<div class="render-all">

We can visualize the output for a few time bins

</div>

```{code-cell} ipython3
:tags: [render-all]

suptitle = "Input feature: Count History"
neuron_id = 0
fig = workshop_utils.plot_features(input_feature, count.rate, suptitle)
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../_static/_check_figs/02-02.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-02.png)
:::
</div>


<div class="render-all">

As you may see, the time axis is backward, this happens because under the hood, the basis is using the convolution operator which flips the time axis.
This is equivalent, as we can interpret the result as how much a spike will affect the future rate.
In the previous tutorial our feature was 1-dimensional (just the current), now
instead the feature dimension is 80, because our bin size was 0.01 sec and the window size is 0.8 sec.
We can learn these weights by maximum likelihood by fitting a GLM.

</div>


#### Fitting a single neuron model

<div class="render-all">

When working a real dataset, it is good practice to train your models on a chunk of the data and
use the other chunk to assess the model performance. This process is known as "cross-validation".
There is no unique strategy on how to cross-validate your model; What works best
depends on the characteristic of your data (time series or independent samples,
presence or absence of trials), and that of your model. Here, for simplicity use the first
half of the wake epochs for training and the second half for testing. This is a reasonable
choice if the statistics of the neural activity does not change during the course of
the recording. We will learn about better cross-validation strategies with other
examples.

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

The weights represent the effect of a spike at time lag $i$ on the rate at time $t$. The next cell display the learned weights.
The model should be called `model` from the previous cell.

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

fig.savefig("../../_static/_check_figs/02-03.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-03.png)
:::
</div>

<div class="render-all">

The response in the previous figure seems noise added to a decay, therefore the response
can be described with fewer degrees of freedom. In other words, it looks like we
are using way too many weights to describe a simple response.
If we are correct, what would happen if we re-fit the weights on the other half of the data?

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

fig.savefig("../../_static/_check_figs/02-04.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-04.png)
:::
</div>

<div class="render-all">

What can we conclude?

The fast fluctuations are inconsistent across fits, indicating that
they are probably capturing noise, a phenomenon known as over-fitting;
On the other hand, the decaying trend is fairly consistent, even if
our estimate is noisy. You can imagine how things could get
worst if we needed a finer temporal resolution, such 1ms time bins
(which would require 800 coefficients instead of 80).
What can we do to mitigate over-fitting now?

</div>

#### Reducing feature dimensionality

<div class="render-all">
Let's see how to use NeMoS' `basis` module to reduce dimensionality and avoid over-fitting!
For history-type inputs, we'll use again the raised cosine log-stretched basis,
[Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003).

</div>

```{code-cell} ipython3
:tags: [render-all]

fig = doc_plots.plot_basis()
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../_static/_check_figs/02-05.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-05.png)
:::
</div>

:::{note}

We provide a handful of different choices for basis functions, and
selecting the proper basis function for your input is an important
analytical step. We will eventually provide guidance on this choice, but
for now we'll give you a decent choice.
:::

<div class="render-all">

We can initialize the `RaisedCosineLogConv` by providing the number of basis functions 
and the window size for the convolution. With more basis functions, we'll be able to represent 
the effect of the corresponding input with the higher precision, at the cost of adding additional parameters.

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

Our spike history predictor was huge: every possible 80 time point chunk of the
data, for $144 \cdot 10^4$ total numbers. By using this basis set we can instead reduce
the predictor to 8 numbers for every 80 time point window for $144 \cdot 10^3$ total
numbers, an order of magnitude less. With 1ms bins we would have
achieved 2 order of magnitude reduction in input size. This is a huge benefit
in terms of memory allocation and, computing time. As an additional benefit,
we will reduce over-fitting.

Let's see our basis in action. We can "compress" spike history feature by convolving the basis
with the counts (without creating the large spike history feature matrix).
This can be performed in NeMoS by calling the `compute_features` method of basis.

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

Let’s focus on two small time windows and visualize the features, which result from convolving the counts with the basis elements.

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

fig.savefig("../../_static/_check_figs/02-06.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-06.png)
:::
</div>

#### Fit a GLM with basis features with reduced dimensionality

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

We can plot the resulting response, noting that the weights we just learned needs to be "expanded" back
to the original `window_size` dimension by multiplying them with the basis kernels.
We have now 8 coefficients,

</div>

```{code-cell} ipython3
:tags: [render-all]

print(model_basis.coef_)
```

<div class="render-all">

In order to get the response of a neuron in response to its history, we need to multiply the coefficients by their corresponding
basis function, and sum them.

Let's do that now. We can reconstruct the history filter by multiplying the basis kernels with the learned coefficients.

We can get the basis kernels by calling the `evaluate_on_grid` method of the basis object.

Then we can multiply the basis kernels with the coefficients using `np.matmul`.

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

Let's check if our new estimate does a better job in terms of over-fitting. We can do that
by visual comparison, as we did previously. Let's fit the second half of the dataset.

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

fig.savefig("../../_static/_check_figs/02-07.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-07.png)
:::
</div>

<div class="render-all">

Let's see if the basis model improves prediction of the firing rate. Here we will compare the firing rate predicted
by the two models on the whole dataset. The model should be called `model` and `model_basis` from the previous cells.

**Question: Can you:**
- Predict the rates from `model` and `model_basis`? Call it `rate_history` and `rate_basis`.

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

Let's plot the predicted rates over a short window not used for training.

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

fig.savefig("../../_static/_check_figs/02-08.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-08.png)
:::
</div>

### All-to-all Connectivity

<div class="render-all">

The same approach can be applied to the whole population. Now the firing rate of a neuron
is predicted not only by its own count history, but also by the rest of the
simultaneously recorded population. We can convolve the basis with the counts of each neuron
to get an array of predictors of shape, `(num_time_points, num_neurons * num_basis_funcs)`.

</div>

#### Preparing the features

<div class="render-all">

**Question: Can you:**
- Re-define the basis?
- Convolve all counts? Call the output in `convolved_count`.
- Print the output shape?

Since this time we are convolving more than one neuron, we need to reset the expected input shape. 
This can be done by passing the population counts to the `set_input_shape` method.

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

Check the dimension to make sure it make sense.

Shape should be `(n_samples, n_basis_func * n_neurons)`

</div>

```{code-cell} ipython3
:tags: [render-all]

print(f"Convolved count shape: {convolved_count.shape}")
```

(head-direction-fit-full)=
#### Fitting the Model

<div class="render-all">

This is an all-to-all neurons model.
We are using the class [`PopulationGLM`](https://nemos.readthedocs.io/en/latest/generated/glm/nemos.glm.PopulationGLM.html) to fit the whole population at once.


Once we condition on past activity, log-likelihood of the population is the sum of the log-likelihood
of individual neurons. Maximizing the sum (i.e. the population log-likelihood) is equivalent to
maximizing each individual term separately (i.e. fitting one neuron at the time).

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

#### Comparing model predictions.

<div class="render-all">

Predict the rate (counts are already sorted by tuning prefs)

**Question: Can you:**
- Predict the firing rate of each neuron? Call it `predicted_firing_rate`.
- Convert the rate from spike/bin to spike/sec?

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

Now we can visualize the tuning curves predicted by the model as well as the real tuning curves and the predicted firing rate.

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

fig.savefig("../../_static/_check_figs/02-09.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-09.png)
:::
</div>

<div class="render-all">

Let's see if our firing rate predictions improved and in what sense.

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

fig.savefig("../../_static/_check_figs/02-10.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-10.png)
:::
</div>

#### Visualizing the connectivity

<div class="render-all">

Finally, we can extract and visualize the pairwise interactions between neurons.

**Question: Can you extract the weights and store it in a `(n_neurons, n_neurons, n_basis_funcs)` array? 

</div>

```{code-cell} ipython3
:tags: [render-all]

# original shape of the weights
print(f"GLM coeff: {model.coef_.shape}")
```

<div class="render-all">

You can use the `split_by_feature` method of `basis` for this. It will reshape the coefficient vector into a 3D array.

![Reshape coefficients](../../_static/coeff_reshape.png)

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

The shape is `(sender_neuron, num_basis, receiver_neuron)`.

Let's reconstruct the coupling filters by multiplying the weights with the basis functions. 
Here we use `np.einsum` for that. It's a powerful function for summing products of arrays over specified axes.
In this case, the operation is :
(sender_neuron, num_basis, receiver_neuron) x (time lag, num_basis) -> (sender_neuron, receiver_neuron, time lag)

</div>

```{code-cell} ipython3
:tags: [render-all]

responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

print(responses.shape)
```

<div class="render-all">

Finally, we can visualize the pairwise interactions by plotting
all the coupling filters.

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

fig.savefig("../../_static/_check_figs/02-11.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/02-11.png)
:::
</div>

<div class="render-all">

These coupling filters represent the influence of one neuron on another over time. 
They have been sorted based on the preferred head-direction of each neuron.
Note that those neurons are not synaptically connected, but they have a functional relationship based on their tuning 
to head-direction.
</div>

