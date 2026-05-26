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

This notebook can be downloaded as **{nb-download}`04_place_cells.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

# Analyzing hippocampal place cells with Pynapple and NeMoS

<div class="render-all">
    
In this tutorial we will review more advanced applications of pynapple; tuning curves, signal processing, and decoding; as well as fitting GLMs to the data using NeMoS. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

This notebook is separated into 5 Parts:
1. Data wrangling
2. 1D neural tuning and model fitting
3. Signal processing
4. 2D neural tuning and model fitting
5. Neural decoding

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
```

```{code-cell} ipython3
:tags: [render-all]

# fetch file path
path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
# load data with pynapple
data = nap.load_file(path)
print(data)
```

```{code-cell} ipython3
:tags: [render-all]

position = data["position"]
lfp = data["eeg"][:,0]
spikes = data["units"]
forward_ep = data["forward_ep"]
```

```{code-cell} ipython3
position = position.restrict(forward_ep)
np.any(np.isnan(position))
```

```{code-cell} ipython3
speed = np.abs(position.derivative())
```

```{code-cell} ipython3
ex_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
ex_lfp = lfp.restrict(ex_ep)
ex_position = position.restrict(ex_ep)
ex_speed = speed.restrict(ex_ep)
```

```{code-cell} ipython3
good_spikes = spikes[(spikes.restrict(forward_ep).rate >= 1) & (spikes.restrict(forward_ep).rate <= 10)]
```

```{code-cell} ipython3
place_fields = nap.compute_tuning_curves(good_spikes, position, 50, feature_names=["position"])
```

```{code-cell} ipython3
:tags: [render-all]

neurons = [82, 92, 220]
p = place_fields.sel(unit=neurons).plot(x="position", col="unit")
p.set_ylabels("firing rate (Hz)")
```

```{code-cell} ipython3
speed_fields = nap.compute_tuning_curves(spikes, speed, bins=30, epochs=speed.time_support, feature_names=["speed"])
```

```{code-cell} ipython3
bin_size = 0.01
counts = good_spikes[neurons].count(bin_size, ep=forward_ep)
```

```{code-cell} ipython3
up_position = position.interpolate(counts)
up_speed = speed.interpolate(counts)
```

```{code-cell} ipython3
position_basis = nmo.basis.BSplineEval(n_basis_funcs=12, label="position")
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=6, label="speed")
fig = workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

```{code-cell} ipython3
sample_rate = 1250
```

```{code-cell} ipython3
lfp = lfp.restrict(forward_ep)
position = position.restrict(forward_ep)
```

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp, (6.0, 12.0), fs=sample_rate)
```

```{code-cell} ipython3
phase = np.angle(sp.signal.hilbert(theta_band)) # compute phase with hilbert transform
phase %= 2 * np.pi # wrap to [0,2pi]
theta_phase = nap.Tsd(t=theta_band.t, d=phase, time_support=theta_band.time_support)
theta_phase
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/pc-11.png)
:::
</div>


<div class="render-all">
    
Similar to what we saw in a single run, there is a negative relationship between theta phase and field position, characteristic of phase precession.

</div>

## Part 4: 2D neural tuning and model fitting
### Computing 2D tuning curves: position vs. phase

<div class="render-all">

The scatter plot above can be similarly be represented as a 2D tuning curve over position and phase. We can compute this using the same function, [`nap.compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves), but now passing second input, `features`, as a 2-column `TsdFrame` containing the two target features.

To do this, we'll need to combine `position` and `theta_phase` into a `TsdFrame`. For this to work, both variables must have the same length. Similar to what we did in Part 2, we can achieve this by upsampling `position` to the length of `theta_phase` using the pynapple object method [`interpolate`](https://pynapple.org/generated/pynapple.Tsd.interpolate.html). Once they're the same length, they can be combined into a single `TsdFrame` and used to compute 2D tuning curves.

</div>

#### 4.1 Interpolate `position` to the time points of `theta_phase`.

<div class="render-user"> 
```{code-cell} ipython3
upsampled_pos = 
```
</div>

```{code-cell} ipython3
upsampled_pos = position.interpolate(theta_phase)
```

#### 4.2 Stack `upsampled_pos` and `theta_phase` together into a single [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html)

<div class="render-all">

- For stacking arrays, you can use a numpy function like `np.stack`.
    - Tip: you may need to transpose to make sure time is in the first dimension of the stacked array
- Make sure to name your `TsdFrame` columns `"position"` and `"phase"`
  
</div>

<div class="render-user">  
```{code-cell} ipython3
# store the resulting TsdFrame into the following variable
features = 
```
</div>

```{code-cell} ipython3
feats = np.stack((upsampled_pos.values, theta_phase.values))
features = nap.TsdFrame(
    t=theta_phase.t,
    d=np.transpose(feats),
    time_support=upsampled_pos.time_support,
    columns=["position", "phase"],
)
features
```

#### 4.3 Apply [`nap.compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves) with `features` on our subselected group of units, `good_spikes`

<div class="render-all">

- Use 50 bins for position and 30 bins for theta phase

</div>

<div class="render-user">
```{code-cell} ipython3
tuning_curves =
```
</div>

```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(good_spikes, features, bins=[50,30])
```

<div class="render-all">

We can plot 2D tuning curves for each unit and phase precession in some example units.

</div>

```{code-cell} ipython3
:tags: [render-all]

neurons = [23, 33, 82, 175, 177, 220]
tc_norm = tuning_curves / tuning_curves.max(axis=(1,2))
p = tc_norm.sel(unit=neurons).plot(x="position", y="phase", col="unit", col_wrap=3, size=2, aspect=2)
```

```{code-cell} ipython3
:tags: [hide-input]

p.fig.savefig("../../_static/_check_figs/pc-12.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/pc-12.png)
:::
</div>

<div class="render-all">

You should be able to notice a negative relationship between position and phase, characteristic of phase precession.

</div>

### Estimating 2D tuning curves using 2D basis functions

<div class="render-all">
    
How can we model 2D tuning curves in a GLM? Similar to Part 2, we can define a 2D basis by using [NeMoS basis composition](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html), but instead *multiplying* two basis objects. In fact, we can use both addition and multiplication together to create arbitrarily complex, multidimensional basis objects.

First, we'll create a basis object for theta phase, specifically using [`CyclicBSplineEval`](https://nemos.readthedocs.io/en/latest/generated/basis/nemos.basis.CyclicBSplineEval.html#nemos.basis.BSplineEval). We use this instead of `BSplineBasis` because the phase angle is a circular variable.

</div>

#### 4.4 Instantiate a [`CyclicBSplineEval`](https://nemos.readthedocs.io/en/latest/generated/basis/nemos.basis.CyclicBSplineEval.html#nemos.basis.BSplineEval) basis object for phase, using 10 basis functions.

<div class="render-all">

- Provide the label `"phase"` for the basis.
- If necessary, reinstantiate the basis objects for position, `position_basis`, and speed, `speed_basis`, as you did in **2.6**.

</div>

<div class="render-user">
```{code-cell} ipython3
phase_basis =
```
</div>

```{code-cell} ipython3
phase_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=10, label="phase")
```

#### 4.5 Create the full basis by multiplying `position_basis` and `phase_basis` and adding `speed_basis`.

<div class="render-user">
```{code-cell} ipython3
full_basis =
```
</div>

```{code-cell} ipython3
full_basis = position_basis * phase_basis + speed_basis
full_basis
```

<div class="render-all">

Before we can call `compute_features`, we need to make sure `theta_phase` has the same number of time points as `counts`. Since `theta_phase` has *more* time points than counts, we'll need to *downsample* the number of time points. We can do this using the pynapple object method [`bin_average`](https://pynapple.org/generated/pynapple.Tsd.bin_average.html). This function will average values within a specified bin size. We can achieve the same sampling rate by using the same bin size as we used for our spike counts.

</div>

#### 4.6 Downsample `theta_phase` using `bin_average` and a bin size of 0.01 s.

<div class="render-all">

- If necessary, redefine `up_position` and `up_speed` the same as **2.5**.

</div>

<div class="render-user">
```{code-cell} ipython3
bin_theta =
```
</div>

```{code-cell} ipython3
bin_theta = theta_phase.bin_average(0.01)
```

#### 4.7 Create a design matrix by calling `compute_features` on `full_basis` using `up_position`, `bin_theta`, and `up_speed`

<div class="render-user">
```{code-cell} ipython3
X =
```
</div>

```{code-cell} ipython3
X = full_basis.compute_features(up_position, bin_theta, up_speed)
```

#### 4.8 Fit a GLM by doing the following:

<div class="render-all">

- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix `X` and spike counts `counts` to the glm object.
    - `counts` should have been computed before in **2.4**.

</div>


<div class="render-user">
```{code-cell} ipython3
glm =
```
</div>

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs=dict(tol=10**-12),
    solver_name="LBFGS"
)

glm.fit(X, counts)
```

#### 4.9 Use [`predict`](https://nemos.readthedocs.io/en/latest/generated/glm/nemos.glm.GLM.predict.html#nemos.glm.GLM.predict) to calculated the predicted firing rate of our model. Use the predicted rate to compute predicted tuning curves using [`nap.compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves).

<div class="render-all">

- Remember to convert the predicted firing rate to spikes per second!
- Compute 1D tuning curves for position and speeds in the same way as **2.10**.
- Compute 2D tuning curves for position x phase using `predicted_rate` and the TsdFrame `features`, using 50 bins for position and 30 bins for phase.

</div>

<div class="render-user">
```{code-cell} ipython3
# predict the model's firing rate
predicted_rate =
# compute the 1D tuning curves for position and speed
glm_pf = 
glm_speed = 
# compute 2D tuning curves for position x phase
glm_pos_theta =   
```
</div>

```{code-cell} ipython3
predicted_rate = glm.predict(X) / bin_size

glm_pf = nap.compute_tuning_curves(predicted_rate, position, 50, feature_names=["position"])
glm_speed = nap.compute_tuning_curves(predicted_rate, speed, 30, feature_names=["speed"])
glm_pos_theta = nap.compute_tuning_curves(
    predicted_rate, features, [50, 30], epochs=forward_ep
)
```

<div class="render-all">

We'll use a helper function from NeMoS to compare the predicted tuning curves to those computed from the data

</div>

```{code-cell} ipython3
:tags: [render-all]

from nemos import _documentation_utils as doc_plots
neuron = 82
idx = np.where(glm_pf.unit == neuron)[0][0]
fig = doc_plots.plot_position_phase_speed_tuning(
    place_fields.sel(unit=neuron),
    glm_pf[idx],
    speed_fields.sel(unit=neuron),
    glm_speed[idx],
    tuning_curves.sel(unit=neuron),
    glm_pos_theta[idx],
    )
```

```{code-cell} ipython3
:tags: [hide-input]

fig.savefig("../../_static/_check_figs/pc-13.png")
```

<div class="render-user">
:::{admonition} Figure check
:class: dropdown
![](../../_static/_check_figs/pc-13.png)
:::
</div>


### Bonus Exercise

<div class="render-all">

As an bonus, more open-ended exercise, we can investigate all the scientific decisions that we swept under the rug: should we regularize the model? What basis should we use? Do we need all inputs? If you're feeling ambitious, here are some suggestions to answer these questions:

- Try to fit and compare the results we just obtained with different models: 
  - A model with position as the only predictor.
  - A model with speed as the only predictor.
  - A model with phase as the only predictor
- Introduce L1 (Lasso) regularization and fit models with increasingly large penalty strengths ($\lambda$). Plot the regularization path showing how each coefficient changes with $\lambda$. Identify which coefficients remain non-zero longest as $\lambda$ increases - these correspond to the most informative predictors.

</div>

```{code-cell} ipython3
# bonus exercise
```
