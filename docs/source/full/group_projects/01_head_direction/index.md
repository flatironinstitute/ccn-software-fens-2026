# Head-direction cells: from pynapple to nemos GLM

:::{note}
The two parts are independent and can be completed in any order.
:::

In this tutorial, we compare two recording modalities — extracellular electrophysiology and calcium imaging — using the same
head-direction system in the mouse as a common reference. Both datasets contain head-direction cells, but the signal properties
differ: spikes are discrete and fast, while calcium transients are continuous and slow. This gives us a natural testbed
to see how the same analysis workflow adapts to different data types.

**Part 1 — Extracellular recordings** uses spike trains from the anterodorsal thalamic nucleus (ADn) recorded 
with a silicon probe ([Peyrache et al., 2015](https://www.nature.com/articles/nn.3968)).

With **pynapple** we will:
1. Load a NWB file and extract spike times and head-direction
2. Compute head-direction tuning curves
3. Compute cross-correlograms during wake and sleep

With **nemos** we will fit a population GLM to characterize functional connectivity:
1. Build spike-history features with a raw history window and with a `RaisedCosineLogConv` basis
2. Fit a single-neuron GLM and compare the two feature representations
3. Fit a `PopulationGLM` to all neurons simultaneously and visualize the coupling filters

**Part 2 — Calcium imaging** uses fluorescence traces from head-direction cells in the postsubiculum recorded with 
the miniscope ([Skromne-Carrasco et al., 2026](https://www.nature.com/articles/s41586-025-10096-w)).

With **pynapple** we will:
1. Load the calcium NWB file and extract transients and head-direction
2. Compute tuning curves and visualize them
3. Decode head-direction from population activity with `nap.decode_template`

With **nemos** we will fit a GLM suited to continuous data:
1. Select significantly tuned neurons with a Rayleigh test
2. Fit a `PopulationGLM` with a Gaussian observation model
3. Re-fit with a feature mask to remove self-coupling and compare the coupling filters

The pynapple documentation can be found [here](https://pynapple.org) and the nemos documentation [here](https://nemos.readthedocs.io/en/latest/).

```{toctree}
:titlesonly:
part1_extracellular
part2_calcium_imaging
```