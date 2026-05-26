#!/usr/bin/env python3

import pynapple as nap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray
from typing import Union, Optional, List
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation
import seaborn as sns
from nemos import _documentation_utils as doc_plots

__all__ = [
    "plot_features",
    "animate_1d_convolution",
    "plot_head_direction_tuning_model",
    "plot_heatmap_cv_results",
    "plot_position_speed",
    "plot_position_speed_tuning",
    "plot_place_fields",
    "plot_pos_speed_bases",
    "visualize_intervals",
    "plot_current_history_features",
    "current_injection_plot",
    "plot_basis_filter",
    "plot_coupling_filters",
]


def plot_features(
    input_feature: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor, NDArray],
    sampling_rate: float,
    suptitle: str,
    n_rows: int = 20,
):
    """
    Plot feature matrix.

    Parameters
    ----------
    input_feature:
        The (num_samples, n_neurons, num_feature) feature array.
    sampling_rate:
        Sampling rate in hz.
    n_rows:
        Number of rows to plot.
    suptitle:
        Suptitle of the plot.

    Returns
    -------

    """
    input_feature = np.squeeze(input_feature).dropna()
    window_size = input_feature.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    time = np.arange(0, window_size) / sampling_rate
    for k in range(n_rows):
        ax = plt.subplot(n_rows, 1, k + 1)
        plt.step(time, input_feature[k].squeeze(), where="post")

        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.axvspan(0, time[-1], alpha=0.4, color="orange")
        ax.set_yticks([])
        if k != n_rows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("lag (sec)")
        if k in [0, n_rows - 1]:
            ax.set_ylabel("$t_{%d}$" % (window_size + k), rotation=0)

    plt.tight_layout()
    return fig


class Plot1DConvolution:
    """
    Class to plot an animation of convolving some 1D kernel with some Tsd array.

    Parameters
    ----------
    tsd :
        The Tsd object to convolve with the kernel.
    kernel :
        The 1D kernel to convolve with the array.
    index :
        The time index. Taken from the Tsd object if not provided.
    start :
        The index along the x-axis to start the animation. Defaults to the start of the window.
    interval :
        The interval between frames in milliseconds.
    figsize :
        The figure size.
    ylim :
        The y-axis limits.
    xlabel :
        The x-axis label.
    ylabel :
        The y-axis label.
    tsd_label :
        The legend label for the Tsd array
    kernel_label :
        The legend label for the kernel
    conv_label :
        The legend label for the convolution output
    split_kernel_yaxis :
        Whether or not to have a separate y-axis (i.e. use twinx()) for plotting the kernel. Useful if the kernel is magnitudes smaller/larger than the Tsd.
    """

    def __init__(
        self,
        tsd: nap.Tsd,
        kernel: NDArray,
        index: NDArray = None,
        start: int = 0,
        interval: float = 100,
        figsize: tuple = (10, 3),
        ylim: float = None,
        xlabel: str = "Time (s)",
        ylabel: str = "Count",
        tsd_label: str = "original array",
        kernel_label: str = "kernel",
        conv_label: str = "convolution",
        split_kernel_yaxis: bool = False,
    ):
        self.tsd = tsd
        self.kernel = kernel
        if index is None:
            self.index = tsd.index.values
        else:
            self.index = index
        self.start = start
        self.conv = tsd.convolve(kernel)
        self.conv_viz = np.zeros_like(tsd)
        self.frames = len(tsd) - start
        self.interval = interval
        if ylim is None:
            if split_kernel_yaxis:
                ymin = np.min((self.tsd.min(), self.conv.min()))
                ymax = np.max((self.tsd.max(), self.conv.max()))
            else:
                ymin = np.min((self.tsd.min(), self.conv.min(), self.kernel.min()))
                ymax = np.max((self.tsd.max(), self.conv.max(), self.kernel.max()))
            ylim = (ymin, ymax)
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tsd_label = tsd_label
        self.kernel_label = kernel_label
        self.conv_label = conv_label
        self.split_kernel_yaxis = split_kernel_yaxis
        (
            self.fig,
            self.kernel_line,
            self.conv_line,
            self.conv_area,
            self.top_idx_line,
            self.bottom_idx_line,
        ) = self.setup(figsize)

    def setup(self, figsize):
        """
        Initialization of the plot.
        """
        # initial placement of kernel
        kernel_full = np.zeros_like(self.tsd)
        kidx, kmid = self.kernel_bounds(0)
        if np.any(kidx):
            kernel_full[kidx] = self.kernel[: len(kidx)]

        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

        ### top plot ###
        ax = axs[0]
        # this is fixed
        ax.plot(self.index, self.tsd, label=self.tsd_label)

        # initial visible convolution output and top center line
        if kmid >= 0:
            self.conv_viz[: kmid + 1] = self.conv[: kmid + 1]
            cx = self.index[kmid]
        else:
            cx = self.index[0]
        top_idx_line = ax.plot((cx, cx), self.ylim, "--", color="black", alpha=0.5)[0]

        # initial filled area
        conv_area = ax.fill_between(
            self.index,
            np.zeros_like(self.tsd),
            self.tsd * kernel_full.values,
            alpha=0.5,
            color="green",
        )

        # initial kernel plot
        if self.split_kernel_yaxis:
            ax = ax.twinx()
            ax.set_ylabel(self.kernel_label)
            ax.set_ylim((kernel_full.min(), kernel_full.max()))
        kernel_line = ax.plot(
            self.index, kernel_full, color="orange", label=self.kernel_label
        )[0]

        ### bottom plot ###
        ax = axs[1]
        # initial convolution output and bottom plot center line
        conv_line = ax.plot(
            self.index, self.conv_viz, color="green", label=self.conv_label
        )[0]
        bottom_idx_line = ax.plot((cx, cx), self.ylim, "--", color="black", alpha=0.5)[
            0
        ]

        ax.set_ylim(self.ylim)

        fig.legend()
        fig.supxlabel(self.xlabel)
        fig.supylabel(self.ylabel)
        plt.tight_layout()

        return fig, kernel_line, conv_line, conv_area, top_idx_line, bottom_idx_line

    def update(self, frame):
        if frame > 0:
            # place kernel at shifted location based on frame number
            kernel_full = np.zeros_like(self.tsd)
            kidx, kmid = self.kernel_bounds(frame)
            kernel_full[kidx] = self.kernel[: len(kidx)]
            self.kernel_line.set_ydata(kernel_full)

            # update visible convolution output
            if kmid >= 0:
                self.conv_viz[kmid] = self.conv[kmid]
                self.conv_line.set_ydata(self.conv_viz)
                self.top_idx_line.set_xdata((self.index[kmid], self.index[kmid]))
                self.bottom_idx_line.set_xdata((self.index[kmid], self.index[kmid]))

            # update filled area
            self.conv_area.set_data(
                self.index, np.zeros_like(self.tsd), self.tsd * kernel_full.values
            )

    def run(self):
        anim = FuncAnimation(
            self.fig, self.update, self.frames, interval=self.interval, repeat=True
        )
        plt.close(self.fig)
        return anim

    def kernel_bounds(self, frame):
        # kernel bounds set to the left of the frame index and start location
        kmin = frame + self.start - len(self.kernel)
        kmax = frame + self.start

        # kernel indices no less than 0 and no more than the length of the Tsd
        kidx = np.arange(np.max((kmin, 0)), np.min((kmax, len(self.tsd))))

        # convolution output w.r.t. the midpoint of where the kernel is placed
        kmid = kmin + np.floor(len(self.kernel) / 2).astype(int)

        return kidx, kmid


def animate_1d_convolution(tsd: nap.Tsd, kernel: NDArray, **kwargs):
    """
    Animate the convolution of a 1D kernel with some Tsd array.

    Parameters
    ----------
    tsd : nap.Tsd
        The Tsd object to be convolved.
    kernel : np.ndarray
        The 1D kernel to convolve with the array.
    **kwargs
        Additional keyword arguments to pass to Plot1DConvolution.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
    """
    anim = Plot1DConvolution(tsd, kernel, **kwargs)
    return anim.run()


def plot_head_direction_tuning_model(
    tuning_curves: xarray.DataArray,
    spikes: nap.TsGroup,
    angle: nap.Tsd,
    predicted_firing_rate: Optional[nap.TsdFrame] = None,
    pref_ang: Optional[pd.Series] = None,
    model_tuning_curves: Optional[pd.DataFrame] = None,
    threshold_hz: int = 1,
    start: float = 8910,
    end: float = 8960,
    cmap_label="hsv",
    figsize=None,
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:
        The tuning curve xarray.
    spikes:
        The spike times.
    angle:
        The heading angles.
    predicted_firing_rate:
        The time series of the predicted rate.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------
    fig:
        The figure.
    """
    second_coord_name = list(tuning_curves.coords)[1]
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.sel(unit=index_keep)
    if pref_ang is None:
        pref_ang = tuning_curves.idxmax(dim=second_coord_name)
    pref_ang = pref_ang.sel(unit=index_keep)
    # spike_tsd = (
    #     spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang.values)
    # )

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    # unq_angles = np.unique(pref_ang.values)
    unq_angles = np.sort(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (
        unq_angles[-1] - unq_angles[0]
    )

    n_rows = 3
    if predicted_firing_rate is not None:
        n_rows += 1
    if model_tuning_curves is not None:
        model_feature_dim_name = list(model_tuning_curves.dims)[1]
        model_tuning_curves = model_tuning_curves.sel(unit=index_keep)
        n_rows += 1
    if figsize is None:
        figsize = [12, 6]
        if n_rows == 5:
            figsize[1] += 2
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid(
        (n_rows, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")
    ax.set_xlim(8910, 8960)

    ax = plt.subplot2grid(
        (n_rows, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Activity")
    for i, neu_idx in enumerate(pref_ang.unit.values[np.argsort(pref_ang).values]):
        spike_tsd = spikes[neu_idx].restrict(plot_ep).fillna(i)
        ax.plot(
            spike_tsd,
            "|",
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(8910, 8960)
    curr_row = 2

    if predicted_firing_rate is not None:
        ax = plt.subplot2grid(
            (n_rows, n_subplots),
            loc=(curr_row, 0),
            rowspan=1,
            colspan=n_subplots,
            fig=fig,
        )
        curr_row += 1
        ax.set_title("Neural Firing Rate")
        fr = predicted_firing_rate.restrict(plot_ep).d
        fr = fr.T / np.max(fr, axis=1)
        ax.imshow(fr[::-1], cmap="Blues", aspect="auto")
        ax.set_ylabel("Sorted Neurons")
        ax.set_xlabel("Time (s)")
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels([8910, 8920, 8930, 8940, 8950, 8960])
        ax.set_xlim(0, 5000)

    sorted_indices = pref_ang.argsort()
    sorted_units = pref_ang.unit.values[sorted_indices]

    for i, neu_idx in enumerate(sorted_units):
        ax = plt.subplot2grid(
            (n_rows, n_subplots),
            loc=(curr_row + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves[second_coord_name].values,
            np.zeros(tuning_curves.shape[1]),
            tuning_curves.sel(unit=neu_idx).values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    curr_row += 1

    if model_tuning_curves is not None:
        for i, ang in enumerate(unq_angles):
            neu_idx = np.argsort(pref_ang)[i].unit

            ax = plt.subplot2grid(
                (n_rows, n_subplots),
                loc=(curr_row + i // n_subplots, i % n_subplots),
                rowspan=1,
                colspan=1,
                fig=fig,
                projection="polar",
            )
            ax.fill_between(
                tuning_curves[second_coord_name].values,
                np.zeros(tuning_curves.shape[1]),
                model_tuning_curves[neu_idx].values,
                color=cmap(relative_color_levs[i]),
                alpha=0.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    return fig


def facet_heatmap(index, columns, values, data, **heatmap_kwargs):
    data = data.pivot(index=index, columns=columns, values=values)
    ax = sns.heatmap(data, **heatmap_kwargs)
    doc_plots.highlight_max_cell(data, ax)


def format_regularizer(x):
    reg = str(x["param_glm__regularizer"]).split("(")[0]
    reg_str = x["param_glm__regularizer_strength"]
    if not np.isnan(reg_str):
        reg += f"({reg_str})"
    return reg


def plot_heatmap_cv_results(
    cv_df,
    facet_col=None,
    rows="param_basis__basis1__n_basis_funcs",
    columns="param_basis__basis2",
    values="mean_test_score",
    fmt=".3f",
):
    if "param_glm" not in cv_df.columns:
        try:
            cv_df["param_glm"] = cv_df.apply(format_regularizer, 1)
        except KeyError:
            pass
    fig = sns.FacetGrid(cv_df, col=facet_col, height=5)
    vmin = cv_df[values].min()
    vmax = cv_df[values].max()
    fig.map_dataframe(
        facet_heatmap,
        index=rows,
        columns=columns,
        values=values,
        annot=True,
        square=True,
        vmin=vmin,
        vmax=vmax,
        linecolor="white",
        linewidth=0.5,
        fmt=fmt,
    )


def _analyze_speed(speed: pd.DataFrame, position: nap.Tsd):
    bins = np.linspace(np.min(position), np.max(position), 20)
    idx = np.digitize(position.values, bins)
    mean_speed = np.array([np.mean(speed[idx == i]) for i in np.unique(idx)])
    std_speed = np.array([np.std(speed[idx == i]) for i in np.unique(idx)])
    return bins, mean_speed, std_speed


def plot_position_speed(
    position: nap.Tsd,
    speed: nap.Tsd,
    position_tuning: xarray.DataArray,
    speed_tuning: xarray.DataArray,
    neuron_id: Union[int, List[int]],
):
    if not hasattr(neuron_id, "__iter__"):
        neuron_id = [neuron_id]
    fig = plt.figure(figsize=(6 * len(neuron_id), 7))
    gs = plt.GridSpec(2, 2 * len(neuron_id), wspace=0.3, hspace=0.35)
    if hasattr(position_tuning, "distance") is False:
        # for place cell notebook
        position_tuning["distance"] = position_tuning.position
    pos_range = (
        np.min([position.min(), position_tuning.distance.min()]),
        np.max([position.max(), position_tuning.distance.max()]),
    )
    speed_range = (
        np.min([speed.min(), speed_tuning.speed.min()]),
        np.max([speed.max(), speed_tuning.speed.max()]),
    )
    fr_range = (
        np.min([position_tuning.min(), speed_tuning.min()]),
        np.max([position_tuning.max(), speed_tuning.max()]),
    )

    ax = fig.add_subplot(gs[0, 0])
    bins, mean_speed, std_speed = _analyze_speed(speed, position)
    ax.plot(bins, mean_speed)
    ax.fill_between(
        bins,
        mean_speed - std_speed,
        mean_speed + std_speed,
        alpha=0.1,
    )
    ax.set(
        xlabel="Position (cm)",
        ylabel="Speed (cm/s)",
        title="Animal behavior",
        xlim=pos_range,
        ylim=speed_range,
    )
    for i, n in enumerate(neuron_id):
        ax = fig.add_subplot(gs[1, 2 * i])
        ax.fill_between(
            position_tuning.sel({"unit": n}).distance.data,
            np.zeros(len(position_tuning.distance)),
            position_tuning.sel({"unit": n}).data,
        )
        ax.set(
            xlabel="Position (cm)",
            ylabel="Firing rate (Hz)",
            title="Position tuning",
            xlim=pos_range,
            ylim=fr_range,
        )
        ax.text(
            1.2, 1.1, f"Neuron {n}", transform=ax.transAxes, size="x-large", ha="center"
        )

        ax = fig.add_subplot(gs[1, 2 * i + 1])
        ax.fill_between(
            speed_tuning.sel({"unit": n}).speed.data,
            np.zeros(len(speed_tuning.speed)),
            speed_tuning.sel({"unit": n}).data,
        )
        ax.set(
            ylabel="Firing rate (Hz)",
            xlabel="Speed (cm/s)",
            title="Speed tuning",
            xlim=speed_range,
            ylim=fr_range,
        )
    return fig


def plot_position_speed_tuning(
    position_tuning: xarray.DataArray,
    speed_tuning: xarray.DataArray,
    model_position_tuning: Optional[xarray.DataArray] = None,
    model_speed_tuning: Optional[xarray.DataArray] = None,
):
    fig = plt.figure(figsize=(6 * position_tuning.shape[0], 3))
    gs = plt.GridSpec(1, 2 * position_tuning.shape[0], wspace=0.3, hspace=0.35)
    pos_ax = None
    speed_ax = None
    for i, n in enumerate(position_tuning.unit):
        ax = fig.add_subplot(gs[0, 2 * i], sharey=pos_ax, sharex=pos_ax)
        position_tuning.sel({"unit": n}).plot(linestyle="--", ax=ax)
        if model_position_tuning is not None:
            model_position_tuning.sel({"unit": n}).plot(ax=ax)
        ax.set(
            xlabel="Position (cm)", ylabel="Firing rate (Hz)", title="Position tuning"
        )
        ax.text(1, 1.2, f"Neuron {n.data}", transform=ax.transAxes, size="x-large")
        if pos_ax is None:
            pos_ax = ax

        ax = fig.add_subplot(gs[0, 2 * i + 1], sharex=speed_ax, sharey=pos_ax)
        speed_tuning.sel({"unit": n}).plot(linestyle="--", ax=ax)
        if model_speed_tuning is not None:
            model_speed_tuning.sel({"unit": n}).plot(ax=ax)
        ax.set(ylabel="Firing rate (Hz)", xlabel="Speed (cm/s)", title="Speed tuning")
        if speed_ax is None:
            speed_ax = ax
    return fig


def plot_place_fields(place_fields: xarray.DataArray, highlight_neurons=[92, 82, 220]):
    # for display purposes, sort place fields by location
    place_fields = place_fields.sortby(place_fields.argmax("distance"))
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(len(place_fields.unit), 1)
    for i, n in enumerate(place_fields):
        plt.subplot(gs[i, 0])
        if n.unit.data in highlight_neurons:
            c = "r"
        else:
            c = "C0"
        plt.fill_between(
            place_fields.distance.data,
            np.zeros(len(place_fields.distance)),
            n.data,
            facecolor=c,
        )
        if i < len(place_fields.unit) - 1:
            plt.xticks([])
        else:
            plt.xlabel("Position (cm)")
        plt.yticks([])


def plot_pos_speed_bases(position_basis, speed_basis):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, bas, t in zip(axes, [position_basis, speed_basis], ["Position", "Speed"]):
        time, kernel = bas.evaluate_on_grid(100)
        ax.plot(time, kernel)
        ax.set_title(t)
    return fig


def visualize_intervals(eps):
    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(eps)))
    ypos = np.linspace(0, 1, len(eps) + 2)[1:-1]
    lgt = 1 / (len(eps) ** 2)
    for i, ep in enumerate(eps):
        for s, e in ep.values:
            plt.axvspan(
                xmin=s, xmax=e, ymin=ypos[i] - lgt, ymax=ypos[i] + lgt, color=colors[i]
            )
    plt.xlabel("Time (s)")
    plt.ylabel("Epochs")


def plot_current_history_features(
    current, features, basis, window_duration_sec, interval=nap.IntervalSet(462.77, 463)
):
    fig, axes = plt.subplots(2, 3, sharey="row", figsize=(8, 3.5))
    time, basis = basis.evaluate_on_grid(basis.window_size)
    time *= window_duration_sec
    current = current.restrict(interval)
    features = (
        features.restrict(interval) / features.restrict(interval).max(0) * current.max()
    )
    for ax in axes[1, :]:
        ax.plot(current, "k--")
        ax.set_xlabel("Time (sec")
    axes[0, 0].plot(time, basis, alpha=0.1)
    axes[0, 0].plot(time, basis[:, 0], "C0", alpha=1)
    axes[0, 0].set_ylabel("Amplitude (A.U.)")
    axes[1, 0].plot(features[:, 0])
    axes[1, 0].set_ylabel("Current")
    axes[0, 0].set_title("Feature 1")
    axes[1, 1].plot(features[:, -1], f"C{basis.shape[1] - 1}")
    axes[0, 1].plot(time, basis, alpha=0.1)
    axes[0, 1].plot(time, basis[:, -1], f"C{basis.shape[1] - 1}", alpha=1)
    axes[0, 1].set_title(f"Feature {basis.shape[1]}")
    axes[0, 2].plot(time, basis)
    axes[1, 2].plot(features)
    axes[0, 2].set_title("All features")


def current_injection_plot(
    current: nap.Tsd,
    spikes: nap.TsGroup,
    firing_rate: nap.TsdFrame,
    *predicted_firing_rates: Optional[nap.TsdFrame],
):
    ex_intervals = current.threshold(0.0).time_support

    # define plotting parameters
    # colormap, color levels and transparency level
    # for the current injection epochs
    cmap = plt.get_cmap("autumn")
    color_levs = [0.8, 0.5, 0.2]
    alpha = 0.4

    fig = plt.figure(figsize=(7, 7))
    # first row subplot: current
    ax = plt.subplot2grid((4, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(current, color="grey")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Injected Current")
    ax.set_xticklabels([])
    ax.axvspan(
        ex_intervals.loc[0, "start"],
        ex_intervals.loc[0, "end"],
        alpha=alpha,
        color=cmap(color_levs[0]),
    )
    ax.axvspan(
        ex_intervals.loc[1, "start"],
        ex_intervals.loc[1, "end"],
        alpha=alpha,
        color=cmap(color_levs[1]),
    )
    ax.axvspan(
        ex_intervals.loc[2, "start"],
        ex_intervals.loc[2, "end"],
        alpha=alpha,
        color=cmap(color_levs[2]),
    )

    # second row subplot: response
    resp_ax = plt.subplot2grid((4, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
    resp_ax.plot(firing_rate, color="k", label="Observed firing rate")
    if predicted_firing_rates:
        if len(predicted_firing_rates) > 1:
            lbls = [" (current history)", " (instantaneous only)"]
        else:
            lbls = [""]
        for pred_fr, style, lbl in zip(predicted_firing_rates, ["-", "--"], lbls):
            resp_ax.plot(
                pred_fr,
                linestyle=style,
                color="tomato",
                label=f"Predicted firing rate{lbl}",
            )
    resp_ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10, label="Observed spikes")
    resp_ax.set_ylabel("Firing rate (Hz)")
    resp_ax.set_xlabel("Time (s)")
    resp_ax.set_title("Neural response", y=0.95)
    resp_ax.axvspan(
        ex_intervals.loc[0, "start"],
        ex_intervals.loc[0, "end"],
        alpha=alpha,
        color=cmap(color_levs[0]),
    )
    resp_ax.axvspan(
        ex_intervals.loc[1, "start"],
        ex_intervals.loc[1, "end"],
        alpha=alpha,
        color=cmap(color_levs[1]),
    )
    resp_ax.axvspan(
        ex_intervals.loc[2, "start"],
        ex_intervals.loc[2, "end"],
        alpha=alpha,
        color=cmap(color_levs[2]),
    )
    ylim = resp_ax.get_ylim()

    # third subplot: zoomed responses
    zoom_axes = []
    for i in range(len(ex_intervals)):
        interval = ex_intervals.loc[[i]]
        ax = plt.subplot2grid((4, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
        ax.plot(firing_rate.restrict(interval), color="k")
        ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
        if predicted_firing_rates:
            for pred_fr, style in zip(predicted_firing_rates, ["-", "--"]):
                ax.plot(pred_fr.restrict(interval), linestyle=style, color="tomato")
        else:
            ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        for spine in ["left", "right", "top", "bottom"]:
            color = cmap(color_levs[i])
            # add transparency
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)
        zoom_axes.append(ax)

    resp_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        bbox_transform=zoom_axes[1].transAxes,
    )


def plot_basis_filter(basis, model, current_history_duration_sec=0.2):
    """Visualize the model's learned filter."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    time, kernel = basis.evaluate_on_grid(200)
    time *= current_history_duration_sec
    axes[0].plot(time, kernel)
    axes[0].set(title="Basis functions", xlabel="Time (sec)", ylabel="Amplitude (A.U.)")
    axes[1].bar(np.arange(len(model.coef_)), model.coef_)
    axes[1].set(title="Coefficient Weights", xlabel="Basis number")
    axes[2].plot(time, kernel * model.coef_)
    axes[2].set(title="Weighted basis functions", xlabel="Time (sec)")
    axes[2].axhline(0, c="k", linestyle="--")
    axes[3].plot(time, np.matmul(kernel, model.coef_))
    axes[3].axhline(0, c="k", linestyle="--")
    axes[3].set(title="Learned linear filter", xlabel="Time (sec)")


def plot_coupling_filters(
    responses,
    tuning,
    cmap_name="seismic",
    figsize=(10, 8),
    fontsize=15,
    alpha=0.5,
    cmap_label="hsv",
):
    second_coord_name = list(tuning.coords)[1]
    pref_ang = tuning.idxmax(dim=second_coord_name)
    cmap_tun = plt.colormaps[cmap_label]
    color_tun = (pref_ang.values - pref_ang.values.min()) / (
        pref_ang.values.max() - pref_ang.values.min()
    )

    # plot heatmap
    sum_resp = np.sum(responses, axis=2)
    # normalize by cols (for fixed receiver neuron, scale all responses
    # so that the strongest peaks to 1)
    sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T

    # scale to 0,1
    color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()

    cmap = plt.colormaps[cmap_name]
    n_row, n_col, n_tp = responses.shape
    time = np.arange(n_tp)
    fig, axs = plt.subplots(n_row + 1, n_col + 1, figsize=figsize, sharey="row")
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            axs[rec, send].plot(time, responses[rec, send], color="k")
            axs[rec, send].spines["left"].set_visible(False)
            axs[rec, send].spines["bottom"].set_visible(False)
            axs[rec, send].set_xticks([])
            axs[rec, send].set_yticks([])
            axs[rec, send].axhline(0, color="k", lw=0.5)
            if rec == n_row - 1:
                axs[n_row, send].remove()  # Remove the original axis
                axs[n_row, send] = fig.add_subplot(
                    n_row + 1,
                    n_col + 1,
                    np.ravel_multi_index((n_row, send + 1), (n_row + 1, n_col + 1)),
                    polar=True,
                )  # Add new polar axis

                axs[n_row, send].fill_between(
                    tuning[send].coords[second_coord_name].values,
                    np.zeros(tuning.shape[1]),
                    tuning[send].values,
                    color=cmap_tun(color_tun[send]),
                    alpha=0.5,
                )
                axs[n_row, send].set_xticks([])
                axs[n_row, send].set_yticks([])

        axs[rec, send + 1].remove()  # Remove the original axis
        axs[rec, send + 1] = fig.add_subplot(
            n_row + 1,
            n_col + 1,
            np.ravel_multi_index((rec, send + 1), (n_row + 1, n_col + 1)) + 1,
            polar=True,
        )  # Add new polar axis

        axs[rec, send + 1].fill_between(
            tuning[rec].coords[second_coord_name].values,
            np.zeros(tuning.shape[1]),
            tuning[rec].values,
            color=cmap_tun(color_tun[rec]),
            alpha=0.5,
        )
        axs[rec, send + 1].set_xticks([])
        axs[rec, send + 1].set_yticks([])
    axs[rec + 1, send + 1].set_xticks([])
    axs[rec + 1, send + 1].set_yticks([])
    axs[rec + 1, send + 1].spines["left"].set_visible(False)
    axs[rec + 1, send + 1].spines["bottom"].set_visible(False)
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            xlim = axs[rec, send].get_xlim()
            ylim = axs[rec, send].get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                alpha=alpha,
                color=cmap(color[rec, send]),
                zorder=1,
            )
            axs[rec, send].add_patch(rect)
            axs[rec, send].set_xlim(xlim)
            axs[rec, send].set_ylim(ylim)
    axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
    axs[n_row, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)

    plt.suptitle("Pairwise Interaction", fontsize=fontsize)
    return fig
