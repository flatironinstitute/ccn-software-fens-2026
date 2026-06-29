import nemos as nmo
import numpy as np

__all__ = [
    "relabel",
    "select_sessions"
]


def relabel(model):
    """
    Relabel the states of a fitted GLM-HMM to match the ordering used in
    Ashwood et al. (2023): engaged state first, followed by biased-left
    and biased-right states.
    
    Parameters
    ----------
    model :
        GLM-HMM model.

    Returns
    -------
    model :
        GLM-HMM model with states reordered in-place.
    """
    # the intercept is the bias term
    bias_right = np.flatnonzero(model.intercept_ < -2)
    bias_left = np.flatnonzero(model.intercept_ > 2)
    # less biased (smallest |intercept|) is the engaged state
    engaged_state = np.argmin(np.abs(model.intercept_))
    relabel = np.concatenate([[engaged_state], bias_left, bias_right])
    
    # apply re-labeling
    model.coef_ = model.coef_[:, relabel]
    model.intercept_ = model.intercept_[relabel]
    model.initial_prob_ = model.initial_prob_[relabel]
    model.transition_prob_ = model.transition_prob_[relabel][:, relabel]
    return model


def select_sessions(
    trials,
    max_violations=10,
    violation_value=0,
    probability_left=0.5,
):
    """
    Select sessions containing 0.2, 0.5 and 0.8 probability blocks,
    and fewer than `max_violations` invalid trials in the specified block.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial-level dataframe with columns 'session', 'probabilityLeft', and 'choice'.
    max_violations : int, optional
        Maximum number of invalid trials allowed in the specified block. Default is 10.
    violation_value : int, optional
        Value in the 'choice' column that indicates an invalid trial. Default is 0.
    probability_left : float, optional
        The probability block to filter trials on. Default is 0.5.

    Returns
    -------
    df_trials : pd.DataFrame
        Trials from valid sessions restricted to the selected block.
    valid_sessions : pd.Index
        Session identifiers that passed the selection criteria.
    """

    has_three_blocks = (
        trials.groupby("session")["probabilityLeft"]
        .agg(lambda s: set(s.unique()) == {0.2, 0.5, 0.8})
    )

    violations = (
        trials.query("probabilityLeft == @probability_left")
        .groupby("session")["choice"]
        .agg(lambda s: s.eq(violation_value).sum())
    )

    valid_sessions = has_three_blocks.index[
        has_three_blocks & (violations < max_violations)
    ]

    df_trials = trials.query(
        "session in @valid_sessions and probabilityLeft == @probability_left"
    )

    return df_trials, valid_sessions