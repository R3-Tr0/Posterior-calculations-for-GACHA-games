import numpy as np
import tkinter as tk
from tkinter import ttk
import scipy.stats as sts
import matplotlib.pyplot as plt

"""
Bayesian Inference: Posterior Distribution for Dice Event Occurrence

This program uses Bayesian updating to infer the per-toss probability of a selected dice event 
(e.g. "At least one 6", "All dice different", etc.) from a dice game that may be fair or biased. 
Users can adjust the number of dice being tossed, which (for fair dice) automatically adjusts the event probability.
"""

# Define the dice event settings: fair and biased probabilities
dice_events = {
    "At least one 6": {
        "fair_p": None,  # computed based on number of dice
        "biased_p": 0.6,
        "n_trials": 50,
    },
    "All dice different": {
        "fair_p": None,  # computed based on number of dice
        "biased_p": 0.4,
        "n_trials": 50,
    },
    "Sum ≥ 15": {
        "fair_p": 0.25,  # approximated for 3 dice (unchanged by dice count here)
        "biased_p": 0.35,
        "n_trials": 50,
    },
    "All dice same": {
        "fair_p": None,  # computed based on number of dice
        "biased_p": 0.05,
        "n_trials": 50,
    },
    "Exactly one 6": {
        "fair_p": None,  # computed based on number of dice
        "biased_p": 0.3,
        "n_trials": 50,
    }
}

DEFAULT_DICE = 3

def adjust_probability(event_key, n_dice):
    """Adjust the fair event probability given the number of dice."""
    if event_key == "At least one 6":
        return 1 - (5/6)**n_dice
    elif event_key == "All dice different":
        if n_dice > 6:
            return 0.0
        prob = 1
        for i in range(n_dice):
            prob *= (6 - i) / 6
        return prob
    elif event_key == "All dice same":
        return 6 / (6**n_dice)
    elif event_key == "Exactly one 6":
        return n_dice * (1/6) * (5/6)**(n_dice - 1)
    elif event_key == "Sum ≥ 15":
        # For "Sum ≥ 15", use simulation to approximate the probability.
        sims = np.random.randint(1, 7, size=(100_000, n_dice))
        return np.mean(sims.sum(axis=1) >= 15)
    else:
        return 0.1

def bayesian_update(event_key, n_dice, prob_type):
    data = dice_events[event_key]
    n_trials = data["n_trials"]

    if prob_type == "biased":
        base_p = data["biased_p"]
        prob_label = "Biased"
    else:
        base_p = adjust_probability(event_key, n_dice)
        prob_label = "Fair"

    # Simulate observed successes using a binomial process.
    observed = np.random.binomial(n_trials, base_p)

    # Create grid for probability values.
    p_grid = np.linspace(0, 1, 500)
    scale = 50
    alpha_prior = base_p * scale
    beta_prior = scale - alpha_prior
    prior = sts.beta.pdf(p_grid, alpha_prior, beta_prior)

    # Likelihood of seeing 'observed' successes from n_trials.
    likelihood = sts.binom.pmf(observed, n_trials, p_grid)

    # Compute the unnormalized and normalized posterior distributions.
    unnorm_post = likelihood * prior
    norm_post = unnorm_post / unnorm_post.sum()

    # Plot the prior, likelihood, and normalized posterior.
    plt.figure(figsize=(8, 6))
    plt.plot(p_grid, prior / prior.max(), label=f"Prior Beta({alpha_prior:.1f},{beta_prior:.1f})", linestyle='--')
    plt.plot(p_grid, likelihood / likelihood.max(), label="Likelihood (scaled)", linestyle='-.')
    plt.plot(p_grid, norm_post / norm_post.max(), label="Normalized Posterior", linewidth=2)
    plt.xlabel("p (Per-toss probability)")
    plt.ylabel("Scaled Density")
    plt.title(f"Posterior for '{event_key}' with {n_dice} dice\n"
              f"Observed {observed} successes in {n_trials} trials ({prob_label} p)")
    plt.legend()
    plt.grid(True)
    plt.show()

def on_event_selected(_event):
    run_update()

def run_update():
    event = event_selector.get()
    dice_input = dice_entry.get().strip()
    n_dice = DEFAULT_DICE
    if dice_input:
        try:
            n_dice = max(1, int(dice_input))
        except ValueError:
            n_dice = DEFAULT_DICE
    prob_type = prob_var.get()
    bayesian_update(event, n_dice, prob_type)

# Build the Tkinter user interface.
root = tk.Tk()
root.title("Dice Event Bayesian Inference")

tk.Label(root, text="Select Dice Event:").pack(pady=5)
event_selector = ttk.Combobox(root, values=list(dice_events.keys()), state="readonly", width=25)
event_selector.pack(pady=5)
event_selector.current(0)
event_selector.bind("<<ComboboxSelected>>", on_event_selected)

# Radio buttons to choose probability type (fair or biased)
prob_var = tk.StringVar(value="fair")
frame = tk.LabelFrame(root, text="Select Probability Type", padx=10, pady=5)
frame.pack(pady=5)
tk.Radiobutton(frame, text="Fair", variable=prob_var, value="fair").pack(side="left", padx=10)
tk.Radiobutton(frame, text="Biased", variable=prob_var, value="biased").pack(side="left", padx=10)

tk.Label(root, text="Enter Number of Dice (default is 3):").pack(pady=5)
dice_entry = tk.Entry(root)
dice_entry.pack(pady=5)

tk.Button(root, text="Plot Posterior", command=run_update).pack(pady=10)

root.mainloop()
