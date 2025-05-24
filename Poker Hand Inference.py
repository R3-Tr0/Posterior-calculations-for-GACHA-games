import numpy as np
import tkinter as tk
from tkinter import ttk
import scipy.stats as sts
import matplotlib.pyplot as plt

"""
Bayesian Inference: Posterior Distribution of a Poker Hand's Occurrence

This program uses Bayesian updating to infer the probability (per hand)
of obtaining a selected poker hand (e.g. Royal Flush, Straight Flush, Four of a Kind, etc.)
from a deck that may be fair or biased. In each game the maximum of 4 players
are dealt a hand. The event considered is "at least one player gets the hand".

A new input was added: number of decks used (or leave blank if unknown).
"""

# Define the poker hand settings: fair and biased probabilities (per-hand)
hand_data = {
    "Royal Flush": {
        "fair_p": 0.00000154,
        "biased_p": 0.000003,  # for reference
        "n_games": 50,
    },
    "Straight Flush": {
        "fair_p": 0.0000139,
        "biased_p": 0.000028,
        "n_games": 50,
    },
    "Four of a Kind": {
        "fair_p": 0.00024010,
        "biased_p": 0.0005,
        "n_games": 50,
    },
    "Full House": {
        "fair_p": 0.001440576,
        "biased_p": 0.003,
        "n_games": 50,
    },
    "Flush": {
        "fair_p": 0.001965401,
        "biased_p": 0.005,
        "n_games": 50,
    },
    "Straight": {
        "fair_p": 0.00392465,
        "biased_p": 0.008,
        "n_games": 50,
    },
    "Three of a Kind": {
        "fair_p": 0.021128451,
        "biased_p": 0.04,
        "n_games": 50,
    },
    "Two Pair": {
        "fair_p": 0.047539015,
        "biased_p": 0.09,
        "n_games": 50,
    },
    "One Pair": {
        "fair_p": 0.42256903,
        "biased_p": 0.5,
        "n_games": 50,
    },
    "High Card": {
        "fair_p": 0.50117739,
        "biased_p": 0.55,
        "n_games": 50,
    }
}

# Number of players per game (max of 4)
N_PLAYERS = 4

def bayesian_update(hand_key, decks):
    # Retrieve the parameters for the chosen hand type.
    data = hand_data[hand_key]
    fair_p = data["fair_p"]
    n_games = data["n_games"]

    # If a number of decks was provided, adjust the per-hand probability.
    if decks is not None:
        fair_p_adj = 1 - (1 - fair_p)**decks
        deck_text = f"{decks} deck{'s' if decks != 1 else ''}"
    else:
        fair_p_adj = fair_p
        deck_text = "unknown number of decks"

    # In each game, the probability that at least one player gets the hand 
    # using the (possibly adjusted) fair probability:
    true_event_prob = 1 - (1 - fair_p_adj)**N_PLAYERS

    # Generate observed evidence randomly:
    observed = np.random.binomial(n_games, true_event_prob)

    # Create grid for p (per-hand probability) in [0, 1] to reflect valid probability values.
    p_grid = np.linspace(0, 1, 500)

    # Construct a prior distribution.
    # Center the prior around fair_p
    scale = 50
    alpha_prior = fair_p_adj * scale
    beta_prior = scale - alpha_prior
    prior = sts.beta.pdf(p_grid, alpha_prior, beta_prior)

    # Likelihood: probability of seeing "observed" successes from n_games.
    # In each game, event probability = 1 - (1 - p)^N_PLAYERS.
    event_prob = 1 - (1 - p_grid)**N_PLAYERS
    likelihood = sts.binom.pmf(observed, n_games, event_prob)

    # Unnormalized posterior and normalization.
    unnorm_post = likelihood * prior
    norm_post = unnorm_post / unnorm_post.sum()

    # Plot the curves, scaled for visual comparison.
    plt.figure(figsize=(8, 6))
    plt.plot(p_grid, prior / prior.max(), label=f"Prior Beta({alpha_prior:.1f},{beta_prior:.1f})", linestyle='--')
    plt.plot(p_grid, likelihood / likelihood.max(), label="Likelihood (scaled)", linestyle='-.')
    plt.plot(p_grid, norm_post / norm_post.max(), label="Normalized Posterior", linewidth=2)
    plt.xlabel("p (Per-hand probability)")
    plt.ylabel("Scaled Density")
    plt.title(f"Posterior for '{hand_key}' Occurrence\n"
              f"({deck_text}, {observed} successes in {n_games} games)")
    plt.legend()
    plt.grid(True)
    plt.show()

def on_hand_selected(_event):
    hand = hand_selector.get()
    # Process decks input.
    deck_input = deck_entry.get().strip()
    decks = None
    if deck_input:
        try:
            decks = int(deck_input)
            if decks < 1:
                decks = 1
        except ValueError:
            decks = None
    bayesian_update(hand, decks)

# Build a simple Tkinter interface.
root = tk.Tk()
root.title("Poker Hand Bayesian Inference")

tk.Label(root, text="Select Poker Hand Type:").pack(pady=5)
hand_selector = ttk.Combobox(root, values=list(hand_data.keys()), state="readonly", width=20)
hand_selector.pack(pady=5)
hand_selector.current(0)  # default selection
hand_selector.bind("<<ComboboxSelected>>", on_hand_selected)

# Add input for number of decks.
tk.Label(root, text="Enter Number of Decks (leave blank if unknown):").pack(pady=5)
deck_entry = tk.Entry(root)
deck_entry.pack(pady=5)

def run_update():
    hand = hand_selector.get()
    deck_input = deck_entry.get().strip()
    decks = None
    if deck_input:
        try:
            decks = int(deck_input)
            if decks < 1:
                decks = 1
        except ValueError:
            decks = None
    bayesian_update(hand, decks)

tk.Button(root, text="Plot Posterior", command=run_update).pack(pady=10)

root.mainloop()
