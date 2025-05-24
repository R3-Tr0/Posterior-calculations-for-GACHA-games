# Posterior Calculations for GACHA Games

This repository demonstrates Bayesian updating techniques applied to games and probabilistic events, including dice events, poker hands, and coin tosses. The examples harness Python libraries such as NumPy, SciPy, Matplotlib, and Tkinter to simulate observed events and plot posterior distributions.

## Overview

The repository contains three main sections:

1. **Dice Event Bayesian Inference**  
    Infers the per-toss probability of a selected dice event (e.g., "At least one 6", "All dice different") using Bayesian updating. It features:
    - A dynamic adjustment of fair probabilities based on the number of dice.
    - Simulation of observed successes with a binomial process.
    - A Tkinter interface for choosing the dice event, probability type (fair/biased), and the number of dice.
    - Visualization of the prior, likelihood, and normalized posterior.

2. **Poker Hand Bayesian Inference**  
    Estimates the per-hand probability of achieving specific poker hands (e.g., Royal Flush, Four of a Kind) from a deck. Key points include:
    - Use of preset probabilities (fair and biased) for different poker hands.
    - Modification of probability when multiple decks are considered.
    - Consideration of up to 4 players per game, with the event being "at least one player gets the hand".
    - A simple Tkinter interface allowing selection of hand type, probability type, and the number of decks.
    - Plots illustrating the Bayesian update.

3. **Coin Toss Posterior Update**  
    Demonstrates the Bayesian update for a coin’s bias:
    - Uses a Beta prior centered around a fair coin (Beta(10,10)).
    - Updates with observed data from coin tosses.
    - Plots showing the prior, likelihood, and normalized posterior distributions.

## Requirements

- Python 3.13.2 (or compatible version)
- NumPy
- SciPy
- Matplotlib
- Tkinter (usually included with Python)

Install required libraries using pip if not already installed:

```bash
pip install numpy scipy matplotlib
```

## Running the Applications

Each section has its own script. To run any of them:

1. **Dice Event Inference**:  
    Execute the script (e.g., `python dice_inference.py`) to launch a Tkinter window where you can select a dice event, adjust the number of dice, choose between fair or biased probability, and then plot the posterior distribution.

2. **Poker Hand Inference**:  
    Run the corresponding script (e.g., `python poker_inference.py`) to open a Tkinter interface. You can select a poker hand, enter the number of decks, choose the probability type, and generate the posterior plot.

3. **Coin Toss Update**:  
    Run the coin toss script (e.g., `python coin_toss.py`) to display a plot that compares the prior, likelihood, and posterior for the coin's bias based on observed coin toss data.

## Code Structure

Each script is structured as follows:

- **Probability Adjustment Functions**:  
  Functions calculate the fair probability for events that depend on variable inputs (e.g., number of dice).

- **Bayesian Update Functions**:  
  The functions simulate observed data (using a binomial process) and compute the posterior distribution from prior assumptions and likelihood.

- **Visualization**:  
  Matplotlib is used for plotting the results. Plots show a scaled comparison of the prior, the likelihood, and the normalized posterior curve.

- **Graphical User Interface (GUI)**:  
  The Tkinter-based interface allows interactive selection and input by the user, triggering real-time Bayesian updates and display of plots.

## Example: Bayesian Update for a Dice Event

Below is a simplified excerpt showing how the Bayesian update is performed for a dice event:

```python
def adjust_probability(event_key, n_dice):
     if event_key == "At least one 6":
          return 1 - (5/6)**n_dice
     # Additional event calculations...

def bayesian_update(event_key, n_dice, prob_type):
     data = dice_events[event_key]
     n_trials = data["n_trials"]

     base_p = data["biased_p"] if prob_type == "biased" else adjust_probability(event_key, n_dice)
     observed = np.random.binomial(n_trials, base_p)
     p_grid = np.linspace(0, 1, 500)
     scale = 50
     alpha_prior = base_p * scale
     beta_prior = scale - alpha_prior
     prior = sts.beta.pdf(p_grid, alpha_prior, beta_prior)
     likelihood = sts.binom.pmf(observed, n_trials, p_grid)
     norm_post = (likelihood * prior) / (likelihood * prior).sum()

     # Plotting code...

     plt.show()
```

## Conclusion

This repository provides interactive examples to visualize how Bayesian inference can be applied to estimate unknown probabilities in gaming contexts and common probabilistic scenarios. Each module combines simulation, statistical modeling, and interactive user interfaces to provide clear insights into the impact of data on posterior distributions.

Feel free to explore the code, modify parameters, and experiment with different scenarios to deepen your understanding of Bayesian inference.

