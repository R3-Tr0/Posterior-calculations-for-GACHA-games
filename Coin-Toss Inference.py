import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from scipy.stats import betabinom
import random

# Fixed prior parameters (Beta(10,10))
prior_alpha = 10
prior_beta = 10

def plot_posterior():
    """Plot the posterior distribution based on user-provided observed data."""
    try:
        # Get observed data from user input.
        obs_trials = int(obs_trials_entry.get())
        obs_heads = int(obs_heads_entry.get())
    except ValueError:
        messagebox.showerror("Input error", "Enter valid integers for observed trials and observed heads.")
        return

    if obs_trials < 1 or not (0 <= obs_heads <= obs_trials):
        messagebox.showerror("Input error", "Ensure: observed_trials >= 1 and 0 <= observed_heads <= observed_trials.")
        return

    # Update posterior parameters based on observed data: Beta(prior_alpha+obs_heads, prior_beta+obs_trials-obs_heads)
    post_alpha = prior_alpha + obs_heads
    post_beta = prior_beta + (obs_trials - obs_heads)

    # Build grid for parameter p (coin bias)
    p_grid = np.linspace(0, 1, 500)
    prior_pdf = sts.beta.pdf(p_grid, prior_alpha, prior_beta)
    likelihood = sts.binom.pmf(obs_heads, obs_trials, p_grid)
    unnorm_post = likelihood * prior_pdf
    norm_post = unnorm_post / unnorm_post.max()

    plt.figure(figsize=(8,6))
    plt.plot(p_grid, prior_pdf / prior_pdf.max(), label=f"Prior Beta({prior_alpha},{prior_beta})", linestyle='--')
    plt.plot(p_grid, likelihood / likelihood.max(), label="Likelihood (scaled)", linestyle='-.')
    plt.plot(p_grid, norm_post, label="Normalized Posterior", linewidth=2)
    plt.xlabel("p (Probability of heads)")
    plt.ylabel("Scaled Density")
    plt.title("Bayesian Update: Posterior distribution of Coin's Bias")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_probability():
    """Calculate the predictive probability using a beta-binomial model."""
    # Determine observed data from user input (for updating posterior)
    try:
        obs_trials = int(obs_trials_entry.get())
        obs_heads = int(obs_heads_entry.get())
    except ValueError:
        messagebox.showerror("Input error", "Enter valid integers for observed trials and observed heads.")
        return

    if obs_trials < 1 or not (0 <= obs_heads <= obs_trials):
        messagebox.showerror("Input error", "Ensure: observed_trials >= 1 and 0 <= observed_heads <= observed_trials.")
        return

    # Update posterior parameters using the observed data.
    post_alpha = prior_alpha + obs_heads
    post_beta = prior_beta + (obs_trials - obs_heads)

    # For the predictive event, use the radio button select:
    if event_mode.get() == "random":
        try:
            n_future = random.randint(5, 50)
            threshold = random.randint(0, n_future)
            op = random.choice(["<", "<=", "=", ">", ">="])
            trials_entry.delete(0, tk.END)
            trials_entry.insert(0, str(n_future))
            threshold_entry.delete(0, tk.END)
            threshold_entry.insert(0, str(threshold))
            operator_combo.set(op)
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting random parameters: {e}")
            return
    else:
        try:
            n_future = int(trials_entry.get())
            threshold = int(threshold_entry.get())
            op = operator_combo.get()
        except ValueError:
            messagebox.showerror("Input error", "Enter valid integers for number of trials and threshold.")
            return

    if n_future < 1 or threshold < 0 or threshold > n_future:
        messagebox.showerror("Input error", "Make sure: Number of trials >= 1 and 0 <= threshold <= number of trials.")
        return

    # Use the beta-binomial predictive: betabinom(n_future, post_alpha, post_beta)
    bb_model = betabinom(n_future, post_alpha, post_beta)
    outcomes = np.arange(n_future+1)
    
    if op == "<":
        valid = outcomes < threshold
    elif op == "<=":
        valid = outcomes <= threshold
    elif op == "=":
        valid = outcomes == threshold
    elif op == ">":
        valid = outcomes > threshold
    elif op == ">=":
        valid = outcomes >= threshold
    else:
        messagebox.showerror("Input error", "Operator not recognized.")
        return

    prob = bb_model.pmf(outcomes[valid]).sum()
    
    messagebox.showinfo("Predictive Probability",
                        f"For a future event of {n_future} trials with outcome {op} {threshold} heads:\n"
                        f"Predictive Probability = {prob:.4f}")

# ---------------- GUI Setup ----------------

root = tk.Tk()
root.title("Bayesian Coin Toss Inference")

# --- Observed Data Section ---
obs_frame = ttk.LabelFrame(root, text="Observed Data (for Posterior Update)")
obs_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

ttk.Label(obs_frame, text="Observed Trials:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
obs_trials_entry = ttk.Entry(obs_frame, width=10)
obs_trials_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
obs_trials_entry.insert(0, "10")

ttk.Label(obs_frame, text="Observed Heads:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
obs_heads_entry = ttk.Entry(obs_frame, width=10)
obs_heads_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
obs_heads_entry.insert(0, "3")

# --- Future Event Parameters Section ---
event_frame = ttk.LabelFrame(root, text="Future Event Parameters")
event_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

event_mode = tk.StringVar(value="custom")
ttk.Radiobutton(event_frame, text="Custom Input", variable=event_mode, value="custom").grid(row=0, column=0, padx=5, pady=5)
ttk.Radiobutton(event_frame, text="Random Event", variable=event_mode, value="random").grid(row=0, column=1, padx=5, pady=5)

ttk.Label(event_frame, text="Operator:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
operator_combo = ttk.Combobox(event_frame, values=["<", "<=", "=", ">", ">="], state="readonly", width=5)
operator_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
operator_combo.set("=")

ttk.Label(event_frame, text="Number of Future Trials:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
trials_entry = ttk.Entry(event_frame, width=10)
trials_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
trials_entry.insert(0, "10")

ttk.Label(event_frame, text="Threshold (# Heads):").grid(row=3, column=0, padx=5, pady=5, sticky="e")
threshold_entry = ttk.Entry(event_frame, width=10)
threshold_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
threshold_entry.insert(0, "3")

# --- Action Buttons ---
calc_button = ttk.Button(root, text="Calculate Predictive Probability", command=calculate_probability)
calc_button.grid(row=2, column=0, columnspan=2, padx=10, pady=8)

plot_button = ttk.Button(root, text="Plot Posterior", command=plot_posterior)
plot_button.grid(row=3, column=0, columnspan=2, padx=10, pady=8)

root.mainloop()
