# DS223 Gaussian Bandit

This project extends the original Bayesian Carousel Bandits application with Gaussian Thompson Sampling for continuous rewards.

Original repository: https://github.com/hovhannisyan91/DS223_Group_Project

Only the frontend was copied into our independent repository. The original pages are preserved.

## Original Algorithm: Beta-Bernoulli Thompson Sampling

The original application uses Thompson Sampling to select alternatives for a carousel.

Each alternative starts with a Beta(1,1) prior. The algorithm samples from each alternative's Beta distribution and displays the alternatives with the highest sampled values.

User interactions are recorded as binary rewards. A click is a success (1), while a no-click is a failure (0).

For a click, alpha increases by 1. For a no-click, beta increases by 1.

The estimated click probability is alpha / (alpha + beta).

Streamlit session state stores the carousels, alternatives, interactions, and statistics. This allows the Create Carousels, Interaction, and Analytics pages to share information.

## Gaussian Thompson Sampling

Our Gaussian model uses three alternatives with unknown true means.

Each alternative starts with a Normal(0,25) prior.

The noise standard deviation sigma is known.

In each round, the algorithm:

1. Samples a possible mean from each posterior.
2. Selects the alternative with the highest sampled mean.
3. Generates a continuous Gaussian reward.
4. Updates only the selected alternative's posterior.

Posterior variance:

new_variance = 1 / (1 / old_variance + 1 / sigma²)

Posterior mean:

new_mean = new_variance * (old_mean / old_variance + reward / sigma²)

The final recommendation is the alternative with the highest posterior mean.

## Experiments

All experiments use true means [1, 2, 3], prior Normal(0,25), and seed 42.

| Experiment | Rounds | Sigma | Estimated Means | Selection Counts | Recommendation |
|---|---:|---:|---|---|---|
| Short run | 100 | 1 | [1.57, 1.729, 2.995] | [4, 2, 94] | Alternative 3 |
| Longer run | 1000 | 1 | [1.306, 2.142, 2.979] | [5, 13, 982] | Alternative 3 |
| Higher noise | 1000 | 3 | [1.154, 1.484, 2.954] | [21, 23, 956] | Alternative 3 |

## Experiment Discussion

The short run demonstrates how Thompson Sampling starts learning which alternative has the highest expected reward.

The longer run provides more observations, which generally improves the estimate of frequently selected alternatives.

Higher noise makes rewards more variable and can slow learning.

An interesting observation is that more rounds do not guarantee better estimates for every alternative. Thompson Sampling focuses on promising alternatives, so alternatives selected only a few times may remain uncertain.

We checked reproducibility by repeating each experiment with the same seed and settings. The results were identical.

We also verified that selection counts sum to the number of rounds.

## Setup Instructions

Requirements: Python 3.12 and uv.

Clone the repository:

    git clone https://github.com/milenamanukian88/DS223_Gaussian_Bandit.git

Enter the folder:

    cd DS223_Gaussian_Bandit

Install dependencies:

    uv sync --locked

Run the application:

    uv run streamlit run main.py

## Project Links

GitHub: https://github.com/milenamanukian88/DS223_Gaussian_Bandit

Streamlit: Deployment link will be added after deployment.
