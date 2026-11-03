import numpy as np


def run_gaussian_bandit(true_means, sigma, rounds, seed):
    true_means = np.asarray(true_means, dtype=float)

    if true_means.shape != (3,) or not np.all(np.isfinite(true_means)):
        raise ValueError("Exactly three finite true means are required.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive.")

    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds <= 0:
        raise ValueError("Rounds must be a positive integer.")

    rng = np.random.default_rng(int(seed))

    posterior_means = np.zeros(3)
    posterior_variances = np.full(3, 25.0)

    counts = np.zeros(3, dtype=int)
    total_reward = 0.0

    for _ in range(rounds):
        samples = rng.normal(
            posterior_means,
            np.sqrt(posterior_variances)
        )

        selected = int(np.argmax(samples))
        reward = rng.normal(true_means[selected], sigma)

        counts[selected] += 1
        total_reward += reward

        old_mean = posterior_means[selected]
        old_variance = posterior_variances[selected]

        new_variance = 1.0 / (
            1.0 / old_variance + 1.0 / sigma**2
        )

        new_mean = new_variance * (
            old_mean / old_variance + reward / sigma**2
        )

        posterior_means[selected] = new_mean
        posterior_variances[selected] = new_variance

    return {
        "estimated_means": posterior_means.tolist(),
        "counts": counts.tolist(),
        "total_reward": float(total_reward),
        "recommendation": int(np.argmax(posterior_means)) + 1,
    }
