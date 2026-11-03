import streamlit as st
import pandas as pd

from gaussian_bandit import run_gaussian_bandit

st.title("Gaussian Thompson Sampling")
st.write("Simulation of three alternatives with continuous Gaussian rewards.")

st.subheader("Simulation Settings")

col1, col2 = st.columns(2)

with col1:
    mu1 = st.number_input("Alternative 1 True Mean", value=1.0)
    mu2 = st.number_input("Alternative 2 True Mean", value=2.0)
    mu3 = st.number_input("Alternative 3 True Mean", value=3.0)

with col2:
    sigma = st.number_input("Known Sigma", value=1.0)
    rounds = st.number_input(
        "Number of Rounds",
        min_value=1,
        value=100,
        step=1
    )
    seed = st.number_input("Random Seed", value=42, step=1)

if st.button("Run Simulation"):
    if sigma <= 0:
        st.error("Sigma must be positive.")
    else:
        results = run_gaussian_bandit(
            [mu1, mu2, mu3],
            sigma,
            int(rounds),
            int(seed)
        )

        df = pd.DataFrame({
            "Alternative": [1, 2, 3],
            "True Mean": [mu1, mu2, mu3],
            "Estimated Mean": results["estimated_means"],
            "Selection Count": results["counts"]
        })

        st.subheader("Simulation Results")
        st.dataframe(df, hide_index=True)

        st.metric(
            "Total Observed Reward",
            f"{results['total_reward']:.2f}"
        )

        st.success(
            f"Recommended Alternative: {results['recommendation']}"
        )

        st.subheader("Selection Counts")
        st.bar_chart(
            df.set_index("Alternative")["Selection Count"]
        )

        st.write("Total selections:", sum(results["counts"]))
