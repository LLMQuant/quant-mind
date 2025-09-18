"""Lightweight tool calculating portfolio metrics."""

from quantmind.tools import tool


@tool
def estimate_greeks(delta: float, gamma: float, underlying_move: float) -> dict:
    """Estimate option PnL impact from delta and gamma.

    Args:
        delta (float): Delta exposure of the option book.
        gamma (float): Gamma exposure of the option book.
        underlying_move (float): Expected underlying price change.

    Returns:
        dict: Estimated delta and gamma contribution.
    """
    delta_pnl = delta * underlying_move
    gamma_pnl = 0.5 * gamma * underlying_move**2
    return {
        "delta_contribution": delta_pnl,
        "gamma_contribution": gamma_pnl,
    }


def main():
    """Run the Greeks estimator with example exposures."""
    pnl = estimate_greeks(delta=1200, gamma=-45, underlying_move=0.8)
    print("Estimated option PnL contributions:")
    for key, value in pnl.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
