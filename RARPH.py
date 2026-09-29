"""Risk-Aware Asymmetric Relative Pseudo-Huber Loss (RARPH)"""
import math

def rarph_loss(
    y_true,
    y_pred,
    epsilon=1.0,
    delta=0.1,
    c_under=2.0,
    c_over=1.0,
    warning_temp=90.0,
    risk_lambda=1.0,
    risk_k=0.5
):
    """Risk-Aware Asymmetric Relative Pseudo-Huber Loss."""

    # Relative residual
    r = (y_pred - y_true) / (abs(y_true) + epsilon)

    # Penalize underprediction more strongly
    asym_weight = c_under if r < 0 else c_over

    # Higher weight near/above warning temperature
    sigmoid = 1.0 / (
        1.0 + math.exp(-risk_k * (y_true - warning_temp))
    )
    risk_weight = 1.0 + risk_lambda * sigmoid

    # Pseudo-Huber term
    huber = delta**2 * (
        math.sqrt(1.0 + (r / delta)**2) - 1.0
    )

    return risk_weight * asym_weight * huber