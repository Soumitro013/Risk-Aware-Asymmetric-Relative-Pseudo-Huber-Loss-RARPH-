# Risk-Aware-Asymmetric-Relative-Pseudo-Huber-Loss-RARPH-

A real-time Python implementation of **RARPH (Risk-Aware Asymmetric Relative Pseudo-Huber) Loss**, a custom regression loss function designed for wind-turbine generator temperature prediction.

## Overview

RARPH is designed to address situations where standard regression losses such as MSE and MAE may not adequately represent the practical consequences of prediction errors.

The proposed loss combines four ideas:

- **Relative error**: normalizes the prediction error with respect to the actual value.
- **Asymmetric penalty**: penalizes underprediction more strongly than overprediction.
- **Risk-aware weighting**: gives greater importance to observations near critical operating temperatures.
- **Pseudo-Huber robustness**: reduces the influence of extreme residuals and sensor anomalies.

The loss is defined as:

$$
L_i = w(y_i)\,a(r_i)\,H_\delta(r_i)
$$

where

$$
r_i = \frac{\hat{y}_i-y_i}{|y_i|+\epsilon}
$$

The asymmetric penalty is defined as:

$$
a(r_i)=
\begin{cases}
c_u, & r_i < 0,\\
c_o, & r_i \geq 0.
\end{cases}
$$

The risk-weighting function is:

$$
w(y_i) = 1+\frac{\lambda}{1+\exp[-k(y_i-T_w)]}
$$

The Pseudo-Huber component is:

$$
H_\delta(r_i) = \delta^2\left[\sqrt{1+\left(\frac{r_i}{\delta}\right)^2}-1\right]
$$

For a dataset containing $n$ observations, the overall RARPH loss is:

$$
L_{\mathrm{RARPH}} = \frac{1}{n}\sum_{i=1}^{n}w(y_i)\,a(r_i)\,H_\delta(r_i)
$$
## Repository Structure

```text
.
├── RARPH.py
├── main.py
├── requirements.txt
└── README.md
