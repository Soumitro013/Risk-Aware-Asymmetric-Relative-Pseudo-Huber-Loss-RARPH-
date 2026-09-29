# Risk-Aware-Asymmetric-Relative-Pseudo-Huber-Loss-RARPH-

A real-time Python implementation of **RARPH (Risk-Aware Asymmetric Relative Pseudo-Huber) Loss**, a custom regression loss function designed for wind-turbine generator temperature prediction.

## Overview

RARPH is designed to address situations where standard regression losses such as MSE and MAE may not adequately represent the practical consequences of prediction errors.

The proposed loss combines four ideas:

- **Relative error** — normalizes the prediction error with respect to the actual value.
- **Asymmetric penalty** — penalizes underprediction more strongly than overprediction.
- **Risk-aware weighting** — gives greater importance to observations near critical operating temperatures.
- **Pseudo-Huber robustness** — reduces the influence of extreme residuals and sensor anomalies.

The loss is defined as:

\[
L_i = w(y_i)\,a(r_i)\,H_\delta(r_i)
\]

where

\[
r_i = \frac{\hat y_i-y_i}{|y_i|+\epsilon}
\]

\[
a(r_i)=
\begin{cases}
c_u, & r_i<0\\
c_o, & r_i\geq0
\end{cases}
\]

\[
w(y_i)=1+\frac{\lambda}{1+\exp[-k(y_i-T_w)]}
\]

and

\[
H_\delta(r_i)
=
\delta^2
\left[
\sqrt{1+\left(\frac{r_i}{\delta}\right)^2}-1
\right].
\]

## Repository Structure

```text
.
├── RARPH.py
├── main.py
├── requirements.txt
└── README.md
