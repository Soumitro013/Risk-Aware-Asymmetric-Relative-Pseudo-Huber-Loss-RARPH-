# RARPH Loss — Real-Time Python Implementation

<p align="center">

**Risk-Aware Asymmetric Relative Pseudo-Huber (RARPH) Loss**

A custom regression loss function for risk-sensitive temperature prediction in wind-turbine generators.

</p>

---

## Overview

RARPH is a custom regression loss function designed for situations where standard losses such as **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** may not adequately represent the practical consequences of prediction errors.

The proposed loss is motivated by **real-time wind-turbine generator temperature prediction**, where:

- prediction errors should be interpreted relative to the actual temperature,
- underprediction can be more costly than overprediction,
- observations near critical temperatures are more important, and
- extreme sensor errors should not dominate the loss.

The proposed loss combines four ideas:

- **Relative error** — normalizes the prediction error with respect to the actual value.
- **Asymmetric penalty** — penalizes underprediction more strongly than overprediction.
- **Risk-aware weighting** — gives greater importance to observations near critical operating temperatures.
- **Pseudo-Huber robustness** — reduces the influence of extreme residuals and sensor anomalies.

---

## Mathematical Formulation

For observation $i$, let:

- $y_i$ = actual temperature
- $\hat{y}_i$ = predicted temperature

### 1. Relative Residual

The relative prediction error is defined as:

$$
r_i =
\frac{\hat{y}_i-y_i}
{|y_i|+\epsilon}
$$

where $\epsilon>0$ is a small stabilizing constant that prevents division by zero.

---

### 2. Asymmetric Penalty

Underprediction and overprediction are assigned different penalty weights:

$$
a(r_i) = 
\begin{cases}
c_u, & r_i < 0,\\
c_o, & r_i \geq 0.
\end{cases}
$$

where:

- $c_u$ = penalty coefficient for underprediction,
- $c_o$ = penalty coefficient for overprediction.

For a risk-sensitive application, typically:

$$
c_u > c_o.
$$

---

### 3. Risk-Weighting Function

Observations approaching a warning temperature $T_w$ receive increased importance:

$$
w(y_i) = 
1+
\frac{\lambda}
{1+\exp[-k(y_i-T_w)]}
$$

where:

- $T_w$ = warning/critical temperature,
- $\lambda$ = maximum additional risk weighting,
- $k$ = sharpness of the risk transition.

At the warning temperature:

$$
w(T_w)=1+\frac{\lambda}{2}.
$$

For very high temperatures:

$$
w(y_i)\rightarrow1+\lambda.
$$

---

### 4. Pseudo-Huber Component

The robust error component is:

$$
H_\delta(r_i) =
\delta^2
\left[
\sqrt{
1+\left(\frac{r_i}{\delta}\right)^2
}
-1
\right]
$$

where $\delta>0$ controls the transition between quadratic and approximately linear behaviour.

For small residuals:

$$
H_\delta(r)
\approx
\frac{1}{2}r^2.
$$

For large residuals:

$$
H_\delta(r)
\approx
\delta |r|-\delta^2.
$$

Thus, the loss behaves approximately like MSE for small errors and becomes more robust like MAE for large errors.

---

### 5. Complete RARPH Loss

The loss for a single observation is:

$$
L_i = w(y_i)\,a(r_i)\,H_\delta(r_i)
$$

and for $n$ observations:

$$
L_{\mathrm{RARPH}} = \frac{1}{n}
\sum_{i=1}^{n}
w(y_i)\,a(r_i)\,H_\delta(r_i)
$$

Therefore:

$$
\boxed{
\text{RARPH} = \text{Relative Error}
+
\text{Asymmetric Penalty}
+
\text{Risk Weighting}
+
\text{Robust Error Behaviour}
}
$$

---

## Why RARPH?

Standard regression losses treat errors primarily according to their numerical magnitude.

Consider an actual generator temperature of:

$$
y=95^\circ C
$$

and two predictions:

$$
\hat{y}_1=85^\circ C
$$

and

$$
\hat{y}_2=105^\circ C.
$$

Both predictions have an absolute error of:

$$
|y-\hat{y}|=10^\circ C.
$$

Therefore:

$$
\mathrm{MAE}=10
$$

and:

$$
\mathrm{MSE}=100
$$

for both predictions.

However, the two errors have different operational meanings:

- $85^\circ C$ is an **underprediction** and may fail to indicate a high-temperature condition.
- $105^\circ C$ is an **overprediction** and is comparatively conservative.

RARPH can explicitly assign a larger loss to the underprediction.

---

## Worked Example

Using the illustrative parameter values:

$$
\epsilon=0.001,
\quad
\delta=0.1,
\quad
c_u=2,
\quad
c_o=1,
$$

$$
\lambda=1,
\quad
k=0.5,
\quad
T_w=90^\circ C.
$$

### Risk Weight

At $y=95^\circ C$:

$$
w(95) = 1+\frac{1}{1+\exp[-0.5(95-90)]}
$$

$$
\approx1.92414.
$$

### Underprediction

For:

$$
y=95,\qquad\hat{y}=85
$$

the relative residual is:

$$
r = \frac{85-95}{95+0.001}\approx-0.105262.
$$

Since this is an underprediction:

$$
a(r)=c_u=2.
$$

The resulting RARPH loss is approximately:

$$
L_{\mathrm{under}}
\approx0.01739.
$$

### Overprediction

For:

$$
y=95,\qquad\hat{y}=105
$$

the relative residual is:

$$
r
\approx0.105262.
$$

Since this is an overprediction:

$$
a(r)=c_o=1.
$$

The resulting loss is approximately:

$$
L_{\mathrm{over}}
\approx0.00870.
$$

Hence:

$$
L_{\mathrm{under}}
\approx
2L_{\mathrm{over}}.
$$

This demonstrates how RARPH distinguishes between errors that have the same absolute magnitude but different operational consequences.

---

# Parameters

The RARPH loss contains seven principal parameters.

| Parameter | Meaning | How it is determined |
|---|---|---|
| $\epsilon$ | Numerical stabilizer for relative error | Sensor resolution or data scale |
| $c_u$ | Underprediction penalty | Relative operational cost |
| $c_o$ | Overprediction penalty | Reference penalty, usually normalised to 1 |
| $T_w$ | Warning/critical temperature | Engineering specification |
| $\lambda$ | Maximum additional risk weighting | Desired risk emphasis |
| $k$ | Sharpness of risk transition | Desired transition temperature range |
| $\delta$ | Pseudo-Huber transition scale | Residual distribution + validation |

---

## Parameter Estimation Strategy

The parameters are not intended to be selected arbitrarily. They are determined using a combination of **engineering information, operational requirements, training data, and validation**.

### Stabilizer $\epsilon$

If the temperature sensor resolution is known, a practical choice is:

$$
\epsilon
\approx
\text{sensor resolution}.
$$

For example, for a sensor resolution of $0.1^\circ C$:

$$
\epsilon=0.1.
$$

When the sensor resolution is unavailable, a small data-scaled value may be used as an initial choice.

---

### Asymmetric Coefficients $c_u$ and $c_o$

If the estimated operational cost of underprediction is $C_u$ and that of overprediction is $C_o$, use:

$$
\frac{c_u}{c_o} = \frac{C_u}{C_o}.
$$

Since only the relative magnitude matters, a convenient normalization is:

$$
c_o=1.
$$

Hence:

$$
c_u=\frac{C_u}{C_o}.
$$

For example, if underprediction is considered three times as costly:

$$
c_u=3,
\qquad
c_o=1.
$$

---

### Warning Temperature $T_w$

$T_w$ should preferably come from engineering information such as:

- manufacturer specifications,
- thermal protection thresholds,
- operating limits,
- engineering standards,
- historical failure analysis.

For example:

$$
T_w=90^\circ C.
$$

---

### Risk Sharpness $k$

The parameter $k$ controls how quickly the risk weight changes around $T_w$.

If the desired 10%-to-90% risk transition occurs across a temperature interval $\Delta T$, then:

$$
k = \frac{2\ln 9}{\Delta T}\approx\frac{4.394}{\Delta T}.
$$

For example, if the transition is desired between $85^\circ C$ and $95^\circ C$:

$$
\Delta T=10^\circ C
$$

and therefore:

$$
k
\approx
0.4394\;^\circ C^{-1}.
$$

---

### Risk Weight $\lambda$

For high temperatures:

$$
w(y)\rightarrow1+\lambda.
$$

Therefore, if the maximum desired risk weight is 3:

$$
1+\lambda=3
$$

and:

$$
\lambda=2.
$$

---

### Pseudo-Huber Parameter $\delta$

A robust residual scale can be estimated from the training residuals using the Median Absolute Deviation (MAD):

$$
MAD(r) = \text{median}
\left(|r_i-\text{median}(r)|\right).
$$

A robust scale estimate is:

$$
s_r=1.4826\,MAD(r).
$$

A practical initial value is:

$$
\delta\approx1.345s_r.
$$

The resulting value can then be refined using validation data.

---

## Example Parameter Set

One illustrative parameterisation is:

$$
\epsilon=0.001
$$

$$
c_u=2,\qquad c_o=1
$$

$$
T_w=90^\circ C
$$

$$
\lambda=1
$$

$$
k=0.5\;^\circ C^{-1}
$$

$$
\delta=0.1
$$

These values are provided for demonstration and are not universal constants. Actual values should be calibrated using the intended wind-turbine dataset and engineering requirements.

---

# Real-Time Python Implementation

The repository contains a real-time/iterative implementation that processes temperature observations sequentially.

Each observation follows the pipeline:

$$
\boxed{
\text{Actual Value}
\rightarrow
\text{Prediction}
\rightarrow
\text{RARPH}
\rightarrow
\text{Running Loss}
\rightarrow
\text{Live Plot}
}
$$

The implementation also compares the proposed loss with standard MAE and MSE.

---

## Repository Structure

```text
RARPH-Real-Time-Loss/
│
├── RARPH.py
├── main.py
├── requirements.txt
├── RARPH.pdf
└── README.md
```

### `RARPH.py`

Contains the implementation of the RARPH loss function.

### `main.py`

Runs the iterative demonstration using sample observations, calculates the RARPH loss, maintains the running average, and generates real-time plots.

### `requirements.txt`

Contains the external Python dependencies required to run the implementation.

---

# Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd RARPH-Real-Time-Loss
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Implementation

Run:

```bash
python main.py
```

The program processes observations sequentially and reports:

- observation number,
- actual temperature,
- predicted temperature,
- RARPH loss,
- running average RARPH loss.

Example output:

```text
Step 01 | Actual =   85.0 °C | Predicted =   72.0 °C | Loss = ...
-------------------------------------------------------------
Step 02 | Actual =   52.0 °C | Predicted =   44.0 °C | Loss = ...
-------------------------------------------------------------
Step 03 | Actual =   53.0 °C | Predicted =   45.0 °C | Loss = ...
-------------------------------------------------------------
...
```

A live plot is used to visualise the evolution of the actual/predicted temperatures and RARPH loss as observations arrive.

---

# Sample Data

The demonstration uses observations in the form:

```python
observations = [
    (85, 72),
    (52, 44),
    (53, 45),
    (25, 21),
    (35, 29),
    (46, 39),
    (37, 31),
    (22, 18),
    (79, 90),
    (28, 32),
]
```

Each tuple has the form:

```text
(actual_temperature, predicted_temperature)
```

The sample contains both underpredictions and overpredictions so that the asymmetric component of RARPH can be demonstrated.

---

# Comparison with Standard Regression Metrics

For the sample observations:

$$
\mathrm{MAE}=7.55
$$

and:

$$
\mathrm{MSE}=68.15.
$$

MAE and MSE are calculated using:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
```

These metrics are included as reference measures for comparison with RARPH.

---

# Properties of RARPH

For fixed positive parameters, the proposed loss has the following characteristics:

| Property | Behaviour |
|---|---|
| Relative error | Normalises error by actual magnitude |
| Asymmetry | Underprediction can receive greater penalty |
| Risk awareness | High-temperature observations receive greater weight |
| Small residuals | Approximately quadratic |
| Large residuals | Approximately linear |
| Differentiability | Differentiable with respect to prediction |
| Robustness | Less sensitive to extreme residuals than MSE |

The pseudo-Huber derivative is:

$$
H_\delta'(r) = \frac{r}{\sqrt{1+(r/\delta)^2}}.
$$

The pseudo-Huber function is smooth and convex.

For fixed parameters and a fixed target $y$, the proposed loss is convex with respect to the scalar prediction $\hat{y}$. However, when it is used inside a nonlinear neural network, the overall optimisation problem with respect to the model parameters is generally non-convex.

---

# Outlier Behaviour

MSE grows quadratically with the residual:

$$
L_{\mathrm{MSE}}\propto r^2.
$$

Therefore, very large residuals can dominate the objective.

RARPH uses pseudo-Huber behaviour:

$$
H_\delta(r)
\approx
\frac12r^2
\qquad
(|r|\ll\delta)
$$

and:

$$
H_\delta(r)
\approx
\delta|r|-\delta^2
\qquad
(|r|\gg\delta).
$$

Therefore, small errors retain smooth quadratic behaviour while very large errors have reduced influence.

---

# Rare High-Temperature Observations

Although the problem is a regression problem rather than a classification problem, high-temperature observations may be relatively rare.

For example:

```text
Normal operating temperatures
████████████████████████████████████

High-temperature observations
████
```

A conventional mean loss may be dominated by the large number of normal observations.

The risk-weighting function:

$$
w(y) = 1+\frac{\lambda}{1+\exp[-k(y-T_w)]}
$$

increases the contribution of observations near and above the warning temperature.

This is intended to make rare but operationally important observations more visible during optimisation.

---

# Limitations

RARPH is application-specific and is not intended to universally replace MSE or MAE.

Important limitations include:

1. The relative residual requires a stabilizing parameter $\epsilon$ when target values can approach zero.
2. The parameters $c_u$, $c_o$, $\lambda$, $k$, and $\delta$ require meaningful calibration.
3. The warning temperature $T_w$ is application-dependent.
4. A fixed asymmetric penalty assumes that underprediction and overprediction have genuinely different consequences.
5. Increasing the loss weight of high-temperature observations can alter the model's performance on normal operating conditions.
6. The loss should therefore be evaluated on a separate validation set before deployment.

---

# Reproducibility

The implementation is designed to be reproducible using the files in this repository.

```text
RARPH.py
    ↓
Custom loss function

main.py
    ↓
Sequential evaluation
    ↓
RARPH + MAE + MSE
    ↓
Real-time visualisation
```

The parameters used for the demonstration are explicitly documented in the source code and README.

---

# Assignment Context

This repository contains the computational implementation for:

**Assignment 3: Designing a Custom Loss Function**

The project demonstrates the design of a custom regression loss function that differs from conventional MSE and MAE by incorporating:

$$
\boxed{
\text{Relative Error}
+
\text{Asymmetry}
+
\text{Risk Awareness}
+
\text{Robustness}
}
$$

The implementation specifically addresses real-time prediction of wind-turbine generator temperature.

---

# Author

**[Soumitro Mukherjee](https://www.linkedin.com/in/soumitro-mukherjee-746487200/)**

**Course:** ME-781 — Statistical Machine Learning and Data Mining

---
