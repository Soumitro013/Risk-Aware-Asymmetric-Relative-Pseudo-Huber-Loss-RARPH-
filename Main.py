import RARPH
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Input observations, Format: (Actual Temperature, Predicted Temperature)
observations = [
    (85, 68),
    (52, 62),
    (53, 45),
    (25, 30),
    (35, 29),
    (46, 52),
    (37, 29),
    (22, 18),
    (79, 90),
    (28, 23),
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

y_true = [actual for actual, predicted in observations]
y_pred = [predicted for actual, predicted in observations]

steps = []
actual_values = []
predicted_values = []
loss_values = []
running_average_losses = []

running_loss = 0.0

# Plotting setup
plt.ion()

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 8)
)

# Plot 1: Actual vs Predicted
actual_line, = ax1.plot(
    [], [],
    marker='o',
    label='Actual Temperature'
)

predicted_line, = ax1.plot(
    [], [],
    marker='x',
    label='Predicted Temperature'
)

ax1.set_title("Real-Time Actual vs Predicted Temperature")
ax1.set_xlabel("Observation")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
ax1.grid(True)


# Plot 2: RARPH loss
loss_line, = ax2.plot(
    [], [],
    marker='o',
    label='RARPH Loss'
)

avg_loss_line, = ax2.plot(
    [], [],
    marker='x',
    label='Running Average RARPH Loss'
)

ax2.set_title("Real-Time RARPH Loss")
ax2.set_xlabel("Observation")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

for t, (actual, predicted) in enumerate(observations, start=1):

    # Calculate custom loss
    loss = RARPH.rarph_loss(actual, predicted)

    # Update running loss
    running_loss += loss
    average_loss = running_loss / t

    # Store values
    steps.append(t)
    actual_values.append(actual)
    predicted_values.append(predicted)
    loss_values.append(loss)
    running_average_losses.append(average_loss)

    # Update plots
    actual_line.set_data(
        steps,
        actual_values
    )
    # Update predicted temperature plot
    predicted_line.set_data(
        steps,
        predicted_values
    )

    # Update RARPH loss plot
    loss_line.set_data(
        steps,
        loss_values
    )
    # Update running average RARPH loss plot
    avg_loss_line.set_data(
        steps,
        running_average_losses
    )

    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Print current step information
    print(
        f"Step {t:02d} | "
        f"Actual = {actual:6.1f} °C | "
        f"Predicted = {predicted:6.1f} °C | "
        f"RARPH Loss = {loss:.6f} | "
        f"Running Avg = {average_loss:.6f}"
    )

    print("-" * 80)

    # Simulate time delay
    time.sleep(0.5)

plt.ioff()

plt.show()

# Final performance metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE")
print("=" * 80)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Mean RARPH Loss:           {running_loss / len(observations):.6f}")