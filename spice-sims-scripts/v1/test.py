import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save the plot as an SVG file
plt.savefig('sine_wave_plot.svg', format='svg')

# Optionally, display the plot
plt.show()