import numpy as np

mu = 0.5
pi_values = np.array([5.0])  # μπ = 2.5
print('μπ =', mu * pi_values)
print('sin(μπ) =', np.sin(mu * pi_values))
print('sin²(μπ)/μ² =', (np.sin(mu * pi_values) / mu)**2)
print('Classical π²/2 =', pi_values**2 / 2)
print('Polymer energy - Classical =', (np.sin(mu * pi_values) / mu)**2 / 2 - pi_values**2 / 2)
