# cml

# gauss elemenation - Program 1
```python
import numpy as np
from scipy.linalg import solve

A = np.array([[2, 2, 3], [4, -2, 1], [1, 5, 4]], dtype=float)
B = np.array([4, 9, 3], dtype=float)

print("Solution:", solve(A, B))
```

# lu decomposition - Program 2
```python
# lu decomposition
from scipy.linalg import lu_factor, lu_solve

A = np.array([[2, 2, 3], [4, -2, 1], [1, 5, 4]], dtype=float)
B = np.array([4, 9, 3], dtype=float)

LU, piv = lu_factor(A)
solution = lu_solve((LU, piv), B)

print("Solution:", solution)
```
# trapezoidal rule - Program 3
```python
import numpy as np 

x = np.linspace(0, 1, 5)  
y = x**2                 

area = np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)

print("Area under the curve:", area)
```
# sd and cv
```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
f = np.array([2, 3, 4, 5, 6])

N = np.sum(f)
mean = np.sum(x * f) / N
variance = np.sum(f * (x - mean)**2) / N
std_dev = np.sqrt(variance)
cv = (std_dev / mean) * 100

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Coefficient of Variation: {cv:.2f}%")

```
# central moments
```python
import numpy as np

# Class intervals and frequencies
classes = [(0,10), (10,20), (20,30), (30,40), (40,50)]
frequencies = np.array([6, 26, 47, 15, 6])

# Midpoints of each class
midpoints = np.array([(a + b) / 2 for a, b in classes])

# Mean
N = np.sum(frequencies)
mean = np.sum(midpoints * frequencies) / N

# Central moments
def central_moment(k):
    return np.sum(frequencies * (midpoints - mean) ** k) / N

mu1 = central_moment(1)
mu2 = central_moment(2)
mu3 = central_moment(3)
mu4 = central_moment(4)

print(f"Mean: {mean:.2f}")
print(f"μ1: {mu1:.2f}")
print(f"μ2 (Variance): {mu2:.2f}")
print(f"μ3: {mu3:.2f}")
print(f"μ4: {mu4:.2f}")

```
# skewness
```python
x = np.array([4, 5, 6, 7, 8, 9, 10])
f = np.array([2, 3, 2, 5, 3, 4, 2])
N = np.sum(f)
mean = np.sum(x * f) / N

# Central moments
mu2 = np.sum(f * (x - mean)**2) / N
mu3 = np.sum(f * (x - mean)**3) / N

# Skewness (Pearson's 2nd skewness coefficient)
skewness = mu3 / (mu2 ** 1.5)

print(f"Mean: {mean:.2f}")
print(f"Skewness: {skewness:.4f}")

```
# kurtosis
```python
x = np.array([1, 2, 3, 4, 5])
f = np.array([2, 3, 4, 5, 6])
N = np.sum(f)
mean = np.sum(x * f) / N

# Central moments
mu2 = np.sum(f * (x - mean)**2) / N
mu4 = np.sum(f * (x - mean)**4) / N

# Kurtosis (Excess)
kurtosis = mu4 / (mu2 ** 2)

print(f"Mean: {mean:.2f}")
print(f"Kurtosis: {kurtosis:.4f}")

```

