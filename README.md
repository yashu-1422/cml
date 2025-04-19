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
# sd and cv
```python
#grouped frequency data
x = np.array([1, 2, 3, 4, 5])
f = np.array([2, 3, 4, 5, 6])
N = np.sum(f)
mean = np.sum(x * f) / N
variance = np.sum(f * (x - mean)**2) / N
std_dev = np.sqrt(variance)
cv = (std_dev / mean) * 100

print(f"Mean: {mean:.2f}, SD: {std_dev:.2f}, CV: {cv:.2f}%")
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
**End Semester Exam (ESE) Preparation: Probable Questions and Programs with Python Code**

---

### ✅ 1. LU Decomposition / Gauss Elimination (5 Marks)

**LU Decomposition (using SciPy):**
```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 3, 1], [1, 2, 3], [3, 1, 2]])
P, L, U = lu(A)

print("L:\n", L)
print("U:\n", U)
```



### ✅ 2. Trapezoidal Rule (5 Marks)
```python
import numpy as np

def f(x):
    return x**2 + 3*x + 2

a, b = 1, 5
n = 4
h = (b - a) / n
x = np.linspace(a, b, n+1)
y = f(x)

integral = (h/2) * (y[0] + 2*sum(y[1:n]) + y[n])
print(f"Approximate integral using Trapezoidal Rule: {integral:.4f}")
```

---

### ✅ 3. SD and CV (15 Marks)

**Raw Data:**
```python
import numpy as np

data = np.array([10, 12, 14, 16, 18])
mean = np.mean(data)
std_dev = np.std(data, ddof=1)
cv = (std_dev / mean) * 100

print(f"Mean: {mean:.2f}, SD: {std_dev:.2f}, CV: {cv:.2f}%")
```

**Grouped Frequency Data:**
```python
x = np.array([1, 2, 3, 4, 5])
f = np.array([2, 3, 4, 5, 6])
N = np.sum(f)
mean = np.sum(x * f) / N
variance = np.sum(f * (x - mean)**2) / N
std_dev = np.sqrt(variance)
cv = (std_dev / mean) * 100

print(f"Mean: {mean:.2f}, SD: {std_dev:.2f}, CV: {cv:.2f}%")
```

---

### ✅ 4. Moments, Skewness, Kurtosis (15 Marks)

**First Four Moments:**
```python
x = np.array([0, 10, 20, 30, 40])
f = np.array([6, 26, 47, 15, 6])
mid = x
N = np.sum(f)
mean = np.sum(mid * f) / N

mu2 = np.sum(f * (mid - mean)**2) / N
mu3 = np.sum(f * (mid - mean)**3) / N
mu4 = np.sum(f * (mid - mean)**4) / N

print(f"Mean: {mean:.2f}\nMu2: {mu2:.2f}\nMu3: {mu3:.2f}\nMu4: {mu4:.2f}")
```

**Skewness and Kurtosis:**
```python
from scipy.stats import skew, kurtosis

data = np.repeat([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
print(f"Skewness: {skew(data):.4f}, Kurtosis: {kurtosis(data):.4f}")
```

---

### ✅ 5. Simple Correlation & Simple Regression (15 Marks)

**Correlation:**
```python
from scipy.stats import pearsonr

x = [23, 48, 42, 17, 26, 35, 29, 37, 16, 46]
y = [25, 22, 38, 21, 27, 39, 24, 32, 18, 44]

corr, _ = pearsonr(x, y)
print(f"Correlation Coefficient: {corr:.4f}")
```

**Simple Linear Regression:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([5, 10, 15, 20, 25, 30]).reshape(-1, 1)
y = np.array([40, 30, 25, 40, 18, 20])

model = LinearRegression()
model.fit(x, y)

print(f"Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
```

---

### ✅ 6. Binomial & Poisson Distributions (15 Marks)

**Binomial Distribution:**
```python
from scipy.stats import binom

n = 5
p = 0.5
x = np.arange(0, n+1)
probs = binom.pmf(x, n, p)

print("Binomial Probabilities:", probs)
```

**Poisson Distribution:**
```python
from scipy.stats import poisson

lam = 2
x = np.arange(0, 10)
probs = poisson.pmf(x, lam)

print("Poisson Probabilities:", probs)
```

---

### ✅ 7. Curve Fitting – Straight Line by Least Squares

```python
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0, 2, 4, 6, 8, 12, 20]).reshape(-1, 1)
y = np.array([10, 12, 18, 22, 20, 30, 30])

model = LinearRegression()
model.fit(x, y)

print(f"Best Fit Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
```


