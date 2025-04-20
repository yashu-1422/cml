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


