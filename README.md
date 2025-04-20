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
# trapezoidal rule - Program 1
```python
import numpy as np 

x = np.linspace(0, 1, 5)  
y = x**2                 

area = np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)

print("Area under the curve:", area)
```
# trapezoidal rule - 2
question - integral 0 to 2 (1/1+X**2)dx h=0.25
```python
import numpy as np

# Step size
h = 0.25

# Generate x values from 0 to 2 with step h
x = np.arange(0, 2 + h, h)

# Define the function f(x) = 1 / (1 + x^2)
y = 1 / (1 + x**2)

# Apply Trapezoidal Rule
area = np.sum((y[1:] + y[:-1]) * h / 2)

print("Approximate area under the curve:", area)

```
# trapezoidal rule - 3
question - Compute the value of the integral numerically ∫ [log(𝑥 + 1) + sin 2𝑥]𝑑𝑥 by using trapezoidal rule taking ℎ = 0.1. 

```python
import numpy as np

# Step size
h = 0.1

# Generate x values from 0 to 0.8 with step h
x = np.arange(0, 0.8 + h, h)

# Define the function f(x) = ln(x+1) + sin(2x)
y = np.log(x + 1) + np.sin(2 * x)

# Apply Trapezoidal Rule
area = np.sum((y[1:] + y[:-1]) * h / 2)

print("Approximate value of the integral:", area)


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

**Simple Linear Regression: 2**
Write python program to obtain regression line y on x for the following data:
X 16 22 36 44 48
Y 29 34 45 38 47
Also find value of y for 𝑥 = 24. Attach the print of output. 

```python
X = np.array([16, 22, 36, 44, 48]).reshape(-1, 1)
Y = np.array([29, 34, 45, 38, 47])

model = LinearRegression()
model.fit(X, Y)

# Predict y for x = 24
x_val = 24
y_pred = model.predict([[x_val]])

print(f"Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
print(f"Predicted y for x = 24: {y_pred[0]:.2f}")

```

### ✅ 6. Binomial & Poisson Distributions (15 Marks)

**Binomial Distribution:**
```python
from scipy.stats import binom

n = 6
p = 0.5

# (1) P(X = 5)
prob_5 = binom.pmf(5, n, p)

# (2) P(X ≥ 5) = P(5) + P(6)
prob_at_least_5 = binom.pmf(5, n, p) + binom.pmf(6, n, p)

# (3) P(X ≤ 5)
prob_at_most_5 = binom.cdf(5, n, p)

print(f"P(X=5): {prob_5:.4f}")
print(f"P(X≥5): {prob_at_least_5:.4f}")
print(f"P(X≤5): {prob_at_most_5:.4f}")

```

**Poisson Distribution:**
```python
from scipy.stats import poisson
import numpy as np

# Given data
x = np.array([0, 1, 2, 3, 4, 5])
f = np.array([142, 158, 67, 27, 5, 1])
total = np.sum(f)

# Step 1: Calculate mean (λ)
lam = np.sum(x * f) / total
print("Mean (λ):", round(lam, 4))

# Step 2: Calculate Poisson probabilities for x values
probs = poisson.pmf(x, lam)

# Step 3: Calculate expected frequencies
expected_freq = total * probs

# Step 4: Display results
print("\nx\tObserved f\tPoisson Prob\tExpected f")
for xi, fi, pi, ei in zip(x, f, probs, expected_freq):
    print(f"{xi}\t{fi}\t\t{pi:.4f}\t\t{ei:.2f}")

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


