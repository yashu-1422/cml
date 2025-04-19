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

