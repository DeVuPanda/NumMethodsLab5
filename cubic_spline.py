import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

x = np.array([0, 0.15, 0.35, 0.6, 0.9, 1.1, 1.4, 1.57, 1.8, 2, 2.3, 2.6, 2.8, 3, 3.1])
y = np.array([1, 0.988, 0.94, 0.825, 0.621, 0.453, 0.169, 0, -0.227, -0.416, -0.669, -0.857, -0.942, -0.99, -1])

n = len(x) - 1

h = [x[i + 1] - x[i] for i in range(n)]

C = np.zeros((n - 1, n - 1))
for i in range(n - 1):
    if i > 0:
        C[i, i - 1] = h[i] / 6
    C[i, i] = (h[i] + h[i + 1]) / 3
    if i < n - 2:
        C[i, i + 1] = h[i + 1] / 6

H = np.zeros((n - 1, n + 1))
for i in range(n - 1):
    H[i, i] = 1 / h[i]
    H[i, i + 1] = -(1 / h[i] + 1 / h[i + 1])
    H[i, i + 2] = 1 / h[i + 1]

b = H @ y

m = np.zeros(n + 1)
m[1:n] = solve(C, b)

A = []
B = []
for i in range(n):
    A_i = ((h[i] ** 2) / 6) * m[i] + y[i]
    B_i = ((h[i] ** 2) / 6) * m[i + 1] + y[i + 1]
    A.append(A_i)
    B.append(B_i)

def calculate_spline(x_val, i, x_i, x_i_prev, h_i, m_i_prev, m_i, A_i, B_i):
    term1 = m_i_prev * ((x_i - x_val) ** 3) / (6 * h_i)
    term2 = m_i * ((x_val - x_i_prev) ** 3) / (6 * h_i)
    term3 = A_i * (x_i - x_val) / h_i
    term4 = B_i * (x_val - x_i_prev) / h_i
    return term1 + term2 + term3 + term4

def calculate_first_derivative(x_val, i, x_i, x_i_prev, h_i, m_i_prev, m_i, A_i, B_i):
    term1 = -m_i_prev * ((x_i - x_val) ** 2) / (2 * h_i)
    term2 = m_i * ((x_val - x_i_prev) ** 2) / (2 * h_i)
    term3 = -A_i / h_i
    term4 = B_i / h_i
    return term1 + term2 + term3 + term4


def calculate_second_derivative(x_val, i, x_i, x_i_prev, h_i, m_i_prev, m_i):
    term1 = m_i_prev * (x_i - x_val) / h_i
    term2 = m_i * (x_val - x_i_prev) / h_i
    return term1 + term2

print("")
print("h_i:", [float(round(val, 2)) for val in h])

print("\nМатриця C:")
for row in C:
    print(f"[{' '.join(f'{val:.8f}' for val in row)}]")

print("\nМатриця H:")
for row in H:
    print(f"[{' '.join(f'{val:.8f}' for val in row)}]")

print("\nВектор b:")
print([float(f"{val:.8f}") for val in b])

print("\nВектор m:")
print([float(f"{val:.8f}") for val in m])

print("\nРезультати A і B для кожного інтервала:")
for i in range(n):
    print(f"Інтервал [{x[i]:.2f}, {x[i + 1]:.2f}]: A = {A[i]:.8f}, B = {B[i]:.8f}")

print("\nФункція сплайна S(x):")
print("S(x) = {")
for i in range(n):
    print(
        f"    {m[i]:.8f} * (({x[i + 1]:.2f} - x)^3)/(6 * {h[i]:.2f}) + {m[i + 1]:.8f} * ((x - {x[i]:.2f})^3)/(6 * {h[i]:.2f}) + {A[i]:.8f} * ({x[i + 1]:.2f} - x)/{h[i]:.2f} + {B[i]:.8f} * (x - {x[i]:.2f})/{h[i]:.2f}, x ∈ [{x[i]:.2f}; {x[i + 1]:.2f}]")
print("}")

print("\nПерша похідна S'(x):")
print("S'(x) = {")
for i in range(n):
    print(
        f"    {-m[i]:.8f} * (({x[i + 1]:.2f} - x)^2)/(2 * {h[i]:.2f}) + {m[i + 1]:.8f} * ((x - {x[i]:.2f})^2)/(2 * {h[i]:.2f}) + {-A[i]:.8f}/{h[i]:.2f} + {B[i]:.8f}/{h[i]:.2f}, x ∈ [{x[i]:.2f}; {x[i + 1]:.2f}]")
print("}")

print("\nДруга похідна S''(x):")
print("S''(x) = {")
for i in range(n):
    print(
        f"    {m[i]:.8f} * ({x[i + 1]:.2f} - x)/{h[i]:.2f} + {m[i + 1]:.8f} * (x - {x[i]:.2f})/{h[i]:.2f}, x ∈ [{x[i]:.2f}; {x[i + 1]:.2f}]")
print("}")

x_plot = np.linspace(0, 3.1, 500)
y_spline = []
y_derivative1 = []
y_derivative2 = []
y_cos = np.cos(x_plot)

for x_val in x_plot:
    i = 0
    while i < n and x_val > x[i + 1]:
        i += 1
    if i >= n:
        i = n - 1

    s = calculate_spline(x_val, i, x[i + 1], x[i], h[i], m[i], m[i + 1], A[i], B[i])
    d1 = calculate_first_derivative(x_val, i, x[i + 1], x[i], h[i], m[i], m[i + 1], A[i], B[i])
    d2 = calculate_second_derivative(x_val, i, x[i + 1], x[i], h[i], m[i], m[i + 1])

    y_spline.append(s)
    y_derivative1.append(d1)
    y_derivative2.append(d2)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(x_plot, y_cos, 'b-', label='cos(x)')
plt.plot(x_plot, y_spline, 'r-', label='S(x)')
plt.plot(x, y, 'ko', label='Вузли інтерполяції')
plt.grid(True)
plt.legend()
plt.title('cos(x) и інтерполяційний сплайн S(x)')

plt.subplot(3, 1, 2)
plt.plot(x_plot, y_derivative1, 'g-', label='S\'(x)')
plt.grid(True)
plt.legend()
plt.title('Перша похідна S\'(x)')

plt.subplot(3, 1, 3)
plt.plot(x_plot, y_derivative2, 'm-', label='S\'\'(x)')
plt.grid(True)
plt.legend()
plt.title('Друга похідна S\'\'(x)')

plt.tight_layout()
plt.show()