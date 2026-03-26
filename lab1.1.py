import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import lagrange
import numpy.polynomial.polynomial as poly

def f(x):
    """Исходная функция"""
    return x**3 - 3*x**2 - 11.25*x + 20.25

def chord_method(f, a, b, eps=1e-6, max_iter=100):
    """
    Метод хорд для решения уравнения f(x) = 0
    """
    iterations = 0
    x_prev = a
    
    for i in range(max_iter):
        # Вычисляем значение функции на концах интервала
        fa = f(a)
        fb = f(b)
        
        # Находим точку пересечения хорды с осью X
        x = a - fa * (b - a) / (fb - fa)
        
        # Проверка на достижение точности
        if abs(f(x)) < eps or abs(x - x_prev) < eps:
            return x, iterations + 1
        
        # Выбор нового интервала
        if fa * f(x) < 0:
            b = x
        else:
            a = x
        
        x_prev = x
        iterations += 1
    
    return x, iterations

# 1.1 Отделение корней графическим способом
x_vals = np.linspace(-5, 6, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(12, 8))

# График функции для отделения корней
plt.subplot(2, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.title('График функции f(x) = x³ - 3x² - 11.25x + 20.25')
plt.xlabel('x')
plt.ylabel('f(x)')

# Определяем интервалы, где функция меняет знак
intervals = []
for i in range(len(x_vals)-1):
    if y_vals[i] * y_vals[i+1] <= 0:
        intervals.append((x_vals[i], x_vals[i+1]))

print("1.1 Интервалы локализации корней:")
for i, interval in enumerate(intervals):
    print(f"Корень {i+1}: [{interval[0]:.2f}, {interval[1]:.2f}]")

# 1.2 Нахождение корней методом хорд
roots_chord = []
for interval in intervals:
    root, iterations = chord_method(f, interval[0], interval[1])
    roots_chord.append(root)
    print(f"\nКорень {len(roots_chord)} (метод хорд): x = {root:.6f}")
    print(f"Значение функции: f(x) = {f(root):.2e}")
    print(f"Количество итераций: {iterations}")

# 1.4 Сравнение с библиотечными функциями
roots_scipy = []
for interval in intervals:
    root = optimize.root_scalar(f, bracket=interval, method='bisect')
    roots_scipy.append(root.root)
    print(f"\nКорень {len(roots_scipy)} (scipy): x = {root.root:.6f}")
    print(f"Значение функции: f(x) = {f(root.root):.2e}")

# График с отмеченными корнями
plt.subplot(2, 2, 2)
x_detailed = np.linspace(-4, 5, 1000)
plt.plot(x_detailed, f(x_detailed), 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Отмечаем найденные корни
for i, root in enumerate(roots_chord):
    plt.plot(root, f(root), 'ro', markersize=8)
    plt.annotate(f'x{i+1}={root:.3f}', (root, f(root)), 
                xytext=(10, 10), textcoords='offset points')

plt.title('Корни уравнения, найденные методом хорд')
plt.xlabel('x')
plt.ylabel('f(x)')

# 1.3 Интерполяция
print("\n" + "="*50)
print("1.3 Интерполяция")

# Создаем точки для интерполяции
x_points = np.array([-3, -1, 2, 4])
y_points = f(x_points)

# Строим интерполяционный полином Лагранжа с помощью scipy
poly_lagrange = lagrange(x_points, y_points)

# Создаем более детальную сетку для построения графика
x_interp = np.linspace(-4, 5, 1000)
y_interp = poly_lagrange(x_interp)

plt.subplot(2, 2, 3)
plt.plot(x_interp, y_interp, 'g-', linewidth=2)
plt.plot(x_points, y_points, 'ro', markersize=8)
plt.grid(True, alpha=0.3)
plt.title('Интерполяционный полином Лагранжа')
plt.xlabel('x')
plt.ylabel('y')

# Ввод точки пользователем
print("\nВведите координату x для интерполяции:")
try:
    user_x = float(input("x = "))
    user_y = poly_lagrange(user_x)
    
    print(f"Значение интерполяционного полинома в точке x = {user_x}: y = {user_y:.6f}")
    print(f"Значение исходной функции: f(x) = {f(user_x):.6f}")
    print(f"Погрешность: {abs(user_y - f(user_x)):.2e}")
    
    # Добавляем точку пользователя на график
    plt.plot(user_x, user_y, 'bs', markersize=10)
    
except ValueError:
    print("Ошибка ввода! Будет использовано значение по умолчанию x=1.5")
    user_x = 1.5
    user_y = poly_lagrange(user_x)
    print(f"Значение интерполяционного полинома в точке x = {user_x}: y = {user_y:.6f}")
    print(f"Значение исходной функции: f(x) = {f(user_x):.6f}")
    print(f"Погрешность: {abs(user_y - f(user_x)):.2e}")
    plt.plot(user_x, user_y, 'bs', markersize=10)

# Сравнение методов
plt.subplot(2, 2, 4)
methods = ['Метод хорд', 'SciPy']
roots_compare = [roots_chord, roots_scipy]

x_pos = np.arange(len(roots_chord))
width = 0.35

for i, root in enumerate(roots_chord):
    plt.bar(i - width/2, root, width, alpha=0.7)

for i, root in enumerate(roots_scipy):
    plt.bar(i + width/2, root, width, alpha=0.7)

plt.xlabel('Номер корня')
plt.ylabel('Значение корня')
plt.title('Сравнение найденных корней')
plt.xticks(x_pos, [f'Корень {i+1}' for i in range(len(roots_chord))])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Дополнительная проверка и анализ
print("\n" + "="*50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print("="*50)

print("\nСравнение методов нахождения корней:")
print("-" * 50)
print(f"{'Корень':<10} {'Метод хорд':<15} {'SciPy':<15} {'Разница':<15}")
print("-" * 50)
for i in range(len(roots_chord)):
    diff = abs(roots_chord[i] - roots_scipy[i])
    print(f"{i+1:<10} {roots_chord[i]:<15.6f} {roots_scipy[i]:<15.6f} {diff:<15.2e}")

print("\nПроверка подстановкой корней в уравнение:")
print("-" * 50)
for i, root in enumerate(roots_chord):
    print(f"Корень {i+1}: x = {root:.6f}, f(x) = {f(root):.2e}")