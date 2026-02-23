from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EPS = 0.0001


def f(x: float):
    return pow(x, 3) - 6 * pow(x, 2) + 5 * x


def analytic_hardcode_derivative(x: float):
    return 3 * pow(x, 2) - 12 * x + 5


def numeric_derivative(f: callable, x: float):
    return (f(x + EPS) - f(x - EPS)) / (2 * EPS)


def sign(val: float):
    if val > 0: return 1
    if val < 0: return -1
    return val


def find_all_roots(method: callable, f: callable, left_border: float, right_border: float,
                   roots_intervals: List[float] = None,
                   iterations: int = 1000,
                   borders_accuracy_step: float = 0.5):
    if not roots_intervals:
        parts = find_roots_intervals(f, left_border, right_border, borders_accuracy_step)
    else:
        parts = roots_intervals
    roots = []
    i = 1
    while i < len(parts):
        root = method(f, parts[i - 1], parts[i], iterations)
        roots.append(root)
        i += 1

    return roots


def find_roots_intervals(f: callable, left_border: float, right_border: float,
                         borders_accuracy_step: float = 0.5):
    parts = [left_border, ]
    current_right = left_border + borders_accuracy_step
    while current_right < right_border:
        l = numeric_derivative(f, current_right - borders_accuracy_step)
        r = numeric_derivative(f, current_right)
        if l * r <= 0:
            parts.append(
                bisection_method(lambda x: numeric_derivative(f, x), current_right - borders_accuracy_step,
                                 current_right,
                                 1000))
            current_right += borders_accuracy_step
        current_right += borders_accuracy_step
    parts.append(right_border)
    return parts


def draw_function_graphic(f: callable, left_border: float, right_border: float, roots_intervals: List[float],
                          step: float = 0.1):
    if not roots_intervals:
        roots_intervals = []
    x_points = []
    y_points = []

    current_x = left_border
    while current_x <= right_border:
        x_points.append(current_x)
        y_points.append(f(current_x))
        current_x += step

    x_root_intervals_points = []
    y_root_intervals_points = []

    for current_x in roots_intervals:
        x_root_intervals_points.append(current_x)
        y_root_intervals_points.append(f(current_x))

    create_visualization_graphic(x_points, y_points, x_root_intervals_points, y_root_intervals_points)


def create_visualization_graphic(x_points: List[float], y_points: List[float], x_root_intervals_points: List[float],
                                 y_root_intervals_points: List[float]):
    data = pd.DataFrame({'x': x_points, 'y': y_points})

    sns.set_theme(style="ticks")
    sns.set_palette("muted")

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(data=data, x='x', y='y',
                 linewidth=2.5,
                 ax=ax)
    __highlight_intervals_borders(x_root_intervals_points, y_root_intervals_points)
    sns.despine(top=True, right=True)

    ax.set_title('График функции', fontsize=16, fontweight='light', pad=20)
    ax.set_xlabel('Ось X', fontsize=12)
    ax.set_ylabel('Ось Y', fontsize=12)

    ax.grid(True, linestyle='-', alpha=0.3, color='gray')

    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.show()


def __highlight_intervals_borders(x_root_intervals_points: List[float], y_root_intervals_points: List[float]):
    plt.scatter(x_root_intervals_points, y_root_intervals_points,
                color='orange', s=200,
                linewidth=4,
                marker='|',
                zorder=5)


def create_comparison_graphic(methods: List[callable], f: callable, borders: List[float], colors: List[str],
                              legend: List[str],
                              iterations_interval: List[int],
                              roots: List[float], step: int = 2, custom_scale: str = None):
    if len(iterations_interval) != 2: raise ValueError('iterations_interval must contain [a, b] only')

    sns.set_theme(style="ticks")
    sns.set_palette("muted")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('Сравнительный график точности', fontsize=16, fontweight='light', pad=20)
    ax.set_xlabel('Число итераций', fontsize=12)
    ax.set_ylabel('Ошибка E суммарная', fontsize=12)
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=11)
    if custom_scale:
        ax.set_yscale(custom_scale)

    iterations = [_ for _ in range(iterations_interval[0], iterations_interval[1], step)]
    parts = find_roots_intervals(f, borders[0], borders[1])
    for i in range(len(methods)):
        __draw_accuracy_graphic(methods[i], f, parts, colors[i], legend[i], iterations, roots, ax)

    plt.tight_layout()
    plt.legend()
    plt.show()


def __draw_accuracy_graphic(method: callable, f: callable, parts: List[float], color: str, legend: str,
                            iterations: List[int],
                            roots: List[float], ax):
    points_x = []
    points_y = []
    for i in iterations:
        res = find_all_roots(method, f, 0, 0, parts, i)
        error = sum(abs(roots[i] - res[i]) for i in range(len(res)))
        points_x.append(i)
        points_y.append(error)

    data = pd.DataFrame({'x': points_x, 'y': points_y})
    sns.lineplot(data=data, x='x', y='y',
                 linewidth=2.5,
                 color=color,
                 label=legend,
                 ax=ax)


def bisection_method(f: callable, left_border: float, right_border: float, iterations: int):
    l = left_border
    val_l = f(l)
    r = right_border
    val_r = f(r)

    cnt_it = iterations
    while cnt_it:
        c = l + (r - l) / 2
        val = f(c)
        if sign(val) == sign(val_l):
            l = c
            val_l = val
        else:
            r = c
            val_r = val

        cnt_it -= 1

    return l + (r - l) / 2


def chord_method(f: callable, left_border: float, right_border: float, iterations: int):
    l = left_border
    val_l = f(l)
    r = right_border
    val_r = f(r)
    c = l - val_l * (r - l) / (val_r - val_l)

    cnt_it = iterations
    while cnt_it:
        c = l - val_l * (r - l) / (val_r - val_l)
        val = f(c)
        if sign(val) == sign(val_l):
            l = c
            val_l = val
        else:
            r = c
            val_r = val

        cnt_it -= 1

    return c


def newton_method(f: callable, left_border: float, right_border: float, iterations: int):
    x = left_border + (right_border - left_border) / 2
    cnt_it = iterations

    while cnt_it:
        x = x - f(x) / analytic_hardcode_derivative(x)
        cnt_it -= 1

    return x


def is_between(s: float, l: float, r: float):
    return l <= s <= r


if __name__ == '__main__':
    # строю график функции
    draw_function_graphic(f, -1, 6, roots_intervals=find_roots_intervals(f, -1, 6))
    print(find_all_roots(bisection_method, f, -1, 6, None, iterations=5))
    print(find_all_roots(chord_method, f, -1, 6, None, iterations=5))
    print(find_all_roots(newton_method, f, -1, 6, None, iterations=5))
    create_comparison_graphic([bisection_method, chord_method, newton_method], f, [-1, 6], ['red', 'green', 'blue'],
                              ['Метод деления отрезка пополам', 'Метод хорд', 'Метод Ньютона'],
                              [1, 17], [0, 1, 5], step=1)
    create_comparison_graphic([bisection_method, chord_method, newton_method], f, [-1, 6], ['red', 'green', 'blue'],
                              ['Метод деления отрезка пополам', 'Метод хорд', 'Метод Ньютона'],
                              [1, 50], [0, 1, 5], step=1, custom_scale='log')
