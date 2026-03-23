from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from matplotlib import pyplot as plt

ANALYTIC_VALUE = -18.11347301218


class NumericalIntegrator(ABC):
    def __init__(self, borders: [], func: callable, intervals_cnt: int):
        self.borders = borders
        self.func = func
        self.intervals_cnt = intervals_cnt
        self._build_intervals()

    def _build_intervals(self):
        l, r = self.borders
        self.intervals = []
        step = (r - l) / self.intervals_cnt
        for i in range(self.intervals_cnt):
            self.intervals.append((l + i * step, l + (i + 1) * step))

    @abstractmethod
    def integrate(self):
        pass


class LeftRectangleMethod(NumericalIntegrator):
    def integrate(self):
        return sum(self.func(interval[0]) * (interval[1] - interval[0]) for interval in self.intervals)


class RightRectangleMethod(NumericalIntegrator):
    def integrate(self):
        return sum(self.func(interval[1]) * (interval[1] - interval[0]) for interval in self.intervals)


class MiddleRectangleMethod(NumericalIntegrator):
    def integrate(self):
        return sum(self.func((interval[1] + interval[0]) / 2) * (interval[1] - interval[0]) for interval in
                   self.intervals)


class TrapezoidMethod(NumericalIntegrator):
    def integrate(self):
        return sum((self.func(interval[1]) + self.func(interval[0])) / 2 * (interval[1] - interval[0]) for interval in
                   self.intervals)


class SimpsonMethod(NumericalIntegrator):
    def integrate(self):
        return sum(
            (self.func(interval[0]) + 4 * self.func(interval[1]) + self.func(interval[2])) / 6 * (
                    interval[2] - interval[0]) for interval in
            self.intervals)

    def _build_intervals(self):
        if self.intervals_cnt % 2 != 0:
            raise ValueError('В методе Симпсона число интервалов должно быть четно')
        l, r = self.borders
        self.intervals = []
        step = (r - l) / self.intervals_cnt
        for i in range(self.intervals_cnt // 2):
            self.intervals.append((l + (i * 2 * step), (l + ((i * 2 + 1) * step)), (l + ((i * 2 + 2) * step))))


def f(x: int):
    return np.pow(x, 2) * np.sin(x)


def plot_errors(
        integrator_class: Type,
        exact_value: float,
        a: float,
        b: float,
        func: callable,
        n_values: [int],
        method_name: str = "",
        y_lim_absolute: tuple[float] = (0, 20),
        y_lim_rel: tuple[float] = (1e-3, 3),
):
    abs_errors = []
    rel_errors = []

    for n in n_values:
        integrator = integrator_class([a, b], func, n)
        approx = integrator.integrate()

        abs_error = abs(approx - exact_value)
        rel_error = abs_error / abs(exact_value) if exact_value != 0 else abs_error

        abs_errors.append(abs_error)
        rel_errors.append(rel_error)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(n_values, abs_errors, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Количество интервалов n', fontsize=12)
    ax1.set_ylabel('Абсолютная погрешность', fontsize=12)
    ax1.set_title(f'Абсолютная погрешность\n{method_name}', fontsize=12)
    if y_lim_absolute:
        ax1.set_ylim(*y_lim_absolute)
    ax1.grid(True, alpha=0.3)

    ax2.plot(n_values, rel_errors, 'o-', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Количество интервалов n', fontsize=12)
    ax2.set_ylabel('Относительная погрешность', fontsize=12)
    ax2.set_title(f'Относительная погрешность\n{method_name}', fontsize=12)
    if y_lim_rel:
        ax2.set_ylim(*y_lim_rel)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return abs_errors, rel_errors


def plot_all_methods_errors(
        integrator_classes: [],
        method_names: [str],
        exact_value: float,
        a: float,
        b: float,
        func: callable,
        n_values: [int],
        y_lim_absolute: tuple[float] = (0, 20),
        y_lim_rel: tuple[float] = (1e-3, 3),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for integrator_class, method_name, color in zip(integrator_classes, method_names, colors):
        abs_errors = []
        rel_errors = []

        for n in n_values:
            integrator = integrator_class([a, b], func, n)
            approx = integrator.integrate()

            abs_error = abs(approx - exact_value)
            rel_error = abs_error / abs(exact_value) if exact_value != 0 else abs_error

            abs_errors.append(abs_error)
            rel_errors.append(rel_error)

        ax1.plot(n_values, abs_errors, 'o-', linewidth=2, markersize=4,
                 label=method_name, color=color)
        ax2.plot(n_values, rel_errors, 'o-', linewidth=2, markersize=4,
                 label=method_name, color=color)

    ax1.set_xlabel('Количество интервалов n', fontsize=12)
    ax1.set_ylabel('Абсолютная погрешность', fontsize=12)
    ax1.set_title('Сравнение абсолютной погрешности', fontsize=12)
    if y_lim_absolute:
        ax1.set_ylim(*y_lim_absolute)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc=1)

    ax2.set_xlabel('Количество интервалов n', fontsize=12)
    ax2.set_ylabel('Относительная погрешность', fontsize=12)
    ax2.set_title('Сравнение относительной погрешности', fontsize=12)
    if y_lim_rel:
        ax2.set_ylim(*y_lim_rel)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    splits = [pow(2, i) for i in range(2, 6)]
    plot_errors(LeftRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                method_name="Левые прямоугольники")
    plot_errors(RightRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                method_name="Правые прямоугольники")
    plot_errors(MiddleRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                method_name="Центральные прямоугольники")
    plot_errors(TrapezoidMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                method_name="Трапеции")
    plot_errors(SimpsonMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                method_name="Симпсон")
    plot_all_methods_errors(
        [LeftRectangleMethod, RightRectangleMethod, MiddleRectangleMethod, TrapezoidMethod, SimpsonMethod],
        ["Левые прямоугольники", "Правые прямоугольники", "Центральные прямоугольники", "Трапеции",
         "Симпсон"], ANALYTIC_VALUE, 0, 5, f, splits)
