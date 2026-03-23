from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from matplotlib import pyplot as plt

ANALYTIC_VALUE = -18.11347301218


class Drawer:
    @staticmethod
    def plot_n_vs_accuracy(
            integrator_classes: list[Type],
            method_names: list[str],
            a: float,
            b: float,
            func: callable,
            accuracy_values: list[float],
            title: str = "Зависимость количества интервалов от требуемой точности"
    ):
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for integrator_class, method_name, color in zip(integrator_classes, method_names, colors):
            n_values = []

            for accuracy in accuracy_values:
                integrator = integrator_class([a, b], func, 2)
                _, n_actual = integrator.integrate_with_accuracy(accuracy)
                n_values.append(n_actual)

            plt.loglog(accuracy_values, n_values, '-', linewidth=2,
                       label=method_name, color=color)

        plt.xlabel('Требуемая точность ε', fontsize=12)
        plt.ylabel('Количество интервалов N', fontsize=12)
        plt.title(title, fontsize=12)
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
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

    @staticmethod
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

    @abstractmethod
    def build_and_integrate(self, intervals_cnt):
        pass

    @abstractmethod
    def integrate_with_accuracy(self, accuracy, max_iterations=1000):
        pass


class LeftRectangleMethod(NumericalIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = dict()

    def integrate_with_accuracy(self, accuracy, max_iterations=10000000):
        l, r = 1, max_iterations
        good_iter_cnt = max_iterations // 2
        while l <= r:
            m = l + (r - l) // 2
            if abs(self.build_and_integrate(m) - self.build_and_integrate(2 * m)) <= accuracy:
                good_iter_cnt = m
                r = m - 1
            else:
                l = m + 1
        return self.build_and_integrate(good_iter_cnt), good_iter_cnt

    def integrate(self):
        return sum(self.func(interval[0]) * (interval[1] - interval[0]) for interval in self.intervals)

    def build_and_integrate(self, intervals_cnt):
        l, r = self.borders
        step = (r - l) / intervals_cnt
        x = np.linspace(l, r - step, intervals_cnt)
        return np.sum(self.func(x)) * step


class RightRectangleMethod(NumericalIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = dict()

    def integrate_with_accuracy(self, accuracy, max_iterations=10000000):
        l, r = 1, max_iterations
        good_iter_cnt = max_iterations // 2
        while l <= r:
            m = l + (r - l) // 2
            if abs(self.build_and_integrate(m) - self.build_and_integrate(2 * m)) <= accuracy:
                good_iter_cnt = m
                r = m - 1
            else:
                l = m + 1
        return self.build_and_integrate(good_iter_cnt), good_iter_cnt

    def build_and_integrate(self, intervals_cnt):
        l, r = self.borders
        step = (r - l) / intervals_cnt
        x = np.linspace(l + step, r, intervals_cnt)
        return np.sum(self.func(x)) * step

    def integrate(self):
        return sum(self.func(interval[1]) * (interval[1] - interval[0]) for interval in self.intervals)


class MiddleRectangleMethod(NumericalIntegrator):
    def build_and_integrate(self, intervals_cnt):
        res = 0
        l, r = self.borders
        step = (r - l) / intervals_cnt
        for i in range(intervals_cnt):
            res += self.func(l + (i + 0.5) * step) * step
        return res

    def integrate_with_accuracy(self, accuracy, max_iterations=1000):
        l, r = 1, max_iterations
        good_iter_cnt = max_iterations // 2
        while l <= r:
            m = l + (r - l) // 2
            if abs(self.build_and_integrate(m) - self.build_and_integrate(2 * m)) / 3 <= accuracy:
                good_iter_cnt = m
                r = m - 1
            else:
                l = m + 1
        return self.build_and_integrate(good_iter_cnt), good_iter_cnt

    def integrate(self):
        return sum(self.func((interval[1] + interval[0]) / 2) * (interval[1] - interval[0]) for interval in
                   self.intervals)


class TrapezoidMethod(NumericalIntegrator):
    def integrate(self):
        return sum((self.func(interval[1]) + self.func(interval[0])) / 2 * (interval[1] - interval[0]) for interval in
                   self.intervals)

    def build_and_integrate(self, intervals_cnt):
        res = 0
        l, r = self.borders
        step = (r - l) / intervals_cnt
        for i in range(intervals_cnt):
            res += (self.func(l + (i + 1) * step) + self.func(l + i * step)) / 2 * step
        return res

    def integrate_with_accuracy(self, accuracy, max_iterations=1000):
        l, r = 1, max_iterations
        good_iter_cnt = max_iterations // 2
        while l <= r:
            m = l + (r - l) // 2
            if abs(self.build_and_integrate(m) - self.build_and_integrate(2 * m)) / 3 <= accuracy:
                good_iter_cnt = m
                r = m - 1
            else:
                l = m + 1
        return self.build_and_integrate(good_iter_cnt), good_iter_cnt


class SimpsonMethod(NumericalIntegrator):
    def integrate(self):
        return sum(
            (self.func(interval[0]) + 4 * self.func(interval[1]) + self.func(interval[2])) / 6 * (
                    interval[2] - interval[0]) for interval in
            self.intervals)

    def build_and_integrate(self, intervals_cnt):
        res = 0
        if intervals_cnt % 2 != 0:
            intervals_cnt -= 1
        l, r = self.borders
        step = (r - l) / intervals_cnt
        for i in range(intervals_cnt // 2):
            res += (self.func(l + i * 2 * step) + 4 * self.func(l + (i * 2 + 1) * step) + self.func(
                l + (i * 2 + 2) * step)) / 3 * step
        return res

    def integrate_with_accuracy(self, accuracy, max_iterations=1000):
        l, r = 2, max_iterations
        good_iter_cnt = max_iterations // 2
        while l <= r:
            m = l + (r - l) // 2
            if abs(self.build_and_integrate(m) - self.build_and_integrate(2 * m)) / 15 <= accuracy:
                good_iter_cnt = m
                r = m - 1
            else:
                l = m + 1
        return self.build_and_integrate(good_iter_cnt), good_iter_cnt

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


if __name__ == '__main__':
    # задание 1
    splits = [pow(2, i) for i in range(2, 6)]
    Drawer.plot_errors(LeftRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                       method_name="Левые прямоугольники")
    Drawer.plot_errors(RightRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                       method_name="Правые прямоугольники")
    Drawer.plot_errors(MiddleRectangleMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                       method_name="Центральные прямоугольники")
    Drawer.plot_errors(TrapezoidMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                       method_name="Трапеции")
    Drawer.plot_errors(SimpsonMethod, ANALYTIC_VALUE, 0, 5, f, splits,
                       method_name="Симпсон")
    Drawer.plot_all_methods_errors(
        [LeftRectangleMethod, RightRectangleMethod, MiddleRectangleMethod, TrapezoidMethod, SimpsonMethod],
        ["Левые прямоугольники", "Правые прямоугольники", "Центральные прямоугольники", "Трапеции",
         "Симпсон"], ANALYTIC_VALUE, 0, 5, f, splits)

    # задание 2
    accuracy_values = [1e-5 + i * 25e-6 for i in range(int((1e-2 - 1e-5) / 25e-6) + 1)]
    Drawer.plot_n_vs_accuracy(
        integrator_classes=[LeftRectangleMethod, MiddleRectangleMethod, TrapezoidMethod],
        method_names=["Левые прямоугольники", "Средние прямоугольники", "Трапеции"],
        a=0, b=5,
        func=f,
        accuracy_values=accuracy_values,
        title="Зависимость N от ε (логарифмический масштаб)"
    )

    Drawer.plot_n_vs_accuracy(
        integrator_classes=[SimpsonMethod],
        method_names=["Симпсон"],
        a=0, b=5,
        func=f,
        accuracy_values=accuracy_values,
        title="Метод Симпсона: зависимость N от ε (логарифмический масштаб)"
    )
