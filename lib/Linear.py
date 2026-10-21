import numpy as np
from linear_low import linear

__all__ = ['Linear']


def transpose_last_two_axes(x: np.ndarray) -> np.ndarray:
    axis1 = x.ndim - 1
    axis2 = x.ndim - 2
    axes = list(range(x.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return np.transpose(x, axes)

from base import Module
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        rng = np.random.default_rng()
        self.weights = rng.random((out_features, in_features), dtype=np.float32)
        if bias:
            self.bias = rng.random(out_features, dtype=np.float32)
        else:
            self.bias = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Поддержка N-мерных тензоров: (..., in_features) @ (in_features, out_features)
        w_t = transpose_last_two_axes(self.weights)
        out = x @ w_t
        if self.bias is not None:
            out = out + self.bias
        return out











'''
class Linear:
    # Python-level
    # ВНИМАНИЕ: матрица весов weights хранится НЕ в транспонированном виде
    # - непригодна для прямого сравнения с torch

    def __init__(self, in_features, out_features):
        # bias=True

        rng = np.random.default_rng()  # создаём генератор

        self.weights = rng.random((out_features, in_features), dtype=np.float32) # веса из равномерного распределения\
        self.bias = rng.random(out_features, dtype=np.float32) # смещения

        # Инициализация низкоуровневой библиотеки?
        # здесь на self-ах заранее сохранять всю нужную C информацию?


    def __call__(self, x):
        y = linear(x, self.weights, self.bias)
        return y

'''



# проверка

'''
lin = Linear(2, 3)

lin.weights[0][0] = 1
lin.weights[0][1] = 4
lin.weights[1][0] = 2
lin.weights[1][1] = 5
lin.weights[2][0] = 3
lin.weights[2][1] = 6

lin.bias[0] = 10
lin.bias[1] = 20
lin.bias[2] = 30

X = np.array([[2, 3]], dtype=np.float32)

print('X:')
print(X)

print('\nlin.weights:')
print(lin.weights)

print('\nlin.bias:')
print(lin.bias)

print('\nResult:')
print(lin(X))
'''





'''
import numpy as np
import torch
import torch.nn as nn
from Linear import Linear

# Устанавливаем seed для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# Параметры
batch_size = 2
in_features = 3
out_features = 2

print("=" * 60)
print("СРАВНЕНИЕ: Мой Linear vs torch.nn.Linear")
print("=" * 60)

# Создаем входные данные
x_np = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]], dtype=np.float32)

x_torch = torch.tensor(x_np, dtype=torch.float32)

print("\nВходные данные (batch=2, in_features=3):")
print(x_np)

# Создаем фиксированные веса для наглядности
weights_np = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]], dtype=np.float32)

bias_np = np.array([0.01, 0.02], dtype=np.float32)

print("\nВеса (out_features=2, in_features=3):")
print(weights_np)

print("\nBias (out_features=2):")
print(bias_np)

# Создаем и настраиваем мой модуль
my_linear = Linear(in_features, out_features)
my_linear.weights = weights_np.copy()
my_linear.bias = bias_np.copy()

# Создаем и настраиваем torch модуль
torch_linear = nn.Linear(in_features, out_features)
torch_linear.weight.data = torch.tensor(weights_np, dtype=torch.float32)
torch_linear.bias.data = torch.tensor(bias_np, dtype=torch.float32)

print("\n" + "=" * 60)
print("ПРОВЕРКА ВЕСОВ")
print("=" * 60)

print("\nМой Linear.weights shape:", my_linear.weights.shape)
print(my_linear.weights)

print("\ntorch.nn.Linear.weight shape:", torch_linear.weight.shape)
print(torch_linear.weight.data.numpy())

print("\nВеса идентичны:", np.allclose(my_linear.weights, torch_linear.weight.data.numpy()))

print("\n" + "=" * 60)
print("FORWARD PASS")
print("=" * 60)

# Мой модуль
y_my = my_linear(x_np)
print("\nРезультат моего Linear:")
print(y_my)

# torch модуль
y_torch = torch_linear(x_torch)
print("\nРезультат torch.nn.Linear:")
print(y_torch.data.numpy())

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

print("\nРезультаты совпадают:", np.allclose(y_my, y_torch.data.numpy()))
print("\nРазница (должна быть ~0):")
print(np.abs(y_my - y_torch.data.numpy()))

print("\n" + "=" * 60)
print("РУЧНАЯ ПРОВЕРКА ВЫЧИСЛЕНИЙ")
print("=" * 60)

# Ручное вычисление для первого элемента батча
print("\nДля x[0] = [1, 2, 3]:")
print(f"  y[0][0] = 1*0.1 + 2*0.2 + 3*0.3 + 0.01 = {1*0.1 + 2*0.2 + 3*0.3 + 0.01}")
print(f"  y[0][1] = 1*0.4 + 2*0.5 + 3*0.6 + 0.02 = {1*0.4 + 2*0.5 + 3*0.6 + 0.02}")

print("\nДля x[1] = [4, 5, 6]:")
print(f"  y[1][0] = 4*0.1 + 5*0.2 + 6*0.3 + 0.01 = {4*0.1 + 5*0.2 + 6*0.3 + 0.01}")
print(f"  y[1][1] = 4*0.4 + 5*0.5 + 6*0.6 + 0.02 = {4*0.4 + 5*0.5 + 6*0.6 + 0.02}")

print("\n" + "=" * 60)
print("ВЫВОД")
print("=" * 60)
print("\nМой Linear полностью совместим с torch.nn.Linear")
print("Веса хранятся в одинаковом формате (out_features, in_features)")
print("Forward pass дает идентичные результаты")
'''