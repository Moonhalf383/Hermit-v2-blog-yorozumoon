---
title: "D2L自学日志01：preliminaries"
date: 2025-12-30T18:14:36+08:00
draft: false 
toc: true 
images:
tags: ['D2L','python', 'torch', '机器学习'] 
---
# 机器学习基础知识

- 多维数组
- pandas（数据处理）
- linear-algebra（线性代数）
- calculus（微积分）
- autograd（自动微分）
- probability（概率）

---

## 多维数组基本知识

```python
import torch

x = torch.arange(12) # tensor([0,1,2,3,4,5,6,7,8,9,10,11])
```

`torch.arange(x,y,z)`的语法和python原生的`range()`语法一致，左闭右开，第三个参数为步长。

```python
x.shape
```

一个张量的形状，返回值为一个一个`torch.Size`对象。你可以把它看成一个列表，直接通过方括号访问维度索引值可以得到该张量的某个维度的大小。

如：

```python
import torch
x = torch.arange(24).reshape([2,3,4])
print(x.shape) # torch.Size([2,3,4])
print(x.shape[1]) # 3
```

`torch.reshape([x,y,z,...])`可以修改一个张量的形状，本质上就是先把这个张量拉成一维，再依次按照对应维度的大小对一维张量进行均分。

比如：一个`x = torch.arange(24)`，我们对它进行`.reshape([2,3,4])`操作，你可以看做：一个`arange(24)`先进行shape\[0]处数字均分，变成前十二个和后十二个，然后对两个12长度的子张量再按照shape\[1]处数字进行均分，分别变为三个长度4的子张量，最后对每个长度4的子张量再次均分，得到长度为1的子张量。每一次均分都代表进入下一个维度。

`x.numel()`返回x张量的总元素个数。

`torch.zeros((2,3,4))`生成一个形状为`torch.Size([2,3,4])`的所有元素为0的张量。

`torch.ones((2,3,4))`生成一个形状为`torch.Size([2,3,4])`的所有元素为1的张量。

---

### 运算符

常见的标准运算符（+，-，\*，\\，\*\*）都会被升级为按元素计算。

`torch.exp(x)`得到一个形状和x一致，但是对应位置元素为exp(x)的张量。

`torch.cat((X, Y), dim = 1)`将两个张量沿1维度对接起来，前提是两者其他维度形状一致。

如：

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

结果：

```
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
```

`X == Y`得到一个张量，元素为布尔值，若X和Y对应位置相等，则对应位置的结果为True，否则为False。

`X.sum()`得到一个单元素张量，数值为X所有元素的总和，如果想直接得到这个值，应该写`X.sum().item()`。

---

### 广播机制

当尝试对两个形状并不匹配的张量相互计算时，若不匹配的维度中某个张量的长度为1，torch就会自动在这个维度上通过复制的方式使两个张量强行匹配数据。

如：

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

结果为：

```
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
```

但是：

```
a + b
```

结果为：

```
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

---

### 切片

`X[2, 3:5]`表示的就是X张量中0维度坐标2、1维度坐标3，4的元素。如果我们想要给多个元素赋同样的值，我们只需要索引所有元素，然后赋值。

```python
x = torch.arange(12).reshape([3, 4])
x[0:2, :] = 12
```

得到的x：

```
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```

---

### 转换为其他对象

转换为numpy多维数组：

```python
x = x.numpy()
```

转换为torch.Tensor：

```python
x = torch.Tensor(x)
```

---

## pandas

数据预处理。

写入数据：

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

读取数据：

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

---

### 处理缺失值

将nan值填补为平均值：

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

---

### pandas部分练习

```python
import pandas as pd
import random
import os
import torch

def generate_row():
    row_data = ''
    for i in range(5):
        added_data = random.random()
        if added_data <= 0.2:
            added_data = "NA"
        row_data += str(added_data)
        if i != 4:
            row_data += ','
    row_data += '\n'
    return row_data
                

os.makedirs(os.path.join('..', 'Moonhalf_data'), exist_ok = True)
mh_file = os.path.join('..', 'Moonhalf_data', 'mh_data.csv')
with open(mh_file, 'w') as f:
    f.write('NumA,NumB,NumC,NumD,NumE\n')  # 列名
    for i in range(50):
        f.write(generate_row())

data = pd.read_csv(mh_file)
print(data)

max_na = data.isna().sum().max()
# print(target_column)

refined_data = data.loc[:, data.isna().sum() < max_na]
print(refined_data)

refined_data = refined_data.fillna(refined_data.mean())

refined_tensor = torch.tensor(refined_data.to_numpy(dtype = float))

print(refined_tensor)
```

这段代码的作用是：生成一个50 * 5的随机数数据集，删除缺失值最多的列，并将预处理后的数据集转换为张量格式。

---

## 线性代数

`X.T`转置。

`X.clone()`一个和X一模一样的张量，但是重新分配内存。

`A.sum(axis = 1)`让A沿着1维坐标轴求和，得到n-1维张量。（想象这个张量沿着1维压扁，重叠的地方就求和）

比如：

```python
import torch
test_sum = torch.arange(24).reshape(2,3,4)
print(test_sum)
print(test_sum.sum(axis = 0))
print(test_sum.sum(axis = 1))
print(test_sum.sum(axis = 2))
print(test_sum.sum(axis = [0, 1]))
```

结果：

```
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])
tensor([[12, 15, 18, 21],
        [48, 51, 54, 57]])
tensor([[ 6, 22, 38],
        [54, 70, 86]])
tensor([60, 66, 72, 78])
```

`A.mean(axis = 1)`沿着1维求平均值，规则类似sum。

不降维求和：

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

---

### 点积

`torch.dot(X, Y)`将X和Y对应位置元素相乘求和，得到一个单元素张量。

注意：虽然听起来很废话，但是dot只能适用于1D张量。

---

### 向量积

`torch.mv(X, Y)`对两个向量求叉乘。

---

### 矩阵乘法

`torch.mm(X, Y)`对两个矩阵做矩阵乘法，注意这不是对应位置元素相乘（Hadamard积）。

---

### 范数

用来衡量一个张量的大小。

---

### L2范数

`torch.norm()`得到一个向量的长度。对于一个多维张量来说，它的含义是求所有元素平方和再开根号。

```python
import torch
u = torch.tensor([3.0, -4.0])
v = torch.tensor([3.0,4.0,5.0])
torch.norm(u), torch.norm(v)
```

结果：

```
(tensor(5.), tensor(7.0711))
```

---

### L1范数

`torch.abs(x).sum()`得到所有分量绝对值之和。

---

## 自动求导（autograd）

```python
import torch
x = torch.tensor([2.0], requires_grad=True)

y = x ** 2
y.backward()
print(x.grad) # 此时是 4.0

z = x * 3
z.backward()
print(x.grad) # 此时是 4.0 + 3.0 = 7.0 ！
```

---

### 为什么x.grad似乎并不符合通常的程序设计原则？

`x.grad`在通常的oop程序设计中应该表示一个对象自身的属性，但是在torch中`x.grad`会随着使用x的表达式backward()而发生改变，对于一般的程序设计而言，这种设计是比较反直觉的。

对于不同的表达式，torch对于梯度的操作并非覆盖而是累加。从上面的例子中我们可以看到x.grad最后的值为7.0。至于为什么要设计成累加的模式，一方面是数学上的便利，根据微积分的链式法则，如果一个变量x同时影响了多个分支，那么总梯度就是各个分支梯度的和；另一方面这样做更能节省内存，即使你的网络极其复杂，分成好几路输出，PyTorch也不需要为每一路都开辟新的空间存梯度，只需要在x.grad这一相同地址上做加法即可。

```python
import torch
x = torch.arange(24).reshape([4,6]).to(torch.float32)
x.requires_grad_(True)
y = (x * x).sum()
y.backward()
x.grad
```

需要注意的是，尽管多维数组对多维数组的求导在数学上是有意义的，pytorch中仅支持从标量开始的反向传播。比如，在如上的例子中，y是一个单元素张量，此时可以对它进行backward()操作，但是如果`y = 2 * x`，那么再尝试`y.backward()`就会报错。

原因同样是出于对机器学习实际计算工作的妥协，在工程上直接对多维数组之间进行求导意味着我们会需要计算两个数组元素个数之积次计算，而机器学习中我们处理的数组大小有时可以达到百万数量级。直接求导会导致极大的显存消耗。

---

我们尝试通过一个例子来解释pytorch的优化逻辑：

假设：

- x = \[x1, x2, ...]是一个10000维向量。
- y = \[y1, y2, ...]也是一个10000维向量，但是它是由x计算出来的。
- L = f(y)是一个**标量**。在机器学习的语境下它通常是一个误差函数。

你的目标是求$\frac{\partial L}{\partial x}$。

对于通常的数学书求法，我们会需要使用链式法则，计算$\frac{\partial L}{\partial y}$，再计算$\frac{\partial y}{\partial x}$，再将两者相乘。其中后者就是向量对向量求导，即**雅可比矩阵**。当x、y体积巨大时，直接计算这个雅可比矩阵的计算量会直接爆炸。

更重要的是，虽然x和y体积巨大，但是真正进行相互计算的时候，对于它们的每个分量来说，另一个向量中和它相关的分量个数是相当有限的。尤其在机器学习的情景下，假如你计算出了雅可比矩阵，你会发现它相当的稀疏，其中绝大多数的数字都是0。因为大多数的分量之间都没有什么关系。

但是对于一个矩阵计算来说，它必须要存储所有分量和所有分量之间的关系，这意味着即使是两者没有任何关系，矩阵也必须要占用空间来表示这种没有关系的关系。

pytorch的优化思路其实很简单：我们大可以只站在分量的视角看待问题，只去记录那些和自己有关的分量以及过程中的计算方式，而对于那些和自己无关的分量，我们根本不去记录，也自然不会产生关联。

我们来用一个相对简单的例子解释torch的工作原理：

```python
import torch

x = torch.Tensor([2, 3, 4])

x.requires_grad_(True)

y1_mask = x > 3

y1 = x[y1_mask].squeeze(0)

y2 = (x * x).sum()

# print(y1, y2)

L = y1 + y2

print(f"L: {L}")

L.backward()

print(x.grad)
```

运行得到的结果是：

```sh
L: 33.0
tensor([4., 6., 9.])
```

在这个例子中，y1是x中大于3的元素构成的单元素张量，y2是x对自己的点积组成的单元素张量，L则是y1和y2求和得到的单元素张量。

首先，我们计算y1时，torch同时记录下了“y1是由x通过y1_mask筛选出来的”；计算y2时，torch也同时记录下了“y2是由x对自己按元素求乘积再求和得到的”。最后，计算L时，torch会记录下“L是由y1加上y2得到的”。

当我们对L进行backward时，torch会去查询L的计算方式。此时，我们假设我们需要L增加1，torch会把这个1传递给计算得到L的y1和y2，因为L对y1和y2求偏导得到的是1，所以y1和y2被传递得到1，意味着y1和y2对于L的贡献都是1。

接着，torch会去查询y1和y2的计算方式，y1是由x的第三个元素组成的，所以x的第三个元素对y1的贡献是1，进而对L的贡献也是1；y2由x自乘求和得到，所以x每个元素对y2的贡献是2 * x，在x = \[2, 3, 4]时，x中元素对y2的贡献依次为4、6、8，进而对L的贡献也是4，6，8。

最后，我们对x中每个元素对L的各种贡献进行求和，我们得到x中每个元素在当前数值下对于L的贡献依次为4，6，9，即为我们最后得到的`tensor([4.,6.,9.,])`

需要明确的是，贡献这个词的使用实际上可能数学并不准确。我们最后得到的东西实际上是L这个标量对x的梯度，即∇L，其意义可以简单理解为L在随x各个维度改变而改变的速率。

---

### 非标量变量的反向传播

从我们上面的推导中可以发现，我们对一个标量结果进行反向传播，最后得到的向量是x中对应元素对结果的贡献值。而理论上假如我们能直接对一个向量进行反向传播，最后得到的应该是一个雅可比矩阵。

然而，我们需要的事实上根本不是这个矩阵，我们需要的是损失向量通过这个矩阵变换得到的结果，进而更新参数。对于我们的实际应用中，这个矩阵本身没有任何应用价值。

用更加精辟的语言来说：

> Pytorch的反向传播根本不需要返回一个矩阵，因为它本身就在模拟雅可比矩阵的线性算子功能。

然而，在某些情景下我们确实会需要计算非标量变量的反向传播，比如多个batch同时进行训练，但是即使在这种情景，我们最后需要得到的依然是一个向量。实现方式是让批量中每个样本对x求梯度，最后再求和或者求平均，或者加权平均。总而言之，反向传播归根结底是获取一个用来更新参数的向量，获取二维及以上的张量没有什么实际意义。

```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
print(x.grad, x)
y = x * x
print(y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

所谓的gradient参数，实际上是一个和x一样长度的向量，含义是**y的每个分量**对于最终结果的贡献。在这里，如果我们写`y.sum()`，本质上就是将y的每个元素进行求和，所以y的每个元素对最终结果的贡献为1；如果写`torch.ones(len(x))`意思就是y的每个参数对于最终结果的贡献都是1，所以两者其实是等价的。

---

### 分离计算

有时候我们只想要把一个`torch.Tensor`当做一个常数来计算，但是得到这个`torch.Tensor`的过程中经过了一些计算，使得最后反向传播的时候可能影响最后的结果。此时，我们会希望得到这个tensor，但是抹除它所有的计算历史。

```python
import torch
x = torch.arange(4)
x.grad.zero_()
y = x * x
u = y.detach()
print(u)
z = u * x

z.sum().backward()
x.grad == u
```

在这个例子中，u被赋值为`y.detach()`，含义是使u的张量内容和y一样，但是失去了所有计算记录。这样，后续的反向传递过程中u处就不会向后传递。

结果是：

```sh
tensor([True, True, True, True])
```

---

### 如何实现二阶导?

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# 1. 定义函数
y = x ** 3

# 2. 计算一阶导数
# 使用 grad 函数，并开启 create_graph=True
# 这会把“计算一阶导数”的过程也记录在计算图里
grads = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"一阶导数在 x=2 时: {grads.item()}") # 3 * 2^2 = 12.0

# 3. 对一阶导数再次求导
# 这次不需要再创建图了（除非你要算三阶导）
grads2 = torch.autograd.grad(grads, x)[0]
print(f"二阶导数在 x=2 时: {grads2.item()}") # 6 * 2 = 12.0
```

我们没办法通过backward实现二阶导（比如L.backward().backward()），因为反向传播的计算本身并不会被存储在计算图中，但是为了实现二阶导，我们需要一阶导的计算图。因此在上面的例子中我们使用`torch.autograd.grad()`函数，并添加`create_graph=True`参数来生成包含一阶导计算图的梯度向量。

---

### 控制流

```python
import torch

a = torch.randn(size=(12,), requires_grad=True)
print(a)

if a.norm() > 3:
    b = a.sum()
else:
    b = (a * a).sum()

b.backward()

print(a.grad)
```

torch支持控制流梯度计算，这意味着即使不同情况下计算b的方式可能不同，对b进行反向传播仍然可以得到当前计算方式下的对应梯度。

---

### 绘制函数与导函数图像（不直接计算导数）

```python
import matplotlib
from matplotlib_inline import backend_inline
import numpy as np
from d2l import torch as d2l
import math
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

x = d2l.arange(0.0, 2 * math.pi, 0.05*math.pi)
x.requires_grad_(True)
def f(x):
    return d2l.sin(x)

y = f(x).sum()

y.backward()

plot(x.detach(), [f(x).detach(), x.grad],"x", "f(x)")
```

需要使用d2l库和jupyter lab。

---

## 概率

### 多项分布(multinomial distribution)

将概率分配给一些离散选项的分布称为多项分布。

```python
import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6

print(multinomial.Multinomial(1, fair_probs).sample())
print(multinomial.Multinomial(2, fair_probs).sample())
print(multinomial.Multinomial(10, fair_probs).sample())
```

`multinomial.Multinomial()`用来生成一个采样器，这个采样器每次采样会根据`fair_probs`也就是概率向量对应的概率来抽取一个位置索引，第一个参数时一次采样的抽取次数。最后得到的向量中对应位置的数字即这个位置被抽中的次数。

上述代码的某次生成结果为：

```python
tensor([1., 0., 0., 0., 0., 0.])
tensor([0., 2., 0., 0., 0., 0.])
tensor([1., 1., 0., 2., 1., 5.])
```

---

### 联合概率

$P(A=a,B=b)$，A和B同时发生的概率。

### 条件概率

$0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$ ，A发生的前提下A、B同时发生的概率。

### 贝叶斯定理

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

### 边际化

$$P(B) = \sum_{A} P(A, B),$$
