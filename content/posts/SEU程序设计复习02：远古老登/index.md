---
date: 2026-01-19
draft: false
images: []
math: true
tags:
- cpp
- blog
- 课内
title: SEU程序设计复习02：远古老登
toc: true
---

# 本篇要素：

- 指针
- C风格字符串
- C风格数组
- 基本算法

# 指针

![](SEU程序设计复习02远古老登-20260119-1.png)

众所周知，无论是什么程序语言，程序运行时的数据都缓存在内存上。内存就像一张超级大的草稿纸。对于机器来说，除了对内存写入的数据本身外，数据写入的位置同样重要。知道了数据写入的位置，我们才能随时利用我们写入的数据进行计算。

于是指针诞生了。如果说一般的变量是用来存储有价值的数据本身，指针的作用则是存储这些变量的内存地址，方便我们随时随地的调用这些变量。指针的本质仍然是变量，只不过它存储的内容是一个长十六进制的内存地址。

我们来举一个简单的例子：

```cpp
#include <iostream>
int main(int argc, char *argv[]) {
  int x = 114514;
  int *y = &x;
  std::cout << y << std::endl;
  std::cout << *y << std::endl;
  return 0;
}
```

输出结果：

```sh
0x7fc7cf03ac
114514
```

其中第一个输出结果就是这个指针指向的变量的地址，而第二个输出结果则是这个变量的值。事实上，`*y`可以看做是x的一个别名，两者的行为完全一致，因为他们都是同一个地址上相同数据类型的变量。你可以给`*y`赋值，`x`的值就会被同步改变，因为在内存上它们的意义是相同的。

由于内存可以表示为一个长十六进制整数，所以自然我们对一个已知的内存进行加减就可以访问到它周围的数据。访问一个内存附近的元素有两种办法：直接对指针做加法，或者给指针追加下标：

```cpp
#include <iostream>
int main(int argc, char *argv[]) {
  int x = 114514;
  int *y = &x;
  *(y + 1) = 191981;
  std::cout << y << std::endl;
  std::cout << *y << std::endl;
  std::cout << y[1] << std::endl;
  return 0;
}
```

其中，`*(y+1)`访问y的内存的后一位并将其赋值为`191981`。在整个程序中，我们没有任何一个变量直接代表这个内存地址，我们是通过解引用的方式间接地访问到了这个地址。之后在输出阶段，我们调用`y[1]`同样表示的是y的内存的下一位的内容，所以理论上输出结果也应该是`191981`。

输出结果如下，符合预期：

```sh
0x7fd11438cc
114514
191981
```

这时候你会发现下标运算符似乎和数组中指定访问某个元素使用的操作符都是中括号。事实上，在C语言以及C++中，数组和指针在行为上确实非常类似，但是从根本上看两者是完全不同的东西。后面会提到。

接下来是比较八股的部分，在实际的C++编程中没有人类会这么写，但是出题人会。

- `int *a, b`，这个语句中a是指针，b则是一个普通的变量。没有为什么，硬要理解的话`*a`和`b`都是int类型的变量。
- `int *p[4]`表示的是一个数组，每个元素为一个`*int`指针。
- `int (*p)[4]`表示的是一个数组，但是每个元素都是int数据。

直接看这个是没用的，真正要尝试理解这种设计，我们必须要站在老登语言C语言的角度看待问题。

## 如何站在C语言老登的视角看待指针

在C语言中几乎没有高级的数据结构，最高级的数据结构就是结构体和原始数组了。这时候处理问题就不得不直面手动内存管理的问题。

C语言一个最恶心的点在于，C语言有意识的在鼓励指针和数组之间的混淆。比如，很多时候指针和数组名的表现完全一致，甚至在将数组传参进入一个函数的时候完全就是指针。但是问题在于，数组和指针的底层实现完全不一致。

比如在下面这个例子中：

```c
#include <stdio.h>

int main(int argc, char *argv[]) {
  int a[4] = {1, 2, 3, 4};
  int *p = a;
  printf("%d\n", p[1]);
  return 0;
}
```

a是一个数组，p是一个指针，`int *p = a`之后p和a看起来指向的都是一个值，但是在底层上：

- p是一个变量，它自身有一个地址，而地址上存储的是数组第一个元素的地址。
- a是一个**别名**。a没有地址，a就是这个数组第一个元素地址的别名。

而对于一个常量指针，比如某个`int * const q = a`，它又有另一重底层意义：

- q是一个常量，但是它有自己的地址，它的地址上存储的数据是数组的第一个元素的地址。

这就意味着数组和指针是完全不同的数据类型。a虽然可以隐式退化为指针，原因完全是因为他们的意义基本一致，但是这不代表它们是同一个东西。

同样，二维数组和二级指针根本上也不是一个东西。当我们写`int a[3][4]`时，a的类型是一个二维数组，当我们通过`a[1][1]`来访问元素的时候，C语言的底层实现方法不是通过指针进行间接访问，而是直接由下标计算偏移量。但是对于一个二级指针，我们可以让这个二级指针指向一个一级指针数组，再让一级指针指向一维数组，但是当我们想要访问元素时，底层的汇编指令是：先访问二级指针的地址，读取二级指针的存储的地址，再由这个地址加上偏移量访问一级指针，再由一级存储的地址加上偏移量访问元素。

所以，虽然你可以造出一个和二维数组行为完全一致的二级指针，但是归根结底它们是完全不同的两个东西。回到我们原本的问题，我们来重新的明确阐述一下我们原本想讨论的问题：

已知：
- `int a[3]`是一个数组
- `int* p`是一个指针
问：
- `int* p[3]`是什么？
- `int (*p)[3]`是什么？

其实这个问题的根本就在于：凭什么`int (*p)[3]`的写法是**合法的**？

比如，我们绝对不会写`int (x+1) = 2`然后希望x = 1。如果你在代码中这么写，那么你会喜提报错。但是`int (*p)[3]`却是合法的，它成功定义了一个变量，并且它的名字就是p。真正的诡异之处就在这里。

假如我们把`int*`看做一个整体，表示后面的东西是一个指针，那么`int* p[3]`是容易理解的，它代表的就是一个存放了一级指针的指针数组。但是在`int (*p)[3]`里，我们会觉得中间的`*p`放在这里是不讲道理的，因为我们理应认为这个地方不应该进行任何的计算操作，而\*p表示的是对p解引用。

问题的答案是，C语言不是现代语言，而是远古老登。现代语言中我们会希望类型和标识符做到完全区分，而C语言的设计者遵循的原则是：声明的样子和使用时的样子应该完全一致。虽然我觉得这样也是不太讲道理的，但是对于理解这个问题意外的好用。

比如，在`int (*p)[3]`定义后，本质上说明了一件事情：假如我想使用p寻找某个元素，我就应该写`(*p)[i]`来做到这件事，即：对p进行一个`(*p)[i]`的操作后，得到的应该是一个整数。

那么，`(*p)`进行一个下标索引的操作后，得到的是一个整数，这就意味着`(*p)`本质上是一个数组，`p`的本质就是一个指针。对p解引用得到数组，就意味着p指针的基本类型就是数组。而p++就是下一个位置的数组，因此p和一个二维数组的行为是一致的，它们都可以表示连续内存中的一系列一维数组。此处的p叫做**数组指针**。

而同样的，假如我们要去理解`int *p[3]`，我们就会知道如果我们要使用p来找到某个元素，我们就会写`*p[3]`，而我们知道下标索引的优先级高于解引用，这就意味着`p[3]`得到的是一个指针，所以`p`的本质是一个数组，内部的元素是指针。此处的p叫做**指针数组**。

### 常量指针&指针常量

常量指针：

```cpp
int a = 10; 
int b = 20; 
const int* p = &a;
```

之后你可以使用`p = &b`来修改p的指向，但是不能使用`*p = 30`来修改对应的值。

指针常量：

```cpp
int a = 10; 
int b = 20; 
int* const p = &a;
```

可以通过`*p = 30`来修改值，但是不能改变指向的地址。

## C风格数组

当我们将C风格数组作为参数传入函数时，数组名会**退化为指针**。这里就是完全意义上变为指针。

比如，当我们写：`void func(int arr[10]){...}`的时候，我们就等同于写`void func(int* arr)`。虽然我们前文中提到数组和指针是完全不同的两种东西，但是这里，数组退化为指针的事情就是真实地发生了。

这样的退化带来的第一个问题就是，我们没办法使用sizeof来获取一个数组的长度。最传统的做法就是添加一个参数来手动传入数组大小。

C风格数组传参时，以下写法是完全等价的。

```cpp
void f(int arr[]);
void f(int arr[10]);
void f(int* arr);
```

等价的原因就在于无论原本是什么东西，在这里全部都会退化为指针。

但是假如我们需要传入一个多维数组，比如某个`int a[3][4]`，此时我们就不能写`f(a)`，原因在我们前面分析指针原则的时候其实涉及到了，程序员有各种各样的方式实现多维数组，所以多维数组没办法直接退化为某个指针。

以二维数组为例，原教旨主义的二维数组就是数组，内存连续。和它的行为最接近的是**数组指针**，所以传参时可以这么写：

```cpp
void foo(int (*a)[4]);
```

此外，如果确定了列数，你也可以这么写：

```cpp
void foo(int a[][4]);
```

但是**不能**这么写：

```cpp
void foo(int a[][]);
```

## C风格字符串

```cpp
char s1[] = "hello";
```

等价于：

```cpp
char s1[] = {'h','e','l','l','o','\0'};
```

注意，`\0`标志一个字符串结束。

例子：

```cpp
#include <iostream>

int main(int argc, char *argv[]) {
  char str[] = "Helloworld!";
  std::cout << str << std::endl;
  char str1[] = {'H', 'e', 'l', 'l', 'o', '\0'};
  std::cout << str1 << std::endl;
  char str2[] = {'H', 'e', '\0', 'l', 'l', 'o'};
  std::cout << str2 << std::endl;
  return 0;
}
```

输出结果：

```cpp
Helloworld!
Hello
He
```


# 基本算法

在工程实践中，手写排序是被严格禁止的，因为你几乎不可能写的比标准库更好。但是对于考试只能说是不得不品的一环。

## 冒泡排序

最佳实现：

```cpp
#include <vector>
#include <algorithm>

template<typename T>
void bubbleSort(std::vector<T>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; ++i) {
        swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        // 如果没有发生交换，直接跳出循环
        if (!swapped) break;
    }
}
```

基本原理就是进行多次迭代，每次都遍历一遍，如果某个位置前一个数字大于后一个，就将其交换；如果某次迭代的遍历过程中没有进行任何交换，就证明已经完成了从小到大的排序，可以结束计算了。

我们也可以使用原始数组来重写一遍：

```cpp
#include <iostream>

void bubbleSort(int arr[], int len) {
  bool swapped = true;
  for (int i = 0; i < len - 1 && swapped; i++) {
    swapped = false;
    for (int j = 0; j < len - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        int temp = arr[j + 1];
        arr[j + 1] = arr[j];
        arr[j] = temp;
        swapped = true;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int arr[10] = {4, 1, 2, 6, 8, 4, 2, 5, 7, 3};
  bubbleSort(arr, 10);
  for (int i = 0; i < 10; i++) {
    std::cout << arr[i] << std::endl;
  }
  return 0;
}
```
## 选择排序

```cpp
template<typename T>
void selectionSort(std::vector<T>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        int minIndex = i;
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        if (minIndex != i) {
            std::swap(arr[i], arr[minIndex]);
        }
    }
}
```

原理是每次遍历后把遍历涉及的元素中最小的元素移到前面，之后的遍历不再关注前面的元素。时间复杂度和冒泡排序一致。（但是冒泡排序更方便设定排序规则，所以我更喜欢冒泡排序。）

同样使用原始数组实现一遍：

```cpp
#include <iostream>

void selectionSort(int arr[], int len) {
  for (int i = 0; i < len; i++) {
    int minIndex = i;
    for (int j = i; j < len; j++) {
      if (arr[j] < arr[minIndex]) {
        minIndex = j;
      }
    }
    int temp = arr[i];
    arr[i] = arr[minIndex];
    arr[minIndex] = temp;
  }
}

int main(int argc, char *argv[]) {
  int arr[10] = {4, 1, 5, 2, 6, 3, 7, 9, 8, 0};
  selectionSort(arr, 10);
  for (int i = 0; i < 10; i++) {
    std::cout << arr[i] << std::endl;
  }
  return 0;
}
```

## 快速排序

```cpp
template<typename T>
int partition(std::vector<T>& arr, int low, int high) {
    // 优化：三数取中选基准
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) std::swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) std::swap(arr[high], arr[mid]);
    
    T pivot = arr[mid];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        if (i >= j) return j;
        std::swap(arr[i], arr[j]);
    }
}

template<typename T>
void quickSort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int p = partition(arr, low, high);
        quickSort(arr, low, p);
        quickSort(arr, p + 1, high);
    }
}

// 辅助调用接口
template<typename T>
void quickSort(std::vector<T>& arr) {
    if (!arr.empty()) {
        quickSort(arr, 0, arr.size() - 1);
    }
}
```

顾名思义，快速排序算法确实非常快。冒泡排序和选择排序的速度都是$O(n^2)$，而快速排序的时间复杂度则是$O(n\log n)$。我们来尝试理解下它的工作原理。

快速排序的基本思想是：每次从一个分区中选取一个**基准值**，比它大的数字全部移动到右边，比它小的全部移动到左边，以交换过程最后的临界点作为分区界线形成了两个分区，之后再在这两个分区中重复这个过程。这样，每次进行分区就意味着一个数字找到了正确的位置，而分区量呈指数级增长，每次分区计算的用时和总数成正比，故时间复杂度为$O(n\log n)$。

在上面这段代码中，选取基准值的方法是在一个分区的最左、最右和最中间的三个元素中选出中间的那个数字，然后动用两个指针向基准值收缩，实现分区，之后再通过递归完成整体的排序。

原始数组实现：

```cpp
#include <iostream>
void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

int partition(int arr[], int low, int high) {
  int mid = low + (high - low) / 2;
  if (arr[mid] < arr[low])
    swap(arr + mid, arr + low);
  if (arr[high] < arr[low])
    swap(arr + high, arr + low);
  if (arr[high] < arr[mid])
    swap(arr + high, arr + mid);

  int pivot = arr[mid];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    do {
      i++;
    } while (arr[i] < arr[mid]);
    do {
      j--;
    } while (arr[j] > arr[mid]);
    if (i >= j)
      return j;
    swap(arr + i, arr + j);
  }
}

void quickSort(int arr[], int low, int high) {
  if (low < high) {
    int p = partition(arr, low, high);
    quickSort(arr, low, p);
    quickSort(arr, low + 1, high);
  }
}

int main(int argc, char *argv[]) {
  int arr[10] = {4, 1, 5, 2, 6, 3, 7, 8, 9, 0};
  quickSort(arr, 0, 9);
  for (int i = 0; i < 10; i++) {
    std::cout << arr[i] << std::endl;
  }
  return 0;
}
```


## 线性查找

从头到尾依次查找最符合条件的元素。就像苏格拉底最大的麦穗的寓言故事。

懒得写了。

## 二分查找

假如你要查找的元素总体是按照从小到大或者从大到小的顺序排列的，那么如果你要找到某个元素的位置就不需要把所有元素都搜一遍。我们只需要每次找到最中间的点，之后如果目标比中值大，则向右寻找，否则想左寻找。这样时间复杂度就大大减少。

以下提供一种基于递归的解法：

```cpp
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
int arr[10];

//这一部分是搜索算法。
int binarySearch(const int arr[], int low, int high, int target) {
  if (low > high)
    return -1;
  int mid = low + (high - low) / 2;
  if (target == arr[mid])
    return mid;
  if (target < arr[mid])
    return binarySearch(arr, low, mid - 1, target);
  return binarySearch(arr, mid + 1, high, target);
}

void renewArray() {
  for (int i = 0; i < sizeof(arr) / sizeof(int); i++) {
    arr[i] = rand() % 100 + 1;
  }
  std::sort(arr, arr + 9); //注意，这里需要进行排序，原因是二分查找只能适用于已经排序的数组。
}

int main(int argc, char *argv[]) {
  std::srand(time(nullptr));
  std::cout << std::left;
  for (int i = 0; i < 20; i++) {
    renewArray();
    for (int j = 0; j < 10; j++) {
      std::cout << std::setw(8) << arr[j];
    }
    std::cout << std::endl
              << "Index: " << binartSearch(arr, 0, 9, 50) << std::endl;
  }
  return 0;
}
```

然而，虽然说着容易，实际实践的时候经常会出现内存越界和死循环的问题。原因是分区的边界情况非常的不好考虑。以下罗列两种公认的最优写法。

第一种被称为“经典精确查找”，做了很多防范问题的措施。

```cpp
#include <vector>

template<typename T>
int binarySearch(const std::vector<T>& nums, T target) {
    int left = 0;
    int right = nums.size() - 1; // [left, right] 闭区间

    while (left <= right) {
        // 防止溢出的写法，等同于 (left + right) / 2
        int mid = left + (high - low) / 2; 

        if (nums[mid] == target) {
            return mid; // 找到了
        } else if (nums[mid] < target) {
            left = mid + 1; // 目标在右半部分
        } else {
            right = mid - 1; // 目标在左半部分
        }
    }
    return -1; // 未找到
}
```

另一种是所谓的“寻找边界写法”，在有目标值的时候返回符合的值中第一个目标的索引值，如果没有则返回第一个大于目标值的位置。

```cpp
template<typename T>
int lowerBound(const std::vector<T>& nums, T target) {
    int left = 0;
    int right = nums.size(); // [left, right) 左闭右开区间

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else {
            // 当 nums[mid] >= target 时，收缩右边界
            // 这样会不断向左逼近第一个等于 target 的位置
            right = mid;
        }
    }
    return left; // 此时 left == right
}
```

## 最小公倍数

```cpp
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
```

原理是$gcd(a,b) = gcd(b,a\%b)$。