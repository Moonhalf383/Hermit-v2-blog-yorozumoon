---
title: "工具不图鉴01：curl&wget"
date: 2025-12-15T23:25:53+08:00
draft: false 
toc: true 
images:
tags: ['工具不图鉴','教程', 'curl', 'wget'] 
---
参考链接：

[curl官网](https://curl.se/)

[wget官网](https://www.gnu.org/software/wget/)

curl和wget是在配置服务器时相当常用的终端工具。

---

## 什么是curl？

curl，全称为Client URL，是一个通用的网络协议客户端工具。功能涵盖请求网页，下载文件，发送http请求，调试接口等。可以说这个工具就是一把网络世界的瑞士军刀。不过对于大多数普通使用者，最常见的用途是用它来下载文件。

在我们的日常生活中，我们不可避免的要和网络打交道。最常见的方式就是通过浏览器来浏览网页和下载文件。浏览器通过图形化界面替你完成了写请求的步骤。所谓请求，就是一种规范化的和服务器对话的方式。你只需要点点屏幕就可以完成一系列与网络服务器的信息交换。

但是对于没有图形化界面的终端来说，所有的信息都是基于文本来传递的，我们当然也没办法通过图形化的浏览器来获取网络资源。curl的功能就在于此。只不过不同于浏览器图形化界面的直白，你必须明确你递交的请求包含了哪些动作。

理解curl，其实只需要理解一句话：

> **curl = 在命令行中手动发送http请求**

一条http请求，本质上就是4个东西：

1. 请求方法
2. 请求地址（url）
3. 请求头
4. 请求体

curl的最基本形式：

```sh
curl URL
```

相当于说：让我看看这个url里有什么东西。

关于curl语法的细则这里不再过多赘述。

---

### 如何使用curl？

光知道curl是发送网络协议的工具这点其实对你的使用过程没有太大帮助。事实上，大多数人都不会从http的原理层面来思考和使用curl。

多数人的真实行为是：

- 从网站/博客中复制一个curl指令。
- 稍加修改或者根本不修改。
- 用就完了。

比如，如果你想下载ohmyzsh，你只需要把官网上的下载指令复制下来运行即可。

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

这本身是一个原理相对复杂的指令，但是你根本不用关系这些事情。开发者已经替你考虑好了一切。

而稍微复杂一点点的情况是，你想要下载一个文件，但是README.md中没有给出下载指令。

我们举一个最简单的例子：

---

比如，你想把一个github上的文件下载到本地。但是呢，没有任何网页告诉你应该输入什么指令来下载这个文件。你可能会想到另外用一个智能设备通过浏览器下载，然后上传到终端对应的机器。这是一个方法，但是不够直接。事实上，我们只需要一个curl指令就可以把文件下载到本地。

比如，你看上了ohmyzsh的README.md。你很想把它下载到本地。你不需要动用浏览器，也不需要git clone整个仓库，你只需要在终端中输入：

```sh
curl -O https://github.com/ohmyzsh/ohmyzsh/blob/master/README.md?plain=1
```

curl就开始帮你下载文件了。当然前提是你的网络环境支持你访问到这个网站，如果网络条件不佳，你可能必须要配置网络代理才能正常使用。

但是，只要你实际尝试过，你就会发现，你下载得到的根本不是你想要的东西。你下载得到的是一大堆html代码，根本不是一个md文件。为什么呢？根本上原因在于你下载的实际是用来显示这个网页的代码，而非你想要的内容。

不过，对于github而言，即使你不能直接curl来获取文件，github为你提供了一个另外的快捷途径，方便你直接使用curl来下载文件。你只需要将url中的`github.com`替换为`raw.githubusercontent.com`即可。

然而，在这个网站中并不存有作者名以及plain之类的参数。所以，你还需要删除`/blob`以及`?plain=1`。

```sh
curl -O https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/README.md
```

如果你觉得这个方法太麻烦，对于github来说还有另外一种方法可以使用：

```sh
curl -L -O "https://github.com/ohmyzsh/ohmyzsh/blob/master/README.md?raw=1"
```

因为我选取的这个README.md网页实际比较特殊，后面自带参数。在这个方法中，你只需要在前面添加`-L`参数，并为url后方添加`raw=1`参数即可下载。在本例子中，直接将`plain=1`替换为`raw=1`即可。

---

### 如何下载curl？

在绝大多数情况下，你的机器实际上已经配备了curl工具，因为这是一个非常基础的工具。

假如你的机器上没有这个工具，通常来说你也不需要经过官网来下载最新版。直接使用系统级包管理器下载即可。

```sh
sudo apt install curl # 以ubuntu为例
```

---

## 什么是wget?

> wget = **World Wide Web + get**

字面意思，从万维网上获取东西，就是wget。

不同于curl丰富的功能，这是一个专注于下载内容的工具。它的设计目的很明确：**稳定、自动化、可恢复的下载文件**。

---

### 如何使用wget？

wget的使用方法比curl要简单的多。其中一个最简形式如下：

```sh
wget url
```

不需要任何参数。比如，我们仍然尝试下载ohmyzsh的README.md：

```sh
wget https://github.com/ohmyzsh/ohmyzsh/blob/master/README.md?plain=1
```

当然，这仍然是个错误的指令，通过它你只能下载到网页代码。我写这句指令，意在说明wget的简便易用。真正有用的指令如下：

```sh
wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/README.md
```

你仍然需要得到文件真正存放的url。同时，wget也允许你想下载得到的文件直接用新文件名存入本地，你只需要加一个`-O`参数。

```sh
wget -O myreadme.md https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/README.md
```

---

此外，对于一些体积较大的下载，下载过程可能相对漫长。假如下载过程中因为网络问题而意外终断，一般的wget下载进度就会归零。在此情况下，你可以使用`-c`参数来保存下载进度，这样即使中间下载中断，继续输入指令，文件就会在原有进度基础上继续下载，不会丢失进度。

```sh
wget -c https://example.com/bigfile.iso # 这不是一个真实存在的网站，只是方便演示。
```

---

### 如何下载wget？

与curl情况基本类似，多数时候你的机器上已经自带wget了。在少数极简开发环境下可能出现不自带wget的情况。此时，只需要使用包管理器下载即可。

```sh
sudo apt install wget # 以Ubuntu为例
```
