---
date: 2025-12-16
draft: true
images: []
math: true
tags:
  - 虚拟环境
  - python
  - conda
title: condaAndVenv
toc: true
---

## 什么是虚拟环境?

多数语境下，虚拟环境是一个独属于python的术语。python因其对第三方库的高度依赖性以及python自身版本本来就比较混乱，导致经常出现python与各路依赖和第三方库乱成一锅粥的情况。

想要解决这种困境倒也并不困难。我们不妨设想这样的场景：假如你养了一只猫和一条狗，但是猫狗放在一起就会开始打架，你又想同时养这两个动物，该怎么办呢？很简单，你只需要将猫和狗隔开来养即可，比如给它们各自安排一个房间。

这样的一个房间，在python语境下就是一个**虚拟环境**。通常来说，你最好让给每个项目都单独创建一个虚拟环境，这样它们之间就可以互不干扰，可以相互重复，也可以独立删除。就像一个动物园，假如把所有动物关在一起就会开始饥饿游戏，但是把每种动物分开关在各自的园区，彼此之间就可以相安无事。

### 虚拟环境是怎么实现的？

在计算机世界，程序员为了环境隔离实际上有过相当多的尝试，比如虚拟机可以模拟一个操作系统，docker容器可以把进程装在一个看起来像独立系统的盒子里。

但是一个python虚拟环境所做的事情远比上述尝试轻量的多。一个python虚拟环境只做三件事：

1. 使用独立的python可执行文件。隔离不同python版本。
2. 使用独立的第三方目录库，隔离第三方库。
3. 控制导入顺序。

### 如何创建一个虚拟环境？

一个虚拟环境在机器中的表现形式其实就是一个文件夹，里面包含了这个环境的各种信息。通常来说，人们会把这个文件夹建立在项目文件中。

但是，需要注意的是，在一个虚拟环境被激活之后，你无论处在哪个目录下，只要不关闭这个虚拟环境，你都处在这个虚拟环境中。也就是说，虚拟环境与你的目录位置没有任何关系。理论上只要找的到，你把它放在任何一个犄角旮旯里都可以正常工作。

尽管如此，仍然建议你直接把虚拟环境直接建在项目目录中，这样项目和环境可以方便的同步迁移，并且可以确保你的ide可以识别到这个虚拟环境。

#### venv

这是python自带的虚拟环境创建方法。

```sh
python3.10 -m venv .venv
```

这句指令的意义是在当前目录下创建一个.venv文件夹，里面装的就是这个虚拟环境的信息。

针对每个参数，逐一分析一下这句指令：

##### python3.10

指定启动这个虚拟环境的python版本。以防你不知道，一台机器是可以同时存在多个python版本的。比如`pkg install python3.10`就可以强制要求安装3.10版本的python。

##### -m

让后面的名字（在我们这里是venv）作为一个python模块来运行。

##### venv

这是一个python标准库中的模块，功能是创建一个虚拟环境。因为它本质上不是一个位于当前目录下的文件，你没办法直接通过运行文件的方式来使用。这也就是为什么前面必须要加`-m`参数才能使用。

##### .venv

在当前目录下生成的虚拟环境文件夹名称。这只是一个名称而已，你可以随意把它换成你喜欢的名字。

比如：

```sh
python3.12 -m venv .asdf1234
```

之所以加`.`是为了让这个文件夹成为一个隐藏文件夹，这是一个工程惯例。实际上不加`.`也无所谓。

### 如何使用一个虚拟环境？

在你创建的虚拟环境文件夹下，你可以看到大致这样的目录结构：

![](venv%201.png)

你可以很清晰的看到这个文件夹里包含了什么：有虚拟环境自己的python、pip，以及专门存放lib的文件目录。此外，在bin目录下，你还可以找到`activate`文件，这就是虚拟环境的启动脚本。

```sh
source .venv/bin/activate
```

运行这个指令，虚拟环境就成功激活了。想要关闭虚拟环境，输入以下指令：

```sh
deactivate
```

在虚拟环境中，你拥有一套和外界环境隔离的第三方库目录。你可以随意增删当前虚拟环境中的第三方库，这些操作都不会对外界环境造成影响。作为补充，这里罗列一些pip常用的指令。

```sh
python -m pip install numpy # 下载numpy库
```

注意，这里之所以先写`python -m`再写pip，是为了确保使用当前虚拟环境下的pip。pip是一个模块，而模块永远属于某个python。使用`-m`参数可以确保永远不会装错环境。

```sh
python -m pip install numpy==1.26.4 # 指定安装版本
```

```sh
python -m pip list # 列出当前环境下安装了哪些包
```

```sh
python -m pip uninstall numpy # 卸载numpy
```

```sh
python -m pip show numpy # 显示某个包的详细信息，包括版本、安装路径、依赖关系。
```

```sh
python -m pip install --upgrade numpy # 在已经安装了numpy的基础上对它进行更新
```

```sh
python -m pip freeze > requirements.txt # 导出当前环境依赖到requirements.txt文件
```

```sh
python -m pip install -r requirements.txt # 从requirements.txt安装
```

如果你想要删除一个虚拟环境，不需要使用什么特殊的指令，直接rm掉你建立的虚拟环境文件夹即可。

## Conda是什么?

一句话定位conda:

> conda不是更高级的venv，
> 而是：“**一个跨语言+跨平台+可管理python版本的环境与包管理系统**”

python官方的pip以及venv只能管理python包，但是在现实中的科学计算以及机器学习等应用场景中，一个完整的环境会涉及很多非python的依赖，并且需要处理各种跨平台问题。这就已经触及到了python原生工具的能力边界。

conda的出现旨在一次性解决这些问题。conda**同时**是一个**环境管理器**（你可以用它建立虚拟环境，并且比venv隔离的更加彻底）、**包管理器**（conda install numpy），兼**python版本管理器**（不管你系统中有没有安装python）。

### Anaconda/Miniconda是什么？

Anaconda = conda + 一大堆科学计算库

Miniconda = conda + Python

通常认为Anaconda更适合用来快速上手，但是比较冗余。如果想要更轻量可控的环境，更建议使用Miniconda。

### 如何安装conda?

这里以Miniconda为例。

[Miniconda官网](https://www.anaconda.com/docs/getting-started/miniconda/main)

#### 方法一：去官网下载安装脚本

根据你的cpu架构去官网下载对应的.sh脚本，然后在你的终端运行。

>注意：在我写博客文的同时，我发现conda不支持termux。如果你想使用conda，请确保你处在一个真正的电脑上。

#### 方法二：使用curl & wget

wget方法：

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

或者curl方法：

```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

下载得到.sh脚本之后，仍然运行该脚本。注意这里给出的指令是x86_64架构且为linux，aarch64需要做出相应修改。

![](conda.png)

大致流程参考上图。

### 如何使用conda？

此处仍然以miniconda为例。

```sh
conda create -n asdf1234 python=3.10
```

这段指令的意义是：

> 创建一个名为“asdf1234”的虚拟环境，其中python版本为3.10。

其中，`-n`参数含义为"name"，等价于`--name`。

**需要注意的是**，conda的虚拟环境并不像venv那样直接作为当前项目的一个子目录存在，conda的环境被conda视为自己管理的资源，默认集中存放。这意味着在创建这个环境之后，你不会在你的目录下得到任何新的内容。

这样的设计哲学意义在于当不同的环境中请求同样的包时，conda不用将额外的空间资源浪费在重复下载上，而是可以通过内部构建软链接的方式来实现资源复用。

conda允许你通过`--prefix ./env`参数来实现类似venv的存储行为。这里不过多展开。

```sh
conda activate asdf1234 # 激活名为asdf1234的虚拟环境
```

```sh
conda deactivate # 退出当前环境
```

```sh
conda install numpy # 安装包
```

```sh
conda env list # 列出所有创建的环境，以及现在所处的环境
```

```sh
conda remove -n asdf1234 --all # 删除一个环境
```

```sh
conda env export > environment.yml # 导出环境
```

```sh
conda env create -f environment.yml # 从配置文件复现环境
```

注意：使用conda下载包的时候，conda有一个专属名词：**channel**。某些时候，使用conda下载某些包的时候会提示你要下载的包不在当前channel中。这时候你需要自行寻找对应的channel，然后通过`-c`参数指定下载渠道。

```sh
conda install pytorch==1.12.0 -c pytorch
```

> 在一个conda虚拟环境中，你可以混用pip安装方式和conda安装方式。但是建议只在conda中找不到你想要的包，或者你要的包是纯python写的时再用pip。