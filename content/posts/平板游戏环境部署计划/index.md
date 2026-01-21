---
date: 2026-01-20
draft: false
images: []
math: true
tags:
- blog
- 随意探索
title: 平板游戏环境部署计划
toc: true
---

## 今日计划

- 实现平板上配置游戏环境。

由于游戏大多面向windows这样开箱即用的系统开发，Linux并不是游戏运行的一等公民。所以如果想要在一台**arm64架构的·android系统，甚至不是一个纯种linux系统的·纯64位，对32位指令完全不支持的**环境下配置windows游戏运行环境，我们需要摄入适量的前置知识。

# 本篇要素

- wine：一个让windows程序可以跑在linux系统的工具；
- box64：一个将x86_64指令翻译为arm64指令的工具；
- mesa：一套开源图像驱动集合；
- dxvk：图像api翻译器
- turnip & zink：mesa组件，负责驱动vulkan和翻译图像api指令；

# Wine

**W**ine **I**s **N**ot **E**mulator，一个递归式的缩写。wine是一个**兼容层**，旨在让POSIX兼容的操作系统（如Linux）上可以运行原本为Windows开发的应用程序。它的实现方法与虚拟机有巨大差异，并不模拟硬件，而是在运行时实时地将windows api调用翻译为POSIX调用。所以wine的理论性能显著优于虚拟机。

几个核心概念：

### WINEPREFIX（Wine前缀）

类似python的虚拟环境，一个wine前缀代表了一个独立的windows虚拟环境。不过在实现层面，wine的行为可能更加类似于conda，都采用了集中式文件管理架构。

wineprefix存在的意义在于避免前置地狱的痛苦。通过环境隔离，你可以在同一台机器上运行前置相互冲突的程序，只要你把它们放在不同的前缀中。

### WINEARCH（Wine架构）

初始化前缀时指定模拟的是32位windows还是64位windows。

### DLL覆盖

windows程序的运行依赖于大量的动态链接库（即dll）。wine内置了这些dll的开源实现，但是有时出于兼容性考虑需要用windows提取原装版本来替换内建版本。

### Winetricks

一个shell脚本，用来下载和安装各种windows运行时库，辅助wine环境配置。

### WoW64 Mode

wine的新能力，可以将32位windows api调用转换为64位windows api调用。

### 局限

- 内核级反作弊的网络游戏无法运行。
- 配置复杂。
- 不是所有软件都能完美运行。

## Box64

一个二进制指令翻译器，用来在非x86_64架构的Linux系统（尤其是树莓派和Android平板或手机）上运行原本为x86_64处理器编译的Linux程序。与wine类似的，它的实现也不是使用虚拟机。核心功能是在Linux程序运行时将x86_64架构CPU指令实时地转换为arm64能理解的指令。

Box64性能优异的原因有二，一方面它采用了**动态重编译**技术，并不100%的解释运行所有指令，而是运行时动态的将指令重写为arm64机器码。此外，box64采用了**原生库包装**技术，在一些图形处理上直接调用arm64原生库，使得图像处理上几乎可以以原生速度进行。

Box64最夯的应用即和Wine一起打出combo。Box64提供硬件适配，Wine提供系统适配，一同使用就可以在一个树莓派或者高性能arm机器上几乎不损失性能地运行windows程序以及游戏。

## Mesa

Mesa是一个**开源图形驱动集合**，实现了跨平台图形api规范。当我们在Linux上运行游戏时，Mesa会接收游戏发出的绘图指令，并将它们翻译为特定的GPU硬件指令。

由于本人从未涉足过图形学编程的领域，此处额外补充一些必备的图形学知识。所谓的图形api，常见的图像api有OpenGL和Vulkan，它们的作用是将CPU给出的图形绘制指令翻译为GPU指令。其中，OpenGL是一个老资历图形api，特点是相对容易上手，缺点是性能一般；而Vulkan是一个相对更新的图形api，特点是上手难度地狱，但是性能可以比OpenGL强的多。如果类比的话，OpenGL就类似于Python，它帮你做了很多事，代价是性能不够理想；Vulkan则是C语言，所有事情都要你自己控制，所以没有中间商赚差价，性能也自然很高，代价是配置和学习痛苦。

此外，在微软windows系统中还存在Direct3D图形api。同样是一个相较于OpenGL更加现代的图形api。很多windows应用和游戏以Direct3D为图像api。

## DXVK

**D**irect**X** over **V**ul**K**an，即将微软Direct3D图像api调用翻译为Vulkan api调用。

在dxvk出现前，wine默认将direct3d指令翻译为openGL指令，由于Direct3D和OpenGL架构差异巨大，这种翻译效率低下，导致CPU瓶颈严重。DXVK的理念是在Direct3D和与它架构更接近、性能更优的Vulkan api之间建立映射。

在Wine容器中运行windows游戏时，Wine会调用DXVK提供的动态链接库，将指令转换为Vulkan并发送给Mesa，Mesa接受到指令后指挥GPU绘制图像。

在winetricks中可以通过`winetricks dxvk`来一键安装dxvk并覆盖动态链接库。

## Turnip&Zink

Turnip和zink是arm64架构设备上运行桌面级应用的热门mesa组件。

其中turnip是linux上运行vulkan应用的首选驱动。不同于Vulkan官方的驱动，Turnip由社区维护，更新速度快，并且针对DXVK做了大量的优化。主要的应用场景是Android运行PC游戏。

Zink的功能则是将OpenGL翻译为Vulkan。对于一个老旧的OpenGL游戏，只使用Turnip无法运行，这时就需要Zink将OpenGL指令翻译为Vulkan指令。因为OpenGL过于老旧臃肿，即使加上了Zink这一翻译层，Vulkan的运行速度也会比原生的OpenGL更快。

# 全要素再放送

1. **宿主层 (Host)**: Android Kernel + Termux (基础终端环境)。
2. **容器层 (Container)**: PRoot Distro (Ubuntu)。**必须是纯 64 位环境**。
3. **显示层 (Display)**: Termux:X11 (X Server) + Ubuntu 桌面环境 (XFCE4/MATE) 或 窗口管理器。
4. **指令翻译层 (Translator)**: **Box64** (将 x86_64 翻译为 ARM64)。
5. **图形驱动层 (Driver)**: **Mesa (Turnip + Zink)**。这是 Adreno 显卡的生命线。
6. **兼容层 (Compatibility)**: **Wine (WoW64 Build)**，负责在纯 64 位环境下运行 32 位 Windows 程序。
7. **图形转译层 (Graphics Wrapper)**: **DXVK** (DX11 -> Vulkan)。
8. **应用层 (App)**: Windows Steam 客户端 + 游戏。

## 环境配置阶段性任务

1. 完成基本环境配置，纯净64位ubuntu distro环境+termux x11桌面环境基本配置；
2. 构建指令翻译器，安装box64并注册Binfmt；
3. 部署图形驱动，获取Mesa并验证Vulkan支持；
4. 部署wine兼容层；
5. 部署steam；
6. 测试游戏运行；

# 目前进度

基本环境配置完成，可以打开桌面环境并在桌面环境中运行终端。

![](今日随笔-20260120-1.png)

