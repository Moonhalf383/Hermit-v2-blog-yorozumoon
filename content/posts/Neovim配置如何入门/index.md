---
title: "Neovim配置如何入门"
date: 2025-11-23T10:53:11+08:00
draft: false
images:
tags: ['Neovim','教程']
toc: true
--- 

如题。neovim因为自定义程度过高导致学习曲线极其陡峭，虽然功能很强大但是配置过程极其折磨，尤其是当你对于neovim配置的原理毫无概念的时候。

闲言少叙，我尝试来总结一下如何快速入门并配置一个能用的neovim。（后记：在学习完如何配置nvim之后，你实际应该做的是去用lazyvim。lazyvim的lazyextra功能直接优化掉了nvim最反人类的lsp配置过程，如果你有生活，不要尝试自己配置lsp。但是如果你想在使用lazyvim时可以轻松应对各种插件问题，你最好学一下nvim插件配置的通用原理，即我下文中会讲到的。）

---

## Nvim配置原理

neovim是一个文本编辑器，完全可以运行在终端中，无需给它一个桌面环境。你可以按照自己的需求给它添加对特定编程语言的支持，添加语法高亮、自动补全等功能，或者切换颜色主题。它非常轻量，运行飞速，极其贴近代码的原生环境，并且可以运行在任何一台看起来像电脑的机器上。

但是代价是，这一切都必须通过你自己手动配置来完成。Neovim并不是一个开箱即用的软件，即使你使用一些诸如Nvchad或者Lazyvim这样的预配置版本，很多功能还是必须要你自己手动添加。

通常，一个neovim的配置文件目录如下：

```sh
~/.config/nvim/
├── init.lua
└── lua/
    ├── config/
    │   ├── options.lua
    │   ├── keymaps.lua
    │   └── autocmds.lua
    └── plugins/
        ├── init.lua       （可选）
        ├── lualine.lua
        ├── treesitter.lua
        └── ……

```

乍一看，你可能会觉得很复杂。其实不然，因为lua目录下的文件实际都是自己想怎么加就怎么加的，真正限定的结构只有根目录下的init.lua和lua这个文件夹。只不过通常而言我们会在lua文件夹下创建这些文件结构，方便我们管理自己加入的插件以及相应配置。

正如你所见，nvim的配置文件全部都使用的是lua语言。但是不用有太高的心理负担，因为这是一个语法甚至比python还要简单的语言。nvim配置过程就相当于构建一个自己的lua语言程序，你需要自己组织其文件结构来让neovim知道如何使用你的配置。

所以显然，在你开始配置nvim之前，你最好能了解一下lua语言的用法。不过也不用着急，你可以接着看，遇到不懂的点再去深入了解。

---

### kickstart

[kickstart链接](https://github.com/nvim-lua/kickstart.nvim)
这是并不是一个neovim发行版，它只是一套配置文件。严谨的说，它是nvim官方维护的一个学习版nvim配置框架。它使用了lazy. nvim这一插件管理器（注意：这和lazyvim完全不是一回事），但是和lazyvim是完全不一样的两个东西，后者是一个“开箱即用”的nvim框架（但也不那么开箱即用）。

你仍然需要安装nvim。kickstart只是提供了清晰的配置文件，方便初学者快速了解如何配置nvim。我们接下来也会以kickvim提供的init.lua为例讲解neovim可供配置的内容。

#### 安装neovim

显然，如果我们想要使用neovim，我们需要先安装它。你可以安装任意你想要的版本，不过这里我们选择Nvim 0.11.5。kickstart要求nvim版本为最新的稳定版，或者每日夜晚自动编译得到的不经过稳定性测试的Nightly版，我们此处使用的是最新的稳定版。

对于大部分普通电脑，cpu采用x86_64架构，你需要下载nvim-linux-x86_64.tar.gz。这是软件文件的压缩包，你需要把它移动到你想要它存在的文件目录下，然后在该目录下执行：

```sh
tar xzvf nvim-linux-x86_64.tar.gz #或者你实际下载的压缩包
```

我这边使用的是termux，平板、手机一般cpu架构是arm64，所以我下载的是nvim-linux-arm64.tar.gz。执行相同命令解压之后得到一个同名文件夹。此时，输入：

```sh
./nvim-linux-x86_64/bin/nvim #注意第一个应当是你实际下载的版本
```

通常来说，这个时候你就会进入一个nvim的默认的黑白界面。这就意味着你neovim安装成功了。

此外，为了使用方便，你最好还要安装一些剪切板工具，如xclip，以及Nerd Font，用来显示更美观的图标。此部分自行参考相应的文档自己选择进行。

#### kickstart安装

现在我们来安装kickstart。在kickstart的文档中，开发者提示说可以将这个kickstart仓库复刻一份到你自己的仓库下，这样你可以通过git来管理你的配置。

如果你懒得复刻一份，直接输入这个指令，kickstart就会开始自动安装（后记：这里你最好记住这个文件位置~/.config/nvim，因为你之后折腾配置的时候要经常打开它）：

```sh
git clone https://github.com/nvim-lua/kickstart.nvim.git "${XDG_CONFIG_HOME:-$HOME/.config}"/nvim
```

如果复刻了一份到自己的仓库，直接从自己的仓库克隆即可。

在下载好后，再次输入`nvim`，如果一切正常，你就会看见一堆进度条，一堆看不懂的东西安装完后，界面显示出Tokyo night的配色方案，这就意味着你安装成功了。

#### 可能遇到的问题

在某些平台上，stylua或者一些其他插件并没有提供预编译版，这意味着Mason（你可以理解为nvim的包管理器，自动安装、管理各种开发工具）无法自动安装这些工具。你需要自行编译安装。其中比较推荐的是使用**世界上最好的包管理器**cargo来进行编译安装。

要使用cargo，你需要先安装rust。

```sh
pkg install rust #这里开头需要用你自己实际的包管理器
```

安装完后，输入：

```sh
cargo install stylua
```

如果没有意外的话，下载成功后二进制文件会位于`~/.cargo/bin/stylua`（cargo理论上会给出提示）。在`~/.bashrc`或者`~/.zshrc`中加入一行`export PATH="$HOME/.cargo/bin:$PATH"` 退出后输入`source ~/.bashrc`或者`source ~/.zshrc`，之后按理来说stylua就已经可以工作了。

但是这个时候，Mason可能会因为只识别Mason安装的stylua而导致进入nvim后依然报错。你可以:

```sh
nvim ~/.config/nvim/init.lua
```

这就是我们一会会讲到的文件，里面罗列了所有你可能会用到的配置文本。在其中大概718行的位置，你可以找到：

```lua
require("mason-tool-installer").setup {
  ensure_installed = {
    -- 删掉 stylua 或者不要写
  }
}

```

正如上文中的注释，此处本来应该有stylua相关的内容。将它删掉或者注释掉即可。保存退出，再次进入nvim，此时就不会有stylua报错了。你可以通过`:!which stylua`来验证nvim可以找到stylua应用。

除stylua外，还有一些包也可能会因为环境缘故无法直接通过mason安装，不再一一列举。

如果你遇到无法解决的问题，你可以暂时先放一放，继续阅读。之后仍然无法解决，则去求助ai或者其他博客。

---

### init.lua

现在，我们终于可以进入根目录的init.lua了。

```sh
nvim ~/.config/nvim/init.lua
```

如果一切顺利，你会进入一个1000多行的文件，这就是kickstart的配置模版。接下来我们会逐段分析这个配置模版。

### 开头部分

```
=====================================================================
==================== READ THIS BEFORE CONTINUING ====================
=====================================================================
========                                    .-----.          ========
========         .----------------------.   | === |          ========
========         |.-""""""""""""""""""-.|   |-----|          ========
========         ||                    ||   | === |          ========
========         ||   KICKSTART.NVIM   ||   |-----|          ========
========         ||                    ||   | === |          ========
========         ||                    ||   |-----|          ========
========         ||:Tutor              ||   |:::::|          ========
========         |'-..................-'|   |____o|          ========
========         `"")----------------(""`   ___________      ========
========        /::::::::::|  |::::::::::\  \ no mouse \     ========
========       /:::========|  |==hjkl==:::\  \ required \    ========
========      '""""""""""""'  '""""""""""""'  '""""""""""'   ========
========                                                     ========
=====================================================================
=====================================================================

What is Kickstart?

  Kickstart.nvim is *not* a distribution.

  Kickstart.nvim is a starting point for your own configuration.
    The goal is that you can read every line of code, top-to-bottom, understand
    what your configuration is doing, and modify it to suit your needs.

    Once you've done that, you can start exploring, configuring and tinkering to
    make Neovim your own! That might mean leaving Kickstart just the way it is for a while
    or immediately breaking it into modular pieces. It's up to you!

    If you don't know anything about Lua, I recommend taking some time to read through
    a guide. One possible example which will only take 10-15 minutes:
      - https://learnxinyminutes.com/docs/lua/

    After understanding a bit more about Lua, you can use `:help lua-guide` as a
    reference for how Neovim integrates Lua.
    - :help lua-guide
    - (or HTML version): https://neovim.io/doc/user/lua-guide.html

```

简单总结，就是告诉你这不是一个neovim发行版，只是一个模版框架。其中它给出的一个10-15min学习lua的网站 <https://learnxinyminutes.com/zh-cn/lua/> 非常有用，可以帮助你快速掌握使用lua的关键知识。lua并不是一个复杂的语言，甚至比python还要简单的多。下面我简单罗列一些我们可能会用到的lua知识。

### 一些必须的lua基础知识

#### local关键字

在lua当中，如果不使用local声明，变量就会被视为全局变量。在声明函数之前加上local被视为是lua编程的良好风格。

#### require关键字

这个关键字非常重要，它的作用是加载和使用模块。简单来说，你的配置文件可能有复杂的结构，但是neovim并不能全自动地去搜索所有的.lua文件。正如很多其他语言只有一个main.xx文件一样，lua通常而言也只有一个主文件，你需要在这个主文件中引用其他文件的内容。

在我个人看来，lua的文件组织方式和python有许多相似之处。两者都将单个的.lua文件或者.py文件看做一个"模块"，不同功能的模块之间除了文件内容之外没有文件后缀名或者文件名的限制。但是python和lua在一些细节的机制上还是有所不同。

python的模块导出机制可以看做是“默认导出”。一个python模块文件（即一个.py文件）所有的顶级变量、函数、类全都是公开的。比如你在`utils.py`中写了一个名为`myfunction()`的函数，你想在另一个.py文件中使用这个函数，你只需要先`import utils`，再写`utils.myfunction()`即可。可以说，python的每个单独的.py文件都是一个“命名空间”，就像namespace std一样。

但是lua采用了一个与此区别较大的管理方式。简单来说，lua的所有内容都是“私有的”，如果你想让别的.lua文件使用这个.lua文件中的函数等内容，你需要把这些东西“打包发出去”。理解这点对于配置neovim非常关键。

我们下面来看一个简单的例子：

```lua
-- shopping.lua

-- (上面的 local 函数和变量定义)

-- 1. 手动创建一个空的“储物柜” (table)
local M = {}

-- 2. 手动把要公开的东西放进去
M.buy_apple = buy_apple --这是一个shopping.lua中定义的函数
M.PRICE_PER_APPLE = PRICE_PER_APPLE --这是一个shopping.lua中定义的变量
-- 注意：我们故意不把 buy_banana 放进去

-- 3. 把这个填充好的储物柜作为模块的出口
return M
```

当我们想从另一个.lua中调用shopping.lua中的函数或者变量时，我们需要这么写：

```lua
local shopping = require('shopping') --此处就用到了require
```

这时候，我们就在这个.lua文件中定义了一个变量，名字叫`shopping`。它的内容是什么呢？很简单，就是shopping.lua中最后“打包”好的M这个table。

此时，如果你想买苹果，你就写`shopping.buy_apple()`，如果你想知道苹果的单价，那你就写`shopping.PRICE_PER_APPLE`。但是你没办法买香蕉，因为你在shopping.lua中没有把buy_banana()打包发出去。

读到这里，你基本上已经明白了lua的模块管理机制了，但是还不够（因为接下来的内容同样非常重要）。lua并不能完全自动地搜索根目录下匹配名称的.lua文件，而是遵循着一种“内部搜索顺序”。简单来说，这个内部搜索顺序就是一堆相对路径，lua文件运行时会挨个搜索这些路径去找符合条件的.lua文件（比如我们前文写的shopping.lua)，如果它在按照内部搜索顺序寻找无果，那么，它就会报错。

这点之所以重要，就是因为有的时候你必须要让neovim知道它需要的文件到底在哪，但是这个工作是没办法交给ai的，因为你没办法把整个项目都发给ai，ai凭刻板印象给出的代码在解决这类问题时通常无法正常工作，因为它根本不了解你到底把什么文件放在哪了（甚至你自己也很可能未必知道自己需要的文件在哪）。这个时候再把问题发给ai让它解决，ai就会继续给你一个更错的答案，最后恶性循环。（别问我是怎么知道的）

下面我来详细讲解一下“内置搜索顺序”的机制。

#### 内置搜索顺序

lua中用于搜索lua模块的路径变量称为“packege.path”，存储在内置的package表中。

它们通常长成这样：

```
./?.lua;./?/init.lua;/usr/local/share/lua/5.4/?.lua;/usr/local/share/lua/5.4/?/init.lua;...
```

其中，分号分隔了不同的路径，`?`代表此处会被自动替换为你`require`中填写的内容。

例如，假设你执行了 require('socket.core')，Lua 会：

1. 将模块名中的 . 替换为目录分隔符（如 / 或 \）。所以 socket.core 变成了 socket/core。
2. 然后用 socket/core 替换掉 package.path 中每个模板的 ?，并依次检查文件是否存在：
    - ./socket/core.lua (在当前目录下找)
    - ./socket/core/init.lua (在当前目录下的 socket 目录中找 init.lua)
    - /usr/local/share/lua/5.4/socket/core.lua (在标准的 Lua 库目录下找)
    - /usr/local/share/lua/5.4/socket/core/init.lua (在标准的 Lua 库目录的子目录中找)
    - ...等等，直到找到一个存在的文件为止。

一旦找到，lua就会执行这个文件并返回结果（在我们前文的例子中，返回的就是打包好的函数和变量），如果找不到，require就会报错。

那么，当我们需要去寻找一个不在默认搜索路径中的文件该怎么做呢？

下面我们举一个简单的例子:

---

首先，让我们在随便一个测试用的文件夹中创建以下文件结构：

```
.
├── libs
│   └── myutils.lua
└── main.lua

```

然后，让我们在`myutils.lua`中随便写一点测试用的代码：

```lua
-- libs/myutils.lua

-- 创建一个 table 用于导出
local M = {}

function M.greet(name)
    return "Hello, " .. name .. "! Welcome to our module."
end

-- 导出这个 table
return M
```

然后，我们来在`main.lua`中写一个**错误**的模块调用代码：

```lua
-- main.lua

print("Attempting to load the 'myutils' module...")

-- 尝试加载位于 libs 文件夹中的 myutils 模块
-- 注意：我们直接写 'myutils' 而不是 'libs.myutils'，因为 'libs' 只是一个普通文件夹，
-- 并不是 Lua 的包名。我们需要让 Lua 知道要去 'libs' 这个文件夹里找。
local utils = require('myutils')

-- 如果加载成功，就使用它的功能
local message = utils.greet("Alice")
print("Successfully loaded module!")
print(message)
```

这段代码是ai写的。你可能会对其中的注释感到疑惑，我来尝试解答一下：

- 直接写`require('libs.myutils')`其实是可以正常工作的，但是这并不利于维护。我们希望让lua自己通过package path去寻找所需要的模块，而代码中应该只考虑需要什么模块。如果我们在require中强行写入物理位置，虽然可以工作，但是一旦我们调整了文件位置，就不得不修改整个程序中所有从libs中调用模块的代码，这样做非常的低效。
- 在这段代码中，我们构建了一个无法通过package path找到所需模版的main.lua，这完全是反面例子。后文我们会给出修改。

保存退出，输入`lua main.lua`，你就会喜提报错：

```sh
$ lua main.lua
Attempting to load the 'myutils' module...
lua: main.lua:8: module 'myutils' not found:
        no field package.preload['myutils']
        no file '/data/data/com.termux/files/usr/share/lua/5.3/myutils.lua'
        no file '/data/data/com.termux/files/usr/share/lua/5.3/myutils/init.lua'
        no file '/data/data/com.termux/files/usr/lib/lua/5.3/myutils.lua'
        no file '/data/data/com.termux/files/usr/lib/lua/5.3/myutils/init.lua'
        no file './myutils.lua'
        no file './myutils/init.lua'
        no file '/data/data/com.termux/files/usr/lib/lua/5.3/myutils.so'
        no file '/data/data/com.termux/files/usr/lib/lua/5.3/loadall.so'
        no file './myutils.so'
stack traceback:
        [C]: in function 'require'
        main.lua:8: in main chunk
        [C]: in ?
```

这样的报错信息会在你配置nvim过程中时常出现（如果你完全不关心背后的原理）。

从报错信息中，你会看到lua查找了所有的内部搜索路径，但是都找不到你要的`myutils.lua`这个模块，因为你把它放在了libs下，lua根本不知道还有这个地方。

那么如何解决呢？很简单，我们只需要让`main.lua`把`libs`添加到package path中就行。

将`main.lua`修改为以下内容：

```lua
-- main.lua

print("--- Experiment 2: Modifying package.path in script ---")

-- 打印修改前的 path，方便对比
print("Original package.path:")
print(package.path)

-- 将我们的 libs 目录添加到搜索路径的最前面
-- './libs/?.lua' 是一个模板，'?' 会被 require 的模块名替换
package.path = './libs/?.lua;' .. package.path

print("\nModified package.path:")
print(package.path)

print("\nAttempting to load the 'myutils' module...")

local utils = require('myutils')

local message = utils.greet("Bob")
print("Successfully loaded module!")
print(message)
```

再次退出，输入`lua main.lua`你就可以得到想要的内容：

```sh
$ lua main.lua
--- Experiment 2: Modifying package.path in script ---
Original package.path:
/data/data/com.termux/files/usr/share/lua/5.3/?.lua;/data/data/com.termux/files/usr/share/lua/5.3/?/init.lua;/data/data/com.termux/files/usr/lib/lua/5.3/?.lua;/data/data/com.termux/files/usr/lib/lua/5.3/?/init.lua;./?.lua;./?/init.lua

Modified package.path:
./libs/?.lua;/data/data/com.termux/files/usr/share/lua/5.3/?.lua;/data/data/com.termux/files/usr/share/lua/5.3/?/init.lua;/data/data/com.termux/files/usr/lib/lua/5.3/?.lua;/data/data/com.termux/files/usr/lib/lua/5.3/?/init.lua;./?.lua;./?/init.lua

Attempting to load the 'myutils' module...
Successfully loaded module!
Hello, Bob! Welcome to our module.
```

注：以上是我在termux中运行的信息，其他平台上显示的path可能会有所区别。

我们打印了修改前后package.path的内容，显然，在`Modified package.path`中第一个搜索路径就是我们添加的`./libs/?.lua`，此时我们也确实成功使用了`myutils.lua`中的函数。

怎么做到的呢？让我们来仔细看一下我们修改的核心内容：

```lua
-- 将我们的 libs 目录添加到搜索路径的最前面 
-- './libs/?.lua' 是一个模板，'?' 会被 require 的模块名替换 
package.path = './libs/?.lua;' .. package.path
```

`..`在lua中表示字符串连接。这段代码的含义非常清晰：在`package.path`这个字符串前面加上我们需要的搜索路径。就这么简单。`package.path`本质就是一个字符串，你可以随意的根据需要修改本工程的搜索路径。

除此之外，如果你的所有lua工程都采用基本一致的文件结构，你可以把你需要的自定义的搜索路径写入环境变量中。比如你可以在终端中输入`export LUA_PATH="./libs/?.lua;;"`，其中`;;`意味着保留原本的路径。这时候，即使你不在`main.lua`开头编辑`package.path`，lua也会自动找到libs下的myutils.lua，因为所需要的搜索路径已经成为一个环境的一部分了。

但是如果你希望自己的代码可以一次写入、处处运行，那么我建议你还是老老实实的把所需要的搜索路径写在开头，以便在不同环境中都可以使用。

---

讲到这里，我们大概理解了lua的工作原理，现在让我们继续阅读根目录下的init.lua文件。

#### 正式开始分析init.lua

```lua
Kickstart Guide:

  TODO: The very first thing you should do is to run the command `:Tutor` in Neovim.

    If you don't know what this means, type the following:
      - <escape key>
      - :
      - Tutor
      - <enter key>

    (If you already know the Neovim basics, you can skip this step.)

  Once you've completed that, you can continue working through **AND READING** the rest
  of the kickstart init.lua.

  Next, run AND READ `:help`.
    This will open up a help window with some basic information
    about reading, navigating and searching the builtin help documentation.

    This should be the first place you go to look when you're stuck or confused
    with something. It's one of my favorite Neovim features.

    MOST IMPORTANTLY, we provide a keymap "<space>sh" to [s]earch the [h]elp documentation,
    which is very useful when you're not exactly sure of what you're looking for.

  I have left several `:help X` comments throughout the init.lua
    These are hints about where to find more information about the relevant settings,
    plugins or Neovim features used in Kickstart.

   NOTE: Look for lines like this

    Throughout the file. These are for you, the reader, to help you understand what is happening.
    Feel free to delete them once you know what you're doing, but they should serve as a guide
    for when you are first encountering a few different constructs in your Neovim config.

If you experience any errors while trying to install kickstart, run `:checkhealth` for more info.

I hope you enjoy your Neovim journey,
- TJ

P.S. You can delete this when you're done too. It's your config now! :)
--]]

```

总结一下：

- 你可以在普通模式下`:Tutor`来进入一个neovim的使用教程，告诉你这个软件的基本使用方式。
  - vim的使用方式可以非常花哨，内置的指令多的数不过来。遗憾的是我只会其中最基本的几个指令，但是这通常来讲已经足够了。你不需要为了少按几个上下左右而去背一大堆指令，这样做有用，但是对于一个入门的人来说完全没有必要。
- 你可以在普通模式下`:help`来查看帮助。里面罗列了nvim支持的各种功能，以及一些基本的使用方式。这就是一部使用说明书，就和你买的家用电器里附带的说明书是一个性质。
- 你可以`<space>+s+h`来进入一个帮助搜索界面。（然而我不会用）
- 文件剩余的内容都随便你配置。

```lua
-- Set <space> as the leader key
-- See `:help mapleader`
--  NOTE: Must happen before plugins are loaded (otherwise wrong leader will be used)
vim.g.mapleader = ' '
vim.g.maplocalleader = ' '
-- Set to true if you have a Nerd Font installed and selected in the terminal
vim.g.have_nerd_font = false
```

- `vim.g`：代表全局变量。这里设置的变量在全局都有效。
- `vim.g.mapleader = ' '`：将空格键设置为“前导键”，用来作为一个通用的快捷键前缀。
- `vim.g.maplocalleader = ' '`：局部映射，简单来说就是只在某些文件中才会生效的“前导键”，之后`<localleader>x`就表示只在此情况下才生效的快捷键。
- `vim.g.have_nerd_font = false`：声明你的终端环境有没有安装Nerd Font。很多nvim的插件需要nerd font来支持一些复杂的图形界面，如果你没有安装nerd font，通过在这里声明false，那些插件就会只显示一些普通文本来适配你的环境。
  - 如果你安装了nerd font，记得把这个变量设置为true。

##### setting options

```lua
-- [[ Setting options ]]
-- See `:help vim.o`
-- NOTE: You can change these options as you wish!
--  For more options, you can see `:help option-list`

-- Make line numbers default
vim.o.number = true
-- You can also add relative line numbers, to help with jumping.
--  Experiment for yourself to see if you like it!
-- vim.o.relativenumber = true

-- Enable mouse mode, can be useful for resizing splits for example!
vim.o.mouse = 'a'

-- Don't show the mode, since it's already in the status line
vim.o.showmode = false

-- Sync clipboard between OS and Neovim.
--  Schedule the setting after `UiEnter` because it can increase startup-time.
--  Remove this option if you want your OS clipboard to remain independent.
--  See `:help 'clipboard'`
vim.schedule(function()
  vim.o.clipboard = 'unnamedplus'
end)

-- Enable break indent
vim.o.breakindent = true

-- Save undo history
vim.o.undofile = true

-- Case-insensitive searching UNLESS \C or one or more capital letters in the search term
vim.o.ignorecase = true
vim.o.smartcase = true

-- Keep signcolumn on by default
vim.o.signcolumn = 'yes'

-- Decrease update time
vim.o.updatetime = 250

-- Decrease mapped sequence wait time
vim.o.timeoutlen = 300

-- Configure how new splits should be opened
vim.o.splitright = true
vim.o.splitbelow = true

-- Sets how neovim will display certain whitespace characters in the editor.
--  See `:help 'list'`
--  and `:help 'listchars'`
--
--  Notice listchars is set using `vim.opt` instead of `vim.o`.
--  It is very similar to `vim.o` but offers an interface for conveniently interacting with tables.
--   See `:help lua-options`
--   and `:help lua-options-guide`
vim.o.list = true
vim.opt.listchars = { tab = '» ', trail = '·', nbsp = '␣' }

-- Preview substitutions live, as you type!
vim.o.inccommand = 'split'

-- Show which line your cursor is on
vim.o.cursorline = true

-- Minimal number of screen lines to keep above and below the cursor.
vim.o.scrolloff = 10

-- if performing an operation that would fail due to unsaved changes in the buffer (like `:q`),
-- instead raise a dialog asking if you wish to save the current file(s)
-- See `:help 'confirm'`
vim.o.confirm = true

```

- `vim.o`：设置nvim的各种行为选项。
- `vim.o.number = true`：显示==行号==。默认显示的是绝对行号
- `vim.o.relativenumber = true`：显示==相对行号==。
  - 你可能会好奇如果两个都启用会怎么样。答案很简单：你所处的行会显示绝对行号，而其他行会显示相对行号。这样你既能确定自己的绝对位置，又方便跳转位置，这是很多人的选择。
- `vim.o.mouse = 'a'`：==鼠标模式==，a表示对所有文件都适用。如果你有鼠标，你没有理由把这个这个关掉。
- `vim.o.showmode = false`：不显示nvim内置的==状态提示==。更加现代的状态栏插件可以提供比文本更美观的状态栏，所以除非你有特殊的美术追求，没必要调为true。
- `vim.schedule(function()vim.o.clipboard = 'unnamedplus'end)`：`vim.schedule`表示在ui完全加载完之后再执行，后面的函数表示将nvim的==剪切板与操作系统剪切板同步==。你可以在nvim中`y`复制文本之后在外部`ctrl v`粘贴，也可以外部`ctrl c`复制后在nvim中`p`粘贴。
  - 这是一个非常有必要的设置，尤其是当你需要和ai博弈的时候。
- `vim.o.breakindent = true`：==打断缩进==，文本太长自动换行后保持和原文本同样的缩进。
- `vim.o.undofile = true`：保存==撤回树==，当你再次打开这个文件时，你仍然可以撤回到之前的状态，即使中间你保存了文件。实现原理是另外建立一个`.un~`文件，所以理论上你可以撤回到宇宙大爆炸（如果你从那时就开始写代码）
- `vim.o.ignorecase = true` `vim.o.smartcase = true`：==搜索==时忽视大小写，但是如果你输入了至少一个大写时又会自动大小写敏感。
- `vim.o.signcolumn = 'yes'`： 保持==侧边栏显示==，Lsp（后面会涉及）和一些其他的插件需要依赖这个侧边栏来显示一些信息，所以保持开启。
- `vim.o.updatetime = 250`：==更新时间==，你可以理解为minecraft中的随机刻，很多插件的使用依赖于这个更新时间，如果减少，一些插件的响应速度会加快。
- `vim.o.timeoutlen = 300`：==映射序列等待时间==，比如你的一个映射是`j+k`，当你按下`j`时，nvim会等待300ms，如果你在这时间内按下`k`，那么就会执行映射。
- `vim.o.splitright = true` `vim.o.splitbelow = true`：新的水平==分屏==和垂直分屏会分别出现在右边和下边。
- `vim.o.list = true`：开启list模式，让nvim可以显示一些特殊的==空白字符==
- `vim.opt.listchars = { tab = '» ', trail = '·', nbsp = '␣' }`：将tab显示为》，将尾部的多余空格显示为·，将不可断空格显示为␣。
- `vim.o.inccommand = 'split'`：执行替换命令时在一个新的==分屏窗口==中实时显示预览效果。
- `vim.o.cursorline = true`：==高亮==光标所在的当前行。
- `vim.o.scrolloff = 10`：光标滚动时会==偏移上下边缘10行==，方便阅读。
- `vim.o.confirm = true`：在一些可能导致数据丢失的情况下==要求你确认==，比如未保存时就退出。除非你是人形计算机永远可以保证正确，那么建议你保持开启。

```lua
-- [[ Basic Keymaps ]]
--  See `:help vim.keymap.set()`

-- Clear highlights on search when pressing <Esc> in normal mode
--  See `:help hlsearch`
vim.keymap.set('n', '<Esc>', '<cmd>nohlsearch<CR>')

-- Diagnostic keymaps
vim.keymap.set('n', '<leader>q', vim.diagnostic.setloclist, { desc = 'Open diagnostic [Q]uickfix list' })

-- Exit terminal mode in the builtin terminal with a shortcut that is a bit easier
-- for people to discover. Otherwise, you normally need to press <C-\><C-n>, which
-- is not what someone will guess without a bit more experience.
--
-- NOTE: This won't work in all terminal emulators/tmux/etc. Try your own mapping
-- or just use <C-\><C-n> to exit terminal mode
vim.keymap.set('t', '<Esc><Esc>', '<C-\\><C-n>', { desc = 'Exit terminal mode' })

-- TIP: Disable arrow keys in normal mode
-- vim.keymap.set('n', '<left>', '<cmd>echo "Use h to move!!"<CR>')
-- vim.keymap.set('n', '<right>', '<cmd>echo "Use l to move!!"<CR>')
-- vim.keymap.set('n', '<up>', '<cmd>echo "Use k to move!!"<CR>')
-- vim.keymap.set('n', '<down>', '<cmd>echo "Use j to move!!"<CR>')

-- Keybinds to make split navigation easier.
--  Use CTRL+<hjkl> to switch between windows
--
--  See `:help wincmd` for a list of all window commands
vim.keymap.set('n', '<C-h>', '<C-w><C-h>', { desc = 'Move focus to the left window' })
vim.keymap.set('n', '<C-l>', '<C-w><C-l>', { desc = 'Move focus to the right window' })
vim.keymap.set('n', '<C-j>', '<C-w><C-j>', { desc = 'Move focus to the lower window' })
vim.keymap.set('n', '<C-k>', '<C-w><C-k>', { desc = 'Move focus to the upper window' })

-- NOTE: Some terminals have colliding keymaps or are not able to send distinct keycodes
-- vim.keymap.set("n", "<C-S-h>", "<C-w>H", { desc = "Move window to the left" })
-- vim.keymap.set("n", "<C-S-l>", "<C-w>L", { desc = "Move window to the right" })
-- vim.keymap.set("n", "<C-S-j>", "<C-w>J", { desc = "Move window to the lower" })
-- vim.keymap.set("n", "<C-S-k>", "<C-w>K", { desc = "Move window to the upper" })

```

- `vim.keymap.set()`：这是 Neovim 推荐的用来设置快捷键的函数。它比老式的 `map` 命令更清晰、功能更强大。
- `vim.keymap.set('n', '<Esc>', '<cmd>nohlsearch<CR>')`：在**普通模式** (Normal Mode)下（n就表示普通模式），当你按下 `<Esc>` 键时，自动执行 `:nohlsearch` 命令来清除上一次搜索结果的高亮。
  - 这是一个极其舒适的设置。你用 `/` 搜索完一个词后，所有匹配项都会高亮，有时会很晃眼。这个快捷键让你可以在不经意间（因为 `<Esc>` 是最常用的键之一）就清除掉这些高亮，让界面恢复清爽。
- `vim.keymap.set('n', '<leader>q', vim.diagnostic.setloclist)`：在**普通模式**下，按下你之前设置的 `Leader` 键（比如空格），再按下 `q`，就会打开一个**诊断信息的列表 (Quickfix list)**。
  - 当你的代码有错误或警告时（由 LSP 提供），这个列表会把所有问题都集中展示出来，方便你逐个跳转和修复。`q` 很好记，可以联想成 “Quickfix”。
- `vim.keymap.set('t', '<Esc><Esc>', '<C-\\><C-n>')`：在**终端模式** (Terminal Mode)下，**连按两次 `<Esc>` 键**，效果等同于按下那个非常难记的 `<C-\><C-n>` 组合键，从而**退出终端模式**回到普通模式。
  - Neovim 内置的终端非常好用，但它的默认退出方式简直反人类。这个设置用一个更符合直觉的方式解决了这个问题。不过注释也提醒了，这个映射在某些复杂的终端环境（比如 tmux 嵌套）下可能不生效，那时就只能老老实实按默认的组合键了。
- `-- TIP: Disable arrow keys in normal mode` (被注释掉的代码块)：这是一个给新手的**提示性配置**，建议**禁用在普通模式下的方向键**。
  - 这么做的目的是为了“强迫”自己习惯使用 `h j k l` 来移动光标。这是 Vim 的精髓之一，因为你的手不需要离开主键区，移动效率会高得多。一旦你习惯了，就会发现方向键又远又慢。如果你想养成这个好习惯，可以把这段代码的注释去掉。
- `vim.keymap.set('n', '<C-h>', '<C-w><C-h>')` (以及下面 `jkl` 的三行)：这一组快捷键让你可以在**普通模式**下，使用 `Ctrl + h/j/k/l` 来**在不同的分屏窗口之间切换焦点**。
  - Neovim 原生的窗口切换命令是 `<C-w>` 再加上 `h/j/k/l`，需要按两次。这组设置把两步操作简化成了一步，让窗口导航变得像在单个文件中移动光标一样流畅自然，极大地提升了分屏工作的效率。
- `-- NOTE: Some terminals have colliding keymaps...` (被注释掉的代码块)：这是另一组**提示性配置**，它尝试将 `Ctrl + Shift + h/j/k/l` 映射为**移动整个窗口的位置**，而不是仅仅移动焦点。
  - 比如，按下 `Ctrl + Shift + h` 会把当前所在的窗口整个移动到左边去。这对于整理窗口布局很有用。
  - 但是，注释明确指出，很多终端程序无法正确识别 `Ctrl + Shift` 和 `Ctrl` 的区别，导致这个快捷键可能无法生效。所以它被默认注释掉了，需要你自己测试和决定是否启用。

原谅我上面这段文字用了ai。ai描述通常会有过多的描述，导致看起来观感不那么清爽。不过没关系，我来总结一下：

- `vim.keymap.set('<mode>','<key>','<command>',{desc = ""})`是配置键盘映射的基本语句，其中第一个参数表示模式，n,v,i,c,t分别代表普通模式、可视模式、插入模式、命令行模式、终端模式。这表明了该映射的生效场景。
- 第二个参数表示你按下的按键，特殊按键使用尖括号包裹。大致如下：
  - \<CR>: 回车键 (Carriage Return)
  - \<Esc>: Escape 键
  - \<Space>: 空格键
  - \<leader>: 你设置的 Leader 键
  - \<C-h>: Ctrl + h
  - \<S-h>: Shift + h
  - \<A-h>: Alt + h (在 Lua 中也写作 \<M-h>)
  - 例：
    - `<leader>q`前导键+q
    - `<C-h>`Ctrl+h
    - `<Esc><Esc>`按两次esc
- 第三个参数表示指令，你可以用单引号包裹一个指令来直接表示一个指令，也可以仿照第二个参数来模拟按键。
- 第四个参数是选项，写入desc可以为快捷键写入描述。其他的可选项不在此罗列。

现在，我们来写一些简单的快捷键，方便我们编写。

```lua
vim.keymap.set('i','jk','<Esc>') --用jk来从插入模式快速切换回普通模式，比按esc快的多
vim.keymap.set('n', '<leader>v', '<cmd>vsplit<CR>', { desc = '[V]ertical split' }) --创建垂直分屏的快速方式
vim.keymap.set('n', '<leader>v', '<cmd>vsplit<CR>', { desc = '[V]ertical split' }) --创建水平分屏
vim.keymap.set('n', '<leader>c', '<cmd>close<CR>', { desc = '[C]lose window' }) --关闭分屏
vim.keymap.set('n', '<leader>bn', '<cmd>bnext<CR>', { desc = 'Go to [N]ext [B]uffer' })
vim.keymap.set('n', '<leader>bp', '<cmd>bprevious<CR>', { desc = 'Go to [P]revious [B]uffer' })
vim.keymap.set('n', '<leader>bd', '<cmd>bdelete<CR>', { desc = '[D]elete/close buffer' })
```

总之，这部分代码主要就是让你在感觉有的快捷键比较反人类的时候可以随意将它修改为自己喜欢的快捷键。以上部分代码仅供参考。

---

限于篇幅，init.lua内容的介绍暂时到这里。但是我们并没有结束：接下来，让我们来尝试将options和keymap部分的代码移植到其他地方去。所有的配置全部都放在一个init.lua文件中并不利于后期的维护与调整。理论上，根据我们前文所学的知识，我们已经足够可以实现配置文件的分模块管理了。

首先，让我们回到根目录：

```sh
cd ~/.config/nvim
```

接着，在`~/.config/nvim/lua/custom`目录下创建两个新的lua文件来存放我们写好的keymap和option部分代码：

```sh
touch lua/custom/options.lua
touch lua/custom/keymaps.lua
```

然后，将根目录下init.lua中`vim.o`与`vim.keymaps`相关的代码分别剪切到这两个文件中。注意，当你使用vim内置的`d`与`p`剪切粘贴工具时，因为你要移动的`vim.o`涉及到了剪切板工具，`vim.keymap`涉及到了快捷键设置，你可能在此过程中会遇到一点小问题。以防万一，你可以先把代码复制到外部，再挨个复制进对应文件，这样以防代码丢失。

此时，你就可以完全删除init.lua开头到keymaps位置的全部代码（除了全局变量），并补上：

```lua
require 'custom.options'
require 'custom.keymaps'
```

不出意外的话，保存退出再重进，你原先写在init.lua中的配置此时再次会发挥作用。你可以通过这种方式让你的配置文件更加易于维护。

大概先这样，我还会就界面个性化、lsp等比较棘手的配置问题再写几篇博客。
