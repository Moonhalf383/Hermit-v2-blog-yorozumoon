---
title: "NeovimPlugins配置方法"
date: 2025-11-23T11:11:53+08:00
draft: false 
toc: true 
images:
tags: ['Neovim','教程'] 
---

我的上一篇博客中，我们探讨了如何配置nvim。但是仅仅是修改了键位和vim行为远远不足以让我们的nvim可以承担代码任务。我们需要插件来拓展nvim的功能。插件是nvim的灵魂，有了插件，你的nvim才能作为一个功能完备的ide来使用。(后记：真正使用时你应该去使用lazyvim，并通过lazyvim的lazyextra功能配置lsp和相关的调试功能。如果你有生活，不要尝试自己配置lsp，尤其是如果你想写java之类的面向大型工程设计的语言。)

### 插件的分类

nvim的插件大致有以下几类:

- 基础辅助。拓展vim的行为。
- git集成
- 外观主题
- 代码智能
  - 自动补全
  - 语法高亮
  - 语法诊断
  - 调试与运行
  - 代码格式化
- 模糊搜索

至少在kickstart中出现的插件可以大致归类为以上内容。

### 插件管理器

光有插件是远远不够的，我们需要一个组件来专门管理插件，决定插件何时运行。通常而言，我们选择使用lazy.nvim插件管理器，它最大的特点就是“懒加载”，只在需要的时候加载插件功能，这极大地提高了载入速度。

让我们看到init.lua中关于下载lazy.nvim插件管理器的部分：

```lua
-- [[ Install `lazy.nvim` plugin manager ]]
--    See `:help lazy.nvim.txt` or https://github.com/folke/lazy.nvim for more info
local lazypath = vim.fn.stdpath 'data' .. '/lazy/lazy.nvim'
if not (vim.uv or vim.loop).fs_stat(lazypath) then
  local lazyrepo = 'https://github.com/folke/lazy.nvim.git'
  local out = vim.fn.system { 'git', 'clone', '--filter=blob:none', '--branch=stable', lazyrepo, lazypath }
  if vim.v.shell_error ~= 0 then
    error('Error cloning lazy.nvim:\n' .. out)
  end
end

---@type vim.Option
local rtp = vim.opt.rtp
rtp:prepend(lazypath)
```

这段代码的含义，简单来说就是：

- 创建一个lazypath作为这个插件管理器的老家
  - 在linux系统中，通常就是`~/.local/share/nvim/lazy/lazy.nvim`
- 将lazy.nvim在github上的仓库克隆到lazypath
- 其余部分的代码基本是出于兼容性而写的（但是并非无用，这段代码的健壮性还是非常强的，能让你在绝大多数情况下都能至少保证lazy.nvim安装成功）

这段代码基本唯一有价值的东西就是告诉你lazypath的老家在哪，实在有问题可能会要去检查那里的文件。但是通常而言用不到。而且你大概率也不会自己写这段代码，所以多数情况下忽视即可。

### 插件

```lua
require('lazy').setup({
```

从接下来开始就到了重头戏了，nvim的插件管理。

在我们kickstart nvim这个配置模版中，大几百行的代码都塞在了`require('lazy').setup({`这个巨大括号里。

`require('lazy').setup()`接受的是两个参数，其中第一个参数就是一个巨大的lua table，里面放的是插件的“说明书”，第二个参数则是可选的，用来定制lazy.nvim自身的外观和行为。我们主要关注第一部分，即插件的“说明书”到底怎么写。

#### 插件说明书 (Plugin Specification)

所谓的插件说明书，就是setup()中第一个参数table中的元素。每一个说明书就代表了一个插件。

一个插件说明书的职能大概有：

- 让lazy.nvim去下载需要的插件
- 传递配置选项给插件
- 加载触发器来实现自动功能
- 设置自定义函数来配置插件（比如设置快捷键）
- 声明依赖项
- 设置加载条件
- 设置打开特定文件时加载插件
- 导入其他文件中的插件配置

好吧，光看这些职能是没有用的，我们来以init.lua中提供的插件说明书来详细讲解。

#### 最简形式

```lua
  'NMAC427/guess-indent.nvim', -- Detect tabstop and shiftwidth automatically
```

没错，一个字符串就足够作为一个插件了。记得后面要加逗号。

这段代码的含义就是，到github上找到`NMAC427`这个用户的`guess-indent.nvim`这个仓库并下载下来，什么都不配置，直接用默认配置即可。这是最省事的写法。

但是多数时候，你为了更方便的使用插件，不得不进行一些相应的配置。这个时候，我们就要用到更加复杂的配置方式。

#### 常见形式

```lua
  -- Use `opts = {}` to automatically pass options to a plugin's `setup()` function, forcing the plugin to be loaded.
  --

  -- Alternatively, use `config = function() ... end` for full control over the configuration.
  -- If you prefer to call `setup` explicitly, use:
  --    {
  --        'lewis6991/gitsigns.nvim',
  --        config = function()
  --            require('gitsigns').setup({
  --                -- Your gitsigns configuration here
  --            })
  --        end,
  --    }
  --
  -- Here is a more advanced example where we pass configuration
  -- options to `gitsigns.nvim`.
  --
  -- See `:help gitsigns` to understand what the configuration keys do
  { -- Adds git related signs to the gutter, as well as utilities for managing changes
    'lewis6991/gitsigns.nvim',
    opts = {
      signs = {
        add = { text = '+' },
        change = { text = '~' },
        delete = { text = '_' },
        topdelete = { text = '‾' },
        changedelete = { text = '~' },
      },
    },
  },
```

这是一个git图标插件，功能是在侧边栏显示当前文本中的改动增减。但是我们想要自定义显示的图标，那么我们就必须进行配置。

配置的方式如上，简单来说就是将插件说明书写成一个table。lua中table是唯一的组合数据结构，功能类似于js的对象，既能当哈希表或字典，又能当列表，还能作为类。如果想要写好lua程序，你将不得不直面这种身兼数职的境况。

在以上这段代码中，配置gitsigns时我们使用了`opts`这个配置键。这是最简洁的配置键，配置的选项又插件作者提供，插件加载完成后会自动调用你给出的配置。

其他的插件配置键我们暂且不谈，我们不妨先由一个简单而效果显著的插件配置出发，来自己手动完成一个配置：配色方案。

### 配色方案配置

```lua
  { -- You can easily change to a different colorscheme.
    -- Change the name of the colorscheme plugin below, and then
    -- change the command in the config to whatever the name of that colorscheme is.
    --
    -- If you want to see what colorschemes are already installed, you can use `:Telescope colorscheme`.
    'folke/tokyonight.nvim',
    priority = 1000, -- Make sure to load this before all the other start plugins.
    config = function()
      ---@diagnostic disable-next-line: missing-fields
      require('tokyonight').setup {
        styles = {
          comments = { italic = false }, -- Disable italics in comments
        },
      }

      -- Load the colorscheme here.
      -- Like many other themes, this one has different styles, and you could load
      -- any other, such as 'tokyonight-storm', 'tokyonight-moon', or 'tokyonight-day'.
      vim.cmd.colorscheme 'tokyonight-night'
    end,
  },
```

在kickstart配置模版中，默认的配色方案长成这样。它下载的是tokyonight，一个我个人不是很喜欢的配色方案。所以我一定要把它改掉。

我们先来分析一下这段代码的组成：

- `'folke/tokyonight.nvim',`作者+插件名，一个插件的最简形式。
- `priority = 1000,`设置优先级。颜色主题需要较高的优先级，这样才能保证在你看到任何界面元素之前，颜色主题已经加载完毕，使视觉体验更平滑。
- `require('tokyonight').setup`调用插件提供的标准配置函数
- `comments = { italic = false }`禁用注释斜体，这个功能在某些平台上表现不佳（比如我的termux）
- `vim.cmd.colorscheme 'tokyonight-night'`在完成前面的setup步骤后，让nvim真正开始加载这个颜色主题。

那么我们现在就来修改这段代码，来改成我更喜欢的onedarkpro主题。

```lua
  {
    'olimorris/onedarkpro.nvim',
    priority = 1000,
    config = function()
      require('onedarkpro').setup {
        highlights = {
          Comment = { italic = false },
          Directory = { bold = true },
          ErrorMsg = { italic = false, bold = true },
        },
      }
      vim.cmd 'colorscheme onedark'
    end,
  },
```

ok，当我保存退出再进入后，我就喜提新颜色主题了。

简单来说，这部分内容难点就在于：如果你不去深入了解配置文件的组成结构，你可能根本不知道github上给出的配置代码到底该放在哪。现在我们知道了，只需要把github上给出的配置代码塞进一个table里，再塞到`require('lazy').setup({`里即可。

颜色方案配置比较简单，因为它不涉及任何按键的设置。接下来我们来为我们的nvim添加一个“浮动终端”，就想一般的图形化界面ide那样在界面的中央弹出一个终端窗口，而非使用nvim原生的类似分屏的终端。

但是在开始之前，我们先进行一些准备工作：我们实际上应该把这些插件说明书分门别类的放在`lua/custom/plugins`目录下，而不是全部扁平化的放在init.lua中。让我们先把颜色方案的代码移动过去。

首先，我们看到`init.lua`中插件说明书的最后部分：

```lua
  -- note: the import below can automatically add your own plugins, configuration, etc from `lua/custom/plugins/*.lua`
  --    this is the easiest way to modularize your config.
  --
  --  uncomment the following line and add your plugins to `lua/custom/plugins/*.lua` to get going.
  { import = 'custom.plugins' },
```

这一部分中，`\{ import = 'custom.plugins' },`原本是注释起来的，将前面的注释符号删除，这样lazy.nvim就会自动去在custom/plugins目录下寻找插件说明书。

接着，输入以下指令：

```sh
nvim lua/custom/plugins/colorscheme.lua
```

这样我们就创建了一个专门存放配色方案的lua模块。将原本我们写在init.lua中的代码移植过来，并用`return{}`包裹，得到：

```lua
--colorscheme.lua
return {
  {
    'olimorris/onedarkpro.nvim',
    priority = 1000,
    config = function()
      require('onedarkpro').setup {
        highlights = {
          Comment = { italic = false },
          Directory = { bold = true },
          ErrorMsg = { italic = false, bold = true },
        },
      }
      vim.cmd 'colorscheme onedark'
    end,
  },
}
```

此时，重新打开nvim，配色方案应该依然工作正常。这就意味着你成功构建了一个插件说明书。

#### 浮动终端

浮动终端的常用插件是`toggleterm.nvim`，简称toggle。

让我们先在plugins下创建一个专属于toggle的lua模块：

```sh
nvim lua/custom/plugins/toggleterm.lua
```

然后，让我写入插件的配置代码：

```lua
return {
  'akinsho/toggleterm.nvim',
  version = '*',
  config = function()
    require('toggleterm').setup {
      -- 设置终端窗口大小
      size = function(term)
        if term.direction == 'horizontal' then
          return 15
        elseif term.direction == 'vertical' then
          return vim.o.columns * 0.4
        end
      end,
      -- 打开终端的快捷键，这里设置为 Ctrl + \
      open_mapping = [[<c-\>]],
      -- 隐藏 toggleterm buffer 中的行号
      hide_numbers = true,
      -- 当 Neovim 目录切换时，终端下次打开也会切换到相应目录
      autochdir = true,
      -- 设置终端窗口的背景色，可以根据你的主题进行调整
      highlights = {
        Normal = {
          guibg = 'none', -- 这里设置为 "none" 表示使用默认背景
        },
        NormalFloat = {
          link = 'Normal',
        },
      },
      -- 阴影效果，如果设置了上面的 highlights 中的 Normal，建议将此项设为 false
      shade_terminals = true,
      -- 阴影的强度，数值越小越深
      shading_factor = -30,
      -- 启动时进入插入模式
      start_in_insert = true,
      -- 在插入模式下也可以使用 open_mapping 打开终端
      -- insert_mappings = true,
      -- 在终端窗口中也可以使用 open_mapping (会覆盖一些终端默认行为)
      -- terminal_mappings = true,
      -- 保持上次终端的大小
      -- persist_size = true,
      -- 终端进程退出时自动关闭窗口
      direction = 'float', -- 设置默认打开方式为浮动窗口
      close_on_exit = true,
      -- 使用 Neovim 的默认 shell
      shell = vim.o.shell,
      -- 自动滚动到底部
      auto_scroll = true,
      -- 浮动窗口的设置
      float_opts = {
        -- 边框样式
        border = 'curved', -- 使用圆角边框
        -- 窗口透明度
        winblend = 20,
      },
    }

    -- 设置一个函数，用于在 normal 模式下按下 <esc> 时退出终端
    function _G.set_terminal_keymaps()
      local opts = { buffer = 0 }
      vim.keymap.set('t', '<esc>', [[<C-\><C-n>]], opts)
      vim.keymap.set('t', 'jk', [[<C-\><C-n>]], opts) -- jk 也可以退出
      vim.keymap.set('t', '<C-h>', [[<Cmd>wincmd h<CR>]], opts)
      vim.keymap.set('t', '<C-j>', [[<Cmd>wincmd j<CR>]], opts)
      vim.keymap.set('t', '<C-k>', [[<Cmd>wincmd k<CR>]], opts)
      vim.keymap.set('t', '<C-l>', [[<Cmd>wincmd l<CR>]], opts)
    end

    -- 当打开终端时，自动应用上面的快捷键设置
    vim.cmd('autocmd! TermOpen term://* lua set_terminal_keymaps()')

    -- 定义一个快捷键，方便地打开一个 LazyGit 终端
    local Terminal = require('toggleterm.terminal').Terminal
    local lazygit = Terminal:new({
      cmd = "lazygit",
      hidden = true,
      direction = "float",
    })

    function _LAZYGIT_TOGGLE()
      lazygit:toggle()
    end

    vim.keymap.set("n", "<leader>gg", "<cmd>lua _LAZYGIT_TOGGLE()<CR>", {
        noremap = true,
        silent = true,
        desc = "Lazygit",
    })

  end,
}
```

在了解了原理之后，和ai交流的难度显著下降了。我得到了一套非常棒的配置，甚至帮我添加了lazygit的界面。效果非常帅。

这时候保存退出，再重进，你就应该可以通过你设置的快捷键来打开toggle终端了。

---

看到这里，你已经可以完全自己上手去解决插件配置问题了。实际上，你基本不需要自己写代码，你只要能读的懂一点代码就行，你只需要把github上插件作者提供的配置样例丢给ai让它生成一个涵盖了大多数功能的配置，再依据自己的需求进行删改即可。
