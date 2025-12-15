---
title: "工具不图鉴02：zsh&ohmyzsh"
date: 2025-12-15T23:25:53+08:00
draft: false 
toc: true 
images:
tags: ['工具不图鉴','教程', 'zsh', 'ohmyzsh'] 
---
参考链接：

[zsh wiki](https://zh.wikipedia.org/wiki/Z_shell)

[ohmyzsh](https://ohmyz.sh/)

## 什么是zsh？

zsh是一个功能丰富的Unix shell，即**命令行解释器**。就像安装git时附送的git bash和windows系统下自带的powershell一样，你把指令输进去，按下回车，指令就运行起来了。

相较于其他命令行解释器如bash，zsh具有更强大的功能，如自动补全，历史记录，切换主题等。这一切都可以通过添加插件来完成。

---

### 如何安装zsh？

zsh并不是一个更新非常迅速的软件，这意味着你并不一定需要安装最新版才能确保最大程度的稳定和功能。通常来说，直接使用你的系统级包管理器来安装即可，不会导致兼容性问题。

```sh
sudo apt install zsh
```

---

## 什么是ohmyzsh？

ohmyzsh是一个开源的社区驱动的zsh配置管理框架。通常来说，直接为zsh添加插件并不方便。但是通过ohmyzsh，你只需要克隆一下插件仓库，再修改一下~/.zshrc即可使用插件。

omz拥有非常丰富的插件生态以及活跃的贡献者社区，可以大大提高终端美观程度以及使用效率。

---

#### 如何安装ohmyzsh？

不同于zsh以及很多其他的基本工具，ohmyzsh是一个迭代迅速的社区工具。这意味着如果想要使用更新更好的插件，最好通过官方渠道进行安装。

这里罗列从[omz官网](https://ohmyz.sh/#install)上摘录的下载指令。你只需要将指令输入终端运行即可。

方法一：通过curl安装

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

方法二：通过wget安装

```sh
sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

这两个指令是相对稳定的，它们不会随omz版本迭代而发生改变。

---

## ohmyzsh如何配置？

将zsh设置为默认shell：

```sh
chsh -s /bin/zsh
```

---

### 常用插件

#### 1. zsh-autosuggestions (自动补全建议)

根据历史记录给出自动补全建议，按下右键自动补全。

[github网页](https://github.com/zsh-users/zsh-autosuggestions)

```sh
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

---

#### 2. zsh-syntax-highlighting (语法高亮)

当你输入命令时，正确的命令显示绿色，错误的显示红色，路径带有下划线。

[github网页](https://github.com/zsh-users/zsh-syntax-highlighting)

```sh
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

---

#### 3. git（内置插件）

提供git简写别名，并支持tab补全。

- 常用别名：
  - gst -> git status
  - gco -> git checkout
  - gcm -> git checkout master (或 main)
  - gp -> git push
  - gl -> git pull

---

#### 4. z（内置插件）

记录你访问过的目录，之后只需输入 `z + 目录名片段` 即可直接跳转，无需输入完整路径。

- 用法：`z down` (可能直接跳转到 `~/Downloads`)

---

#### 5. extract（内置插件）

通用解压插件。不需要记住 `tar -xvf`、`unzip` 等繁琐参数，统一使用 `x` 命令解压任何格式的压缩包。

- 用法：`x filename.tar.gz`

---

### 启用插件

下载完第三方插件后，必须在 `.zshrc` 文件中配置才能生效。

**1. 编辑配置文件：**

```bash
nano ~/.zshrc
# 或者使用 vim ~/.zshrc
```

**2. 找到 `plugins=(...)` 这一行，修改为：**
(注意：插件名之间用**空格**分隔，不要用逗号)（想使用哪些插件就写入哪些插件）

```bash
plugins=(
    git
    z
    extract
    sudo
    web-search
    docker
    zsh-autosuggestions       # 第三方
    zsh-syntax-highlighting   # 第三方，建议放在列表最后
)
```

**3. 使配置生效：**

```bash
source ~/.zshrc
```

---

### 常见美化方案

[可视化主题百科](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)

在`~/.zshrc`中找到`ZSH_THEME`配置项，将对应位置的主题改为你下载的主题，之后`source ~/.zshrc`即可启用。

---

#### 1. Powerlevel10k

目前相当流行的主题，配置方便，高度可自定义。

[github网页](https://github.com/romkatv/powerlevel10k)

```sh
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

主题名：`powerlevel10k/powerlevel10k`

---

#### 2. ys

我个人比较喜欢的主题，非常简洁美观，并且是omz的内置主题。直接将主题名改为ys即可。

---

# 集成式配置脚本

一键配置zsh&omz全套环境。

```sh
#!/bin/bash

# ============================================================
#  Zsh + OMZ + Plugins + P10k 一键安装脚本 (含 Termux 支持)
# ============================================================

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 定义路径
OMZ_DIR="$HOME/.oh-my-zsh"
ZSH_CUSTOM="$OMZ_DIR/custom"
ZSHRC="$HOME/.zshrc"

echo -e "${BLUE}>>> 开始执行集成安装脚本...${NC}"

# 1. 环境检测与软件安装
# -----------------------------------------------------------
install_packages() {
    echo -e "${YELLOW}正在检查系统环境...${NC}"
    
    CMD_INSTALL=""
    CMD_UPDATE=""
    
    # 检测是否为 Termux
    if [ -n "$TERMUX_VERSION" ]; then
        echo -e "${GREEN}识别到 Termux 环境。${NC}"
        CMD_INSTALL="pkg install -y"
        CMD_UPDATE="pkg update -y"
        # Termux 通常不需要 sudo，除非安装了 tsu，但常规包管理直接用 pkg
    
    # 检测 MacOS
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${GREEN}识别到 MacOS 环境。${NC}"
        if command -v brew &> /dev/null; then
            CMD_INSTALL="brew install"
        else
            echo -e "${RED}未找到 Homebrew，请先安装 Homebrew。${NC}"
            exit 1
        fi
        
    # 检测 Linux 发行版
    elif command -v apt-get &> /dev/null; then
        CMD_INSTALL="sudo apt-get install -y"
        CMD_UPDATE="sudo apt-get update"
    elif command -v yum &> /dev/null; then
        CMD_INSTALL="sudo yum install -y"
    elif command -v dnf &> /dev/null; then
        CMD_INSTALL="sudo dnf install -y"
    elif command -v pacman &> /dev/null; then
        CMD_INSTALL="sudo pacman -S --noconfirm"
    else
        echo -e "${RED}未找到支持的包管理器 (pkg/apt/yum/dnf/pacman/brew)。${NC}"
        exit 1
    fi

    # 更新源 (仅 Termux 和 Debian/Ubuntu)
    if [ ! -z "$CMD_UPDATE" ]; then
        echo -e "正在更新软件源..."
        $CMD_UPDATE
    fi

    # 安装 git, zsh, curl, vim (vim用于防止sed有时需要编辑器环境)
    # Termux 基础包可能不含 vim/nano，建议安装一个编辑器
    DEPENDENCIES="git zsh curl"
    
    for pkg in $DEPENDENCIES; do
        if ! command -v $pkg &> /dev/null; then
            echo -e "${YELLOW}正在安装 $pkg ...${NC}"
            $CMD_INSTALL $pkg
        else
            echo -e "${GREEN}$pkg 已安装。${NC}"
        fi
    done
}

install_packages

# 2. 安装 Oh My Zsh
# -----------------------------------------------------------
if [ -d "$OMZ_DIR" ]; then
    echo -e "${GREEN}Oh My Zsh 已安装，跳过下载。${NC}"
else
    echo -e "${YELLOW}正在安装 Oh My Zsh...${NC}"
    # 使用官方脚本
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    
    if [ ! -d "$OMZ_DIR" ]; then
        echo -e "${RED}Oh My Zsh 安装失败，请检查网络 (Termux可能需要开启VPN)。${NC}"
        exit 1
    fi
fi

# 3. 下载插件
# -----------------------------------------------------------
install_plugin() {
    local name=$1
    local repo=$2
    local target_dir="$ZSH_CUSTOM/plugins/$name"

    if [ ! -d "$target_dir" ]; then
        echo -e "${YELLOW}正在下载插件: $name ...${NC}"
        git clone --depth=1 $repo $target_dir
    else
        echo -e "${GREEN}插件 $name 已存在。${NC}"
    fi
}

install_plugin "zsh-autosuggestions" "https://github.com/zsh-users/zsh-autosuggestions"
install_plugin "zsh-syntax-highlighting" "https://github.com/zsh-users/zsh-syntax-highlighting.git"

# 4. 下载 Powerlevel10k 主题
# -----------------------------------------------------------
THEME_DIR="$ZSH_CUSTOM/themes/powerlevel10k"
if [ ! -d "$THEME_DIR" ]; then
    echo -e "${YELLOW}正在下载主题: Powerlevel10k ...${NC}"
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git $THEME_DIR
else
    echo -e "${GREEN}主题 Powerlevel10k 已存在。${NC}"
fi

# 5. 配置 .zshrc
# -----------------------------------------------------------
echo -e "${YELLOW}正在配置 .zshrc ...${NC}"

# 备份
if [ -f "$ZSHRC" ]; then
    cp "$ZSHRC" "${ZSHRC}.backup.$(date +%Y%m%d%H%M%S)"
fi

# 覆盖模板
cp "$OMZ_DIR/templates/zshrc.zsh-template" "$ZSHRC"

# 设置 sed 命令 (MacOS 需要 -i '')
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_CMD="sed -i ''"
else
    SED_CMD="sed -i" # Linux & Termux
fi

# 修改主题
$SED_CMD 's/^ZSH_THEME=".*"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' "$ZSHRC"

# 修改插件
# 注意：Termux 环境下 docker 插件通常无效，但保留也不会报错
NEW_PLUGINS="plugins=(git z extract web-search zsh-autosuggestions zsh-syntax-highlighting)"

# 如果不是 MacOS 且不是 Termux (即普通 Linux 服务器)，加上 sudo 插件
# Termux 默认没 sudo，加上会报错；MacOS 也不常用 sudo 插件
if [[ -z "$TERMUX_VERSION" ]] && [[ "$OSTYPE" != "darwin"* ]]; then
     NEW_PLUGINS="plugins=(git z extract sudo web-search docker zsh-autosuggestions zsh-syntax-highlighting)"
fi

$SED_CMD "s/^plugins=(git)/$NEW_PLUGINS/" "$ZSHRC"

# 兜底检查
if ! grep -q "zsh-autosuggestions" "$ZSHRC"; then
    echo -e "\n$NEW_PLUGINS" >> "$ZSHRC"
fi

echo -e "${GREEN}.zshrc 配置完成！${NC}"

# 6. 设置默认 Shell
# -----------------------------------------------------------
CURRENT_SHELL=$(basename "$SHELL")
if [ "$CURRENT_SHELL" != "zsh" ]; then
    echo -e "${YELLOW}正在设置 zsh 为默认 Shell...${NC}"
    if command -v chsh &> /dev/null; then
        chsh -s $(which zsh)
    else
        echo -e "${RED}无法自动修改默认 Shell，请手动执行: chsh -s zsh${NC}"
    fi
else
    echo -e "${GREEN}当前 Shell 已经是 Zsh。${NC}"
fi

# 7. 结束
# -----------------------------------------------------------
echo -e "\n${BLUE}==============================================${NC}"
echo -e "${GREEN}  Termux / Linux / Mac 环境安装完成！${NC}"
echo -e "${BLUE}==============================================${NC}"
echo -e "请执行: ${YELLOW}exec zsh${NC} 或重启 Termux 以生效。"
echo -e "Powerlevel10k 配置向导将在首次进入 zsh 时启动。"
echo -e "\n💡 Termux 提示: 如果图标显示乱码，请长按 Termux 界面 -> More -> Style -> Font"
echo -e "   并选择一款 Nerd Font (推荐 MesloLGS NF)。"
echo -e "==============================================\n"
```

这个脚本是ai写的。不过这不重要，好用就行。

理论上这段脚本具有一定的健壮性，对于多数的linux发行版、macos，乃至termux，这段脚本按理都可以正常工作，不过我没测试过。

---

### 如何使用?

1. 将这个脚本写入`xx.sh`中。xx可以是任意名字，比如`install_zsh.sh`。
2. `chmod +x xx.sh`，输入这个指令为脚本赋权。
3. 在当前目录下运行这个.sh文件，即输入`./xx.sh`。脚本就开始工作了。
