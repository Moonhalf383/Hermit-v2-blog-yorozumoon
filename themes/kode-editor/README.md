# Kode Editor

一个白底黑字、文件优先、支持键盘操作的 Hugo 主题。英文使用本地托管的 [Kode Mono](https://kodemono.com/)，中文依次回退到 Noto Sans Mono CJK SC、Sarasa Mono SC、霞鹜文楷等宽版和系统等宽字体。

## 内容模型

主题不会要求移动 Hugo 的物理文件。页面包会自动折叠成一个虚拟文件：

```text
content/posts/Linux/index.md  ->  posts/Linux.md
content/links.md              ->  links.md
content/_index.md             ->  README.md
```

可以在任意 Markdown 的 frontmatter 中覆盖虚拟位置：

```yaml
---
title: Linux 笔记
virtualPath: notes/system/linux.md # .md 可省略
icon: LiTerminal
iconColor: "#000000"
folderIcon: LiFolderOpen
---
```

`virtualPath` 只影响左侧树和编辑器标签，不改变 Hugo URL、图片相对路径或磁盘位置。任何路径分段以 `.` 开头的物理或虚拟目录都不会出现在文件树中。

`icon` 兼容 emoji、短文本，以及 Iconize 常见的 `LiHouse`、`LiFolder`、`LiFileCode`、`LiBook`、`LiImage`、`LiGithub`、`LiLink`、`LiTag` 等语义名称。为控制体积，主题不加载完整图标库，而是把这些名称映射为内置线框 SVG（不依赖字体字形）；未知 `Li*` 名称回退为文件/目录 SVG。

左栏提供 `FILES` 和 `TAGS` 两个视图。标签直接读取已有的 `tags` frontmatter，不需要复制内容或生成另一套文件。目录保留字母序；同目录和同标签下的文件按 `date` 从新到旧排序，同日期按文件名排序，无日期排末，首页 `README.md` 始终置顶。

现有文章均已按主题配置黑色图标（包括草稿）。额外内置 `LiCpu`、`LiBrain`、`LiChart`、`LiNotebook`、`LiGamepad`、`LiMusic`、`LiPen`、`LiBox`、`LiFlask`、`LiCloud` 等 SVG，不增加字体或外部图标依赖。

## Obsidian 语法

标准 Markdown 保持由 Hugo/Goldmark 渲染。正文加载后，主题只扫描代码块之外的文本节点，并支持：

```md
[[文件名]]
[[文件名|显示文字]]
[[文件名#标题]]
![[image.png]]
![[image.png|640]]
![[image.png|640x360]]
```

普通 wikilink 会根据标题、虚拟路径、文件名和 aliases 查找页面。图片默认相对于当前 page bundle 解析。这样不会误改代码块里的 shell、Lua 或数组语法。

## Mermaid 与公式

`mermaid` 和 `merm` 两种 fenced code block 都受支持：

````md
```merm
graph LR
  A[Write] --> B[Build]
```
````

Mermaid 只在页面存在图表时动态加载。MathJax 只在页面 frontmatter 含 `math: true` 时加载；行内和块公式外围会以浅灰色显示 `$` 与 `$$`。

## 键位

| 键位 | 动作 |
| --- | --- |
| `j` / `k` | 正文按浏览器视觉行移动并保持列，代码包含空行；侧栏逐项移动 |
| `w` / `e` / `b` | 下一个词首 / 词尾 / 上一个词首 |
| `0` / `$` | 当前视觉行首 / 尾 |
| `gg` / `G` | 文档首 / 尾 |
| `h` / `l` | 正文中逐字符移动；侧栏中收起、展开或打开 |
| `Ctrl-h` / `Ctrl-l` | 在文件树、正文、大纲间切换 |
| `H` / `L` | 收放左、右侧栏 |
| `v` / `V` | 可视选择 / 整行选择 |
| `y` / `yy` | 复制选择 / 当前视觉行（代码中为整个代码块） |
| `gd` | 打开光标下的 wikilink / 当前友链 |
| `?` | 打开帮助 |
| `Esc` | 返回 NORMAL 或关闭帮助 |

鼠标、触摸、原生链接和各栏独立滚动保持可用。布局和展开状态保存在浏览器 localStorage 中，并在样式加载之前恢复，避免换页闪动。

正文光标是 `mix-blend-mode: difference` 的空矩形，不复制字符、不插入隐藏 span；只反转其覆盖区域的颜色。文字几何信息按文本块惰性缓存，在字体、宽度、内容或图片布局改变后重新测量。代码使用 Kode Mono、无边框背景，长行横向滚动而非强制折行。英文标题不在单词内部强制断开。`w/e/b` 使用 Vim 的空白、字母数字下划线、标点三类词边界（不额外做中文语义分词）。

友链每张卡片独占一行，作为一个完整导航项：`j/k` 在卡片间移动，`h/l` 不钻入卡片文字，光标仅反转左上角的 10px 小方块。`gd` 打开该友链（沿用新标签页策略）；原生鼠标点击、Tab 和 Enter 仍然可用。

## 评论与站点图标

`posts` 默认启用旧站 giscus；用 `comments: false` 禁用，其他页面可设 `comments: true`。配置在 `layouts/_partials/editor-comments.html`：保留原仓库和 Announcements 分类，以 `specific + .Title` 显式复用旧站的标题映射，避免新标签页标题的站名后缀创建另一条讨论。评论区进入视口附近才加载 giscus，配色为 light。

`static/images/face.png` 是 `~/.face` 的发布副本；构建不依赖访问用户的主目录。

## 回归测试

需要 Hugo、Firefox、Node.js。测试自动建立并清理临时站点，不写项目 `public/`：

```sh
npm install --prefix /tmp/kode-browser-tests puppeteer-core@25.12.0
PUPPETEER_PATH=/tmp/kode-browser-tests/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js \
  node themes/kode-editor/tests/browser.mjs
```

用例覆盖视觉行（段落、代码空行、表格）、词边界、Unicode 字素、visual、首行滚动、侧栏状态/对齐、标题换行、黑色图标、文件/标签日期排序、友链逐卡片移动与 `gd` 新标签页跳转，以及评论加载配置。测试中的 giscus 网络响应使用桩，不验证实际 GitHub 登录、讨论历史或发表评论。

## 字体许可

`static/fonts/kode-mono-variable.woff2` 来自 Kode Mono 项目，使用 SIL Open Font License 1.1；许可全文见 `static/fonts/OFL.txt`。
