// Run: PUPPETEER_PATH=/tmp/kode-browser-tests/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js node themes/kode-editor/tests/browser.mjs
// Requires Hugo and Firefox. Test content/build/server are isolated in /tmp.
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, writeFile, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, dirname, join, extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {createServer} from 'node:http';
const {default: puppeteer} = await import(process.env.PUPPETEER_PATH || 'puppeteer-core');
const theme = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const temp = await mkdtemp(join(tmpdir(), 'kode-regression-'));
await mkdir(join(temp, 'content/posts'), {recursive:true});
await writeFile(join(temp, 'hugo.toml'), `baseURL = 'http://localhost/'\ntheme = 'kode-editor'\ndisableKinds = ['taxonomy','term','RSS','sitemap']\n[markup.highlight]\nnoClasses = false\n`);
await writeFile(join(temp,'content/_index.md'), '---\ntitle: README.md\nicon: LiHouse\n---\n\n[[Visual]]\n');
const paragraph = 'Render geometry follows actual browser wrapping, not paragraph boundaries. '.repeat(10);
const code = 'alpha beta_gamma delta\nsecond row here\n\nfourth line\n中文 👩‍💻 emoji\n' + 'long_line_'.repeat(35) + '\nlast line';
await writeFile(join(temp,'content/posts/visual.md'), `---\ntitle: Visual Movement International Typography Example\nicon: LiTerminal\ntags: [testing]\n---\n\n${paragraph}\n\nalpha beta_gamma delta\n\nalpha\n\nbeta\n\n| Left | Right |\n| --- | --- |\n| ${'wrap words '.repeat(30)} | ${'more text '.repeat(22)} |\n\n## Multiple Line Heading With International Words\n\n\`\`\`python\n${code}\n\`\`\`\n\n${paragraph}\n`);
// Date fixtures deliberately disagree with alphabetic order; include ties and no date.
for (const [name,date,icon] of [['Z-old','1960-01-01','LiCpu'],['B-new','2024-01-01','LiBrain'],['A-new','2024-01-01','LiNotebook'],['C-undated','', 'LiMusic']]) {
  await writeFile(join(temp,`content/posts/${name}.md`),`---\ntitle: ${name}\n${date ? `date: ${date}\n` : ''}virtualPath: notes/${name}.md\ntags: [ordering]\nicon: ${icon}\niconColor: '#000000'\n---\n\nDate test.\n`);
}
await mkdir(join(temp,'layouts/shortcodes'),{recursive:true});
await mkdir(join(temp,'data'),{recursive:true});
await writeFile(join(temp,'layouts/shortcodes/friendlinks.html'),await readFile(resolve(theme,'../../layouts/shortcodes/friendlinks.html')));
await writeFile(join(temp,'data/links.json'),JSON.stringify({list:[
  {name:'First friend',desc:'First description',url:'/friend-one/',avatar:'/images/face.png'},
  {name:'Second friend',desc:'A long description '.repeat(20),url:'/friend-two/',avatar:'/images/face.png'}
]}));
await writeFile(join(temp,'content/links.md'),'---\ntitle: Friends\n---\n\nIntroduction.\n\n{{< friendlinks >}}\n\nAfter friends.\n');
await writeFile(join(temp,'content/friend-one.md'),'---\ntitle: Friend destination\n---\n\nDestination.');
execFileSync('hugo',['--source',temp,'--themesDir',dirname(theme),'--destination',join(temp,'public'),'--panicOnWarning'],{stdio:'pipe'});
const server = createServer(async (req,res) => {
  try {
    const url = new URL(req.url,'http://localhost');
    const path = resolve(temp,'public','.' + decodeURIComponent(url.pathname));
    assert(path.startsWith(join(temp,'public') + '/') || path === join(temp,'public'));
    const file = path.endsWith('/') || !extname(path) ? join(path,'index.html') : path;
    const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.woff2':'font/woff2'};
    res.setHeader('Content-Type',mime[extname(file)] || 'application/octet-stream');
    res.end(await readFile(file));
  } catch {res.statusCode=404;res.end('not found');}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser;
const errors=[];
const results=[];
const check=(label,value)=>{assert(value,label);results.push(label);};
try {
  browser=await puppeteer.launch({browser:'firefox',executablePath:process.env.FIREFOX || '/usr/bin/firefox',headless:true,extraPrefsFirefox:{'ui.prefersReducedMotion':1}});
  const page=await browser.newPage();
  await page.setViewport({width:1440,height:900});
  page.on('pageerror',e=>errors.push(e.message));
  await page.setRequestInterception(true);
  page.on('request',r=>r.url().startsWith(origin) ? r.continue() : r.respond({status:200,contentType:'application/javascript',headers:{'access-control-allow-origin':'*'},body:'/* external service stub: configuration/loading only */'}));
  const tick=()=>new Promise(r=>setTimeout(r,60));
  const key=async k=>{await page.keyboard.press(k);await tick();};
  const rect=()=>page.$eval('#vim-cursor',e=>({x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height,visible:e.classList.contains('visible'),text:e.textContent}));
  const glyph=async (selector,text,offset=0)=>page.evaluate(({selector,text,offset})=>{
    const parent=document.querySelector(selector),walker=document.createTreeWalker(parent,NodeFilter.SHOW_TEXT);
    while(walker.nextNode()) {
      const start=walker.currentNode.data.indexOf(text);
      if(start<0)continue;
      const range=document.createRange();range.setStart(walker.currentNode,start+offset);range.setEnd(walker.currentNode,start+offset+1);
      const r=range.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};
    }
    throw Error(`Missing text ${text}`);
  },{selector,text,offset});
  const clickGlyph=async(selector,text,offset=0)=>{
    await page.$eval(selector,e=>e.scrollIntoView({block:'center',behavior:'instant'}));await tick();
    const r=await glyph(selector,text,offset);await page.mouse.click(r.x+r.w/2,r.y+r.h/2);await tick();return r;
  };
  const aligned=(a,b)=>Math.abs(a.x-b.x)<2&&Math.abs(a.y-b.y)<2;
  await page.goto(origin+'/posts/visual/',{waitUntil:'networkidle0'});
  await page.evaluate(()=>document.fonts.ready);await tick();
  check('empty difference cursor',await page.$eval('#vim-cursor',e=>!e.textContent&&getComputedStyle(e).mixBlendMode==='difference'));
  check('SVG icons render',await page.$$eval('.file-icon svg',nodes=>nodes.length>=2));
  const sortedPaths=['A-new','B-new','Z-old','C-undated'].map(name=>`notes/${name}.md`);
  const treePaths=selector=>page.$eval(selector,e=>[...e.parentElement.querySelector(':scope>ul').querySelectorAll('a.tree-row')].map(a=>a.dataset.path));
  check('files date descending, ties by name, undated last',JSON.stringify(await treePaths('[data-path="notes"]'))===JSON.stringify(sortedPaths));
  check('README remains pinned',await page.$eval('#file-tree .tree-row',e=>e.dataset.path==='README.md'));
  check('assigned SVG icons black and distinct',await page.$$eval('[data-path^="notes/"] .file-icon',nodes=>nodes.length===4&&nodes.every(e=>getComputedStyle(e).color==='rgb(0, 0, 0)')&&new Set(nodes.map(e=>e.querySelector('path').getAttribute('d'))).size===4));
  await page.click('[data-explorer-view="tags"]');
  check('tag files use the same date ordering',JSON.stringify(await treePaths('[data-path="@tag/ordering"]'))===JSON.stringify(sortedPaths));
  await page.click('[data-explorer-view="files"]');
  await clickGlyph('.document-header h1','Visual');
  check('no duplicate collapse buttons',await page.$$eval('.pane-title[data-toggle-pane]',nodes=>nodes.length===0));
  check('tabs fit inside their borders',await page.$$eval('.explorer-tabs button',nodes=>nodes.every(e=>{const a=e.getBoundingClientRect(),b=e.parentElement.getBoundingClientRect();return a.top>=b.top&&a.bottom<=b.bottom&&a.left>=b.left&&a.right<=b.right;})));
  check('current file outline',await page.$eval('.tree-row.active',e=>getComputedStyle(e).boxShadow.includes('2px')));
  check('code typography/no border',await page.$eval('pre',e=>getComputedStyle(e).fontFamily.includes('Kode Mono')&&parseFloat(getComputedStyle(e).fontSize)>=14&&getComputedStyle(e).borderTopWidth==='0px'));

  await clickGlyph('.content>p','Render');
  const before=await rect();await key('j');const after=await rect();
  const lh=await page.$eval('.content>p',e=>parseFloat(getComputedStyle(e).lineHeight));
  check('wrapped paragraph j moves one visual row',Math.abs((after.y-before.y)-lh)<2);
  await key('k');check('wrapped paragraph k returns',aligned(await rect(),before));

  await clickGlyph('.content>p:nth-of-type(3)','alpha');await key('e');
  check('e stops at block boundary',aligned(await rect(),await glyph('.content>p:nth-of-type(3)','alpha',4)));
  await clickGlyph('.content>p:nth-of-type(4)','beta',2);await key('b');
  check('b stops at block start',aligned(await rect(),await glyph('.content>p:nth-of-type(4)','beta')));

  await clickGlyph('table','wrap');
  const tableRows=[];
  for(let i=0;i<4;i++) { tableRows.push((await rect()).y + await page.$eval('#document-pane',e=>e.scrollTop)); await key('j'); }
  check('wrapped table j is monotonically downwards',tableRows.every((y,i)=>!i || y>tableRows[i-1]));

  await clickGlyph('pre','alpha');const first=await rect();
  await key('j');const second=await rect();
  const codeLH=await page.$eval('pre',e=>parseFloat(getComputedStyle(e).lineHeight));
  check('code j advances only one source row',Math.abs(second.y-first.y-codeLH)<2);
  await key('j');const blank=await rect();
  check('code blank row is a stop',Math.abs(blank.y-second.y-codeLH)<2);
  await key('j');const fourth=await rect();
  check('code j after blank remains adjacent',Math.abs(fourth.y-blank.y-codeLH)<2);
  await key('k');check('code k returns to blank',aligned(await rect(),blank));

  await clickGlyph('pre','alpha');await key('w');
  check('w goes to next word',aligned(await rect(),await glyph('pre','beta_gamma')));
  await key('e');check('e goes to word end',aligned(await rect(),await glyph('pre','beta_gamma',9)));
  await key('b');check('b returns to word start',aligned(await rect(),await glyph('pre','beta_gamma')));
  await key('v');await key('l');
  check('visual hl selects exact characters',await page.evaluate(()=>getSelection().toString()==='be'));
  await key('Escape');await key('0');await key('V');
  check('V selects a visual row not whole code block',await page.evaluate(()=>getSelection().toString().startsWith('alpha beta_gamma delta')&&!getSelection().toString().includes('second')));
  await key('Escape');
  await clickGlyph('pre','fourth');await key('j');
  await key('0');await key('l');await key('l');await key('l'); // 中 -> 文 -> space -> emoji cluster
  await key('v');check('Unicode grapheme remains whole',await page.evaluate(()=>getSelection().toString()==='👩‍💻'));await key('Escape');

  await key('g');await key('g');await key('k');
  check('top boundary reveals title',await page.$eval('#document-pane',e=>e.scrollTop===0));
  check('title aligned with body',await page.evaluate(()=>Math.abs(document.querySelector('.document-header h1').getBoundingClientRect().left-document.querySelector('.content>p').getBoundingClientRect().left)<1));
  for(const side of ['H','L','H','L']) {
    await key(side);
    check('alignment after '+side+' '+results.length,await page.evaluate(()=>Math.abs(document.querySelector('.document-header h1').getBoundingClientRect().left-document.querySelector('.content>p').getBoundingClientRect().left)<1));
  }
  await clickGlyph('.content>p','Render');await key('H');
  const shifted=await rect();await key('j');
  check('j keeps column after sidebar reflow',Math.abs((await rect()).x-shifted.x)<12);
  await key('H');await key('L');
  await page.goto(origin+'/');
  check('collapsed outline persists across document load',await page.evaluate(()=>document.documentElement.classList.contains('right-collapsed')&&getComputedStyle(document.querySelector('#outline-pane')).opacity==='0'));
  await page.goto(origin+'/posts/visual/',{waitUntil:'networkidle0'});await tick();
  check('cursor matches new document title',aligned(await rect(),await glyph('.document-header h1','Visual')));
  await page.keyboard.down('Control');await page.keyboard.press('h');await page.keyboard.up('Control');await tick();await key('k');
  check('sidebar selected and active differentiated',await page.evaluate(()=>{
    const a=document.querySelector('.tree-row.active'),s=document.querySelector('#file-tree .selected');
    return a!==s&&getComputedStyle(s).backgroundColor==='rgb(17, 17, 17)'&&getComputedStyle(a).backgroundColor!=='rgb(17, 17, 17)';
  }));

  check('giscus preserves original title identity',await page.$eval('[data-giscus]',e=>e.dataset.mapping==='specific'&&e.dataset.term==='Visual Movement International Typography Example'&&e.dataset.repo==='Moonhalf383/Hermit-v2-blog-yorozumoon'&&e.dataset.theme==='light'));
  await page.$eval('#comments',e=>e.scrollIntoView({behavior:'instant'}));await tick();
  check('giscus script lazy-loads near comments',await page.$eval('[data-giscus]',e=>!!e.querySelector('script[src="https://giscus.app/client.js"]')));
  const icon=await page.$eval('link[rel=icon]',e=>e.href);
  check('face favicon served',(await fetch(icon)).status===200);

  await page.setViewport({width:390,height:844});await tick();
  await page.goto(origin+'/posts/visual/',{waitUntil:'networkidle0'});await tick();
  check('mobile title does not split English words',await page.$eval('.document-header h1',e=>{
    const node=e.firstChild;return [...node.data.matchAll(/\S+/g)].every(m=>{const r=document.createRange();r.setStart(node,m.index);r.setEnd(node,m.index+m[0].length);return r.getClientRects().length===1;});
  }));
  await page.setViewport({width:1440,height:900});
  await page.goto(origin+'/links/',{waitUntil:'networkidle0'});await tick();
  const cardState=()=>page.evaluate(()=>{
    const card=document.activeElement,rect=card.getBoundingClientRect(),c=document.querySelector('#vim-cursor').getBoundingClientRect();
    return {name:card.querySelector('.friend-name')?.textContent,corner:Math.abs(c.x-rect.x)<1&&Math.abs(c.y-rect.y)<1&&c.width===10&&c.height===10,background:getComputedStyle(card).backgroundColor};
  });
  check('friend cards form one column',await page.$$eval('.friend-card',cards=>cards.every((c,i)=>!i || c.getBoundingClientRect().top>=cards[i-1].getBoundingClientRect().bottom && Math.abs(c.getBoundingClientRect().left-cards[0].getBoundingClientRect().left)<1)));
  check('all friend cards remain native tab stops',await page.$$eval('.friend-card',cards=>cards.every(c=>c.tabIndex===0)));
  await key('j');await key('j');
  let card=await cardState();
  check('j reaches first friend with corner-only cursor',card.name==='First friend'&&card.corner&&card.background==='rgb(255, 255, 255)');
  await key('l');await key('h');check('hl keep card atomic',(await cardState()).name==='First friend'&&(await cardState()).corner);
  await key('j');check('j advances exactly one card',(await cardState()).name==='Second friend'&&(await cardState()).corner);
  await key('j');check('j exits cards into following text',await page.evaluate(()=>document.activeElement.textContent==='After friends.'));
  await key('k');await key('k');check('k returns through individual cards',(await cardState()).name==='First friend');
  await key('Tab');check('native Tab reaches next card',(await cardState()).name==='Second friend');
  await key('k');
  const popupReady=new Promise(resolve=>page.once('popup',resolve));
  await key('g');await key('d');
  const popup=await Promise.race([popupReady,new Promise((_,reject)=>setTimeout(()=>reject(Error('gd did not open friend')),5000))]);
  await popup.waitForFunction(()=>location.pathname==='/friend-one/');
  check('gd opens focused friend in new tab',new URL(popup.url()).pathname==='/friend-one/');
  check('friend destination has no opener',await popup.evaluate(()=>window.opener===null));
  await popup.close();
  if(errors.length) console.error(errors);
  check('no runtime errors',errors.length===0);
  console.log(JSON.stringify({passed:results.length,results,errors},null,2));
} finally {
  await browser?.close();
  await new Promise(r=>server.close(r));
  await rm(temp,{recursive:true,force:true});
}
