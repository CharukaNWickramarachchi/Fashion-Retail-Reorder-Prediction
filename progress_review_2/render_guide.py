"""Minimal dependency-free renderer for the included plain-language Markdown guides."""
import html
import re
from pathlib import Path
from src.inventory_report import CSS
ROOT=Path(__file__).resolve().parent

def inline(text):
    text=html.escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    return text

def render(text):
    out=[]; paragraph=[]; list_kind=None; code=None
    def flush():
        if paragraph: out.append('<p>'+inline(' '.join(paragraph))+'</p>'); paragraph.clear()
    def end_list():
        nonlocal list_kind
        if list_kind: out.append('</'+list_kind+'>'); list_kind=None
    for line in text.splitlines()+['']:
        if line.startswith('```'):
            flush(); end_list()
            if code is None: code=[]
            else: out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>'); code=None
            continue
        if code is not None: code.append(line); continue
        if not line.strip(): flush(); end_list(); continue
        if line.startswith('#'):
            flush(); end_list(); n=len(line)-len(line.lstrip('#')); out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>'); continue
        m=re.match(r'^(?:([-]) |(\d+)\. )(.*)',line)
        if m:
            flush(); kind='ul' if m[1] else 'ol'
            if list_kind!=kind: end_list(); out.append('<'+kind+'>'); list_kind=kind
            out.append('<li>'+inline(m[3])+'</li>')
        else: end_list(); paragraph.append(line)
    return '\n'.join(out)

if __name__=='__main__':
    for name in ['GROUP_GUIDE','WINDOWS_STEPS']:
        source=ROOT/'docs'/f'{name}.md'
        body=render(source.read_text(encoding='utf-8'))
        page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+name.replace('_',' ')+'</title><style>'+CSS+'pre{overflow:auto;background:#eaf0f5;padding:16px}h2{border-top:1px solid #ccd8e3;padding-top:24px}@media print{h2{break-after:avoid}}</style><main><nav><a href="../results/STOCK_REVIEW.html">Stock demonstration</a><a href="GROUP_GUIDE.html">Group guide</a><a href="WINDOWS_STEPS.html">Windows steps</a></nav>'+body+'</main></html>'
        source.with_suffix('.html').write_text(page,encoding='utf-8')
    print('Group guide and Windows guide rendered.')
