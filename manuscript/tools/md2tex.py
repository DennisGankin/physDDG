#!/usr/bin/env python3
"""Regenerate manuscript/supplementary.tex from manuscript/supplementary.md.

The markdown file is the source of truth; run this after editing it:
    python3 manuscript/tools/md2tex.py
"""
import re, sys

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
src = open('supplementary.md').read()
lines = src.split('\n')
out = []

def esc(t):
    # protect inline code first
    parts = re.split(r'(`[^`]*`)', t)
    res = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            body = p[1:-1].replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
            res.append('\\texttt{' + body + '}')
            continue
        p = p.replace('\\', '\\textbackslash{}')
        p = p.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#').replace('_', '\\_')
        p = p.replace('Å', '\\AA{}').replace('−', '$-$').replace('–', '--').replace('—', '---')
        p = p.replace('±', '$\\pm$').replace('→', '$\\rightarrow$').replace('²', '$^2$')
        p = p.replace('Δ', '$\\Delta$').replace('½', '$\\tfrac{1}{2}$').replace('…', '\\ldots{}')
        p = p.replace('ε', '$\\varepsilon$').replace('α', '$\\alpha$').replace('β', '$\\beta$')
        p = p.replace('×', '$\\times$').replace('≥', '$\\geq$').replace('≤', '$\\leq$').replace('·', '$\\cdot$')
        p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\emph{\1}', p)
        p = p.replace('|', '$|$')
        res.append(p)
    return ''.join(res)

i = 0
intable = False
while i < len(lines):
    L = lines[i]
    if L.startswith('|'):
        # collect table
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
            i += 1
        header = rows[0]
        body = [r for r in rows[2:]]
        n = len(header)
        widths = []
        for c in range(n):
            w = max([len(r[c]) if c < len(r) else 0 for r in rows])
            widths.append(min(max(w, 6), 110))
        tot = sum(widths)
        if tot > 95:
            scale = 0.90 if n <= 5 else 0.84
            frac = [scale * w / tot for w in widths]
            colspec = ''.join('>{\\raggedright\\arraybackslash}p{%.3f\\linewidth}' % f for f in frac)
        else:
            colspec = 'l' * n
        out.append('\\begin{center}\\scriptsize')
        out.append('\\begin{tabular}{@{}' + colspec + '@{}}')
        out.append('\\toprule')
        out.append(' & '.join(esc(c) for c in header) + ' \\\\')
        out.append('\\midrule')
        for r in body:
            r = (r + [''] * n)[:n]
            out.append(' & '.join(esc(c) for c in r) + ' \\\\')
        out.append('\\bottomrule\n\\end{tabular}\\end{center}')
        continue
    if L.startswith('# '):
        out.append('')
    elif L.startswith('## '):
        out.append('\\section{' + esc(L[3:]) + '}')
    elif L.startswith('### '):
        out.append('\\subsection*{' + esc(L[4:]) + '}')
    elif L.strip() == '---':
        out.append('')
    elif L.startswith('- '):
        if not intable:
            out.append('\\begin{itemize}')
            intable = True
        out.append('\\item ' + esc(L[2:]))
    elif re.match(r'^\d+\. ', L):
        out.append('\\par\\noindent ' + esc(L))
    else:
        if intable and not L.startswith('  '):
            out.append('\\end{itemize}')
            intable = False
        out.append(esc(L))
    i += 1
if intable:
    out.append('\\end{itemize}')
body = '\n'.join(out)
HEADER = open('tools/supplementary_preamble.tex').read()
open('supplementary.tex', 'w').write(HEADER + body + '\n\\end{document}\n')
print('wrote supplementary.tex,', len(out), 'lines of body')
