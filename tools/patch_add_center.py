import sys, re
p = sys.argv[1]
txt = open(p, 'r', encoding='utf-8').read()

# Ensure center_root var exists near other option vars
if re.search(r'(?m)^\\s*center_root\\s*=\\s*(True|False)\\s*$', txt) is None:
    # insert after first scale assignment if possible
    m = re.search(r'(?m)^(\\s*scale\\s*=\\s*[^\\r\\n]+)\\s*$', txt)
    if m:
        ins = m.end(0)
    else:

# Insert parse block inside args loop: after --gltf parsing block if present, else after first 'if a in (' block
])

if '--center-root' not in txt and '--center' not in txt:
    # try to find the args parsing loop 'while i < len(args):' then insert after first occurrence of --gltf handler
    mloop = re.search(r'(?ms)^\\s*while\\s+i\\s*<\\s*len\\(args\\)\\s*:\\s*\\r?\\n', txt)
    if not mloop:
        raise SystemExit('Could not find args parsing loop (while i < len(args))')
    anchor = re.search(r'(?ms)(^\\s*if\\s+.*--gltf.*\\r?\\n(?:^\\s+.*\\r?\\n){0,8})', txt[mloop.end():])
    if anchor:
        astart = mloop.end() + anchor.end(0)
    else:
        # fallback: insert right after loop start line
        astart = mloop.end()
        txt = txt[:astart] + parse_block + txt[astart:]

# Root creation: patch the __VFX_ROOT__ append line to apply translation when center_root
# We replace a single append dict with a small block that keeps indentation consistent
m = root_pat.search(txt)
if not m:
    # also allow double quotes variant
    m = root_pat.search(txt)
if not m:
    raise SystemExit('Could not find __VFX_ROOT__ append line to patch')

indent = m.group(1)
])

txt = txt[:m.start()] + block + txt[m.end():]

open(p, 'w', encoding='utf-8').write(txt)
print('Patched:', p)
