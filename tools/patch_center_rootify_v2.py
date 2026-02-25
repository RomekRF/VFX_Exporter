import sys
p = sys.argv[1]
lines = open(p, 'r', encoding='utf-8').read().splitlines(True)

def find_first(pred):
    for i,l in enumerate(lines):
        if pred(l): return i
    return None

# 1) Ensure center_root exists (insert after first 'scale =')
if not any(l.lstrip().startswith('center_root =') for l in lines):
    for i,l in enumerate(lines):
        if l.lstrip().startswith('scale ='):
            indent = l[:len(l)-len(l.lstrip())]
            lines.insert(i+1, indent + 'center_root = False  # --center\\n')
            break

# 2) Insert --center parsing right after the first 'a = args[i]' inside the args loop
need_center = True
for l in lines:
        need_center = False
        break
if need_center:
    ai = find_first(lambda s: s.lstrip().startswith('a = args[i]'))
    if ai is None:
    indent = lines[ai][:len(lines[ai]) - len(lines[ai].lstrip())]
    blk = [
    ]
    lines[ai+1:ai+1] = blk

# 3) Fix scene_nodes placement:
#    - remove any 'scene_nodes = []' that are indented inside the node loop
new_lines = []
for l in lines:
    if l.lstrip().startswith('scene_nodes = []'):
        # drop it; we'll reinsert in the right place
        continue
    new_lines.append(l)
lines = new_lines

#    - insert 'scene_nodes = []' right after gltf scenes init (or right before first scene_nodes.append)
if si is None:
if si is None:
    raise SystemExit('Could not find gltf init to anchor scene_nodes')
indent = lines[si][:len(lines[si]) - len(lines[si].lstrip())]
lines.insert(si+1, indent + 'scene_nodes = []\\n')

#    - ensure 'scene_nodes.append(node_i)' exists after node creation
if appi is not None:
    ind = lines[appi][:len(lines[appi]) - len(lines[appi].lstrip())]
    # insert only if next few lines don't already append node_i
    already = False
    for j in range(appi, min(appi+8, len(lines))):
        if 'scene_nodes.append(node_i)' in lines[j]:
            already = True
            break
    if not already:
        lines.insert(appi+1, ind + 'scene_nodes.append(node_i)\\n')

# 4) Replace the broken commented rootify line with real code
ri = find_first(lambda s: s.lstrip().startswith('# Rootify:'))
if ri is None:
    raise SystemExit('Could not find \
indent = lines[ri][:len(lines[ri]) - len(lines[ri].lstrip())]
root_blk = [
    indent + \
]
lines[ri:ri+1] = root_blk

open(p, 'w', encoding='utf-8', newline='').writelines(lines)
print('Patched OK')
