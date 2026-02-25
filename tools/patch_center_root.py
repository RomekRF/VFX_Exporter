import sys
p = sys.argv[1]
lines = open(p, 'r', encoding='utf-8').read().splitlines(True)  # keep line endings

# --- insert center_root var near first 'scale =' assignment ---
if not any(l.lstrip().startswith('center_root =') for l in lines):
    for i,l in enumerate(lines):
        if l.lstrip().startswith('scale ='):
            indent = l[:len(l)-len(l.lstrip())]
            break

# --- insert --center parsing right after args-loop start (while i < len(args):) ---
    for i,l in enumerate(lines):
        if l.lstrip().startswith('while i < len(args):'):
            indent = l[:len(l)-len(l.lstrip())] + '    '
            blk = [
            ]
            for j,ln in enumerate(blk):
                lines.insert(i+1+j, ln)
            break

# --- replace the __VFX_ROOT__ append line with a multi-line block ---
for i,l in enumerate(lines):
        indent = l[:len(l)-len(l.lstrip())]
        blk = [
        ]
        lines[i:i+1] = blk
        break

open(p, 'w', encoding='utf-8', newline='').writelines(lines)
print('Patched OK:', p)
