import sys, re

p = sys.argv[1]
text = open(p, "r", encoding="utf-8").read().splitlines(True)

def indent(s: str) -> str:
    return re.match(r"^\s*", s).group(0)

def is_blank(s: str) -> bool:
    return s.strip() == ""

def needs_fix(i: int) -> bool:
    # if-line has extra stuff after ":" -> one-liner (bad)
    m = re.match(r"^(\s*)if\s+center_geom\s*:\s*(.*)$", text[i])
    if not m:
        return False
    tail = m.group(2).rstrip("\r\n")
    if tail != "" and not tail.startswith("#"):
        return True
    # find next non-blank line; if not indented deeper -> bad
    j = i + 1
    while j < len(text) and is_blank(text[j]):
        j += 1
    if j >= len(text):
        return True
    return len(indent(text[j])) <= len(indent(text[i]))

def build_block(if_indent: str):
    i1 = if_indent + "    "
    i2 = if_indent + "        "
    return [
        if_indent + "if center_geom:\n",
        i1 + "# Recenter POSITION accessor 0 (float32 VEC3) by editing bin_blob in-place\n",
        i1 + "import struct\n",
        i1 + "try:\n",
        i2 + "acc0 = gltf['accessors'][0]\n",
        i2 + "bv0  = gltf['bufferViews'][acc0['bufferView']]\n",
        i2 + "base0 = int(bv0.get('byteOffset', 0)) + int(acc0.get('byteOffset', 0))\n",
        i2 + "cnt0  = int(acc0['count'])\n",
        i2 + "minx=miny=minz=1e30; maxx=maxy=maxz=-1e30\n",
        i2 + "for ii in range(cnt0):\n",
        i2 + "    o = base0 + ii*12\n",
        i2 + "    x,y,z = struct.unpack_from('<fff', bin_blob, o)\n",
        i2 + "    if x<minx: minx=x\n",
        i2 + "    if y<miny: miny=y\n",
        i2 + "    if z<minz: minz=z\n",
        i2 + "    if x>maxx: maxx=x\n",
        i2 + "    if y>maxy: maxy=y\n",
        i2 + "    if z>maxz: maxz=z\n",
        i2 + "cx=(minx+maxx)*0.5; cy=(miny+maxy)*0.5; cz=(minz+maxz)*0.5\n",
        i2 + "for ii in range(cnt0):\n",
        i2 + "    o = base0 + ii*12\n",
        i2 + "    x,y,z = struct.unpack_from('<fff', bin_blob, o)\n",
        i2 + "    struct.pack_into('<fff', bin_blob, o, x-cx, y-cy, z-cz)\n",
        i2 + "if 'min' in acc0 and 'max' in acc0 and acc0.get('type') == 'VEC3':\n",
        i2 + "    acc0['min'] = [float(acc0['min'][0]) - cx, float(acc0['min'][1]) - cy, float(acc0['min'][2]) - cz]\n",
        i2 + "    acc0['max'] = [float(acc0['max'][0]) - cx, float(acc0['max'][1]) - cy, float(acc0['max'][2]) - cz]\n",
        i1 + "except Exception as e:\n",
        i2 + "print('[WARN] center-geom failed:', e)\n",
        "\n",
    ]

fixed = False
i = 0
while i < len(text):
    m = re.match(r"^(\s*)if\s+center_geom\s*:", text[i])
    if m and needs_fix(i):
        blk = build_block(m.group(1))
        # Replace ONLY the broken if-line; leave whatever follows (we're only fixing indentation errors)
        text[i:i+1] = blk
        fixed = True
        i += len(blk)
        continue
    i += 1

if not fixed:
    print("No broken 'if center_geom:' block found to fix.")
else:
    open(p, "w", encoding="utf-8", newline="").writelines(text)
    print("Fixed broken 'if center_geom:' indentation blocks.")
