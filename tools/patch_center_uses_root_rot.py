import sys, re

p = sys.argv[1]
s = open(p, "r", encoding="utf-8").read()

# We patch the specific line that assigns root_node['translation'] from sx/sy/sz.
needle = "root_node['translation'] = [-sx / nmesh, -sy / nmesh, -sz / nmesh]"
if needle not in s:
    raise SystemExit("Could not find expected center assignment line to patch:\n" + needle)

# Insert helper q-rotate function once (near rootify block)
if "def _qrot(" not in s:
    # place it near the rootify block by finding the root_node creation line
    m = re.search(r"(?m)^\s*root_node\s*=\s*\{.*'__VFX_ROOT__'.*\}\s*$", s)
    if not m:
        raise SystemExit("Could not find root_node creation line to anchor qrot helper.")
    indent = re.match(r"^\s*", m.group(0)).group(0)

    helper = "\n".join([
        indent + "def _qmul(a,b):",
        indent + "    ax,ay,az,aw=a; bx,by,bz,bw=b",
        indent + "    return (",
        indent + "        aw*bx + ax*bw + ay*bz - az*by,",
        indent + "        aw*by - ax*bz + ay*bw + az*bx,",
        indent + "        aw*bz + ax*by - ay*bx + az*bw,",
        indent + "        aw*bw - ax*bx - ay*by - az*bz",
        indent + "    )",
        indent + "def _qconj(q):",
        indent + "    x,y,z,w=q; return (-x,-y,-z,w)",
        indent + "def _qrot(q, v3):  # q is glTF [x,y,z,w]",
        indent + "    vx,vy,vz = v3",
        indent + "    p = (vx,vy,vz,0.0)",
        indent + "    return _qmul(_qmul(q,p), _qconj(q))[:3]",
        ""
    ])

    s = s[:m.end()] + "\n" + helper + s[m.end():]

# Replace centering assignment with rotation-aware version
replacement = "\n".join([
    "            # center_root: translate root by -(R * avg(mesh_translation)) so world lands near origin",
    "            avg = (sx / nmesh, sy / nmesh, sz / nmesh)",
    "            q = root_node.get('rotation', [0.0,0.0,0.0,1.0])  # glTF [x,y,z,w]",
    "            vx,vy,vz = _qrot(tuple(q), avg)",
    "            root_node['translation'] = [-vx, -vy, -vz]",
])

s = s.replace(needle, replacement)

open(p, "w", encoding="utf-8", newline="").write(s)
print("Patched center_root to account for root rotation.")
