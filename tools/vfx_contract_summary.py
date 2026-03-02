import struct, sys, os

def i32(b,o): return struct.unpack_from("<i",b,o)[0], o+4
def u8(b,o):  return b[o], o+1

def cstr(b,o):
    e=b.index(0,o)
    return b[o:e].decode("latin1","replace"), e+1

def printable4cc(t): return len(t)==4 and all(0x20<=c<=0x7E for c in t)

def scan_start(b, max_scan=65536):
    n=min(len(b)-8, max_scan)
    for off in range(8, n, 4):
        t=b[off:off+4]
        if not printable4cc(t):
            continue
        size=struct.unpack_from("<I", b, off+4)[0]
        if size < 8 or off+4+size > len(b):
            continue
        cur=off
        ok=True
        for _ in range(6):
            if cur+8>len(b): break
            t2=b[cur:cur+4]
            if not printable4cc(t2): ok=False; break
            s2=struct.unpack_from("<I", b, cur+4)[0]
            if s2<8 or cur+4+s2>len(b): ok=False; break
            cur=cur+4+s2
        if ok:
            return off
    return 128

def sections(b):
    start=scan_start(b)
    out=[]
    off=start
    while off+8<=len(b):
        t=b[off:off+4]
        if not printable4cc(t): break
        size=struct.unpack_from("<I", b, off+4)[0]
        if size<8 or off+4+size>len(b): break
        out.append((off, t.decode("latin1"), size, b[off:off+4+size]))
        off=off+4+size
    return start, out, off

def sfxo_brief(sec_bytes):
    body=sec_bytes[8:]
    o=0
    name,o = cstr(body,o)
    parent,o = cstr(body,o)
    sp,o = u8(body,o)
    nv,o = i32(body,o)
    nf,o = i32(body,o)
    return name, parent, sp, nv, nf

def dmmy_names(sec_bytes):
    body=sec_bytes[8:]
    names=[]
    o=0
    while o < len(body):
        try:
            nm,o2 = cstr(body,o)
            if nm and all(32 <= ord(ch) <= 126 for ch in nm):
                names.append(nm)
            o=o2
        except ValueError:
            break
    out=[]
    for n in names:
        if n not in out:
            out.append(n)
    return out

def analyze(path):
    b=open(path,"rb").read()
    bn=os.path.basename(path)
    if b[:4]!=b"VSFX":
        print(f"\n{bn}: NOT VSFX")
        return
    ver=struct.unpack_from("<I", b, 4)[0]
    start, secs, end=sections(b)
    types=[t for _,t,_,_ in secs]
    print(f"\n=== {bn} ===")
    print(f"size={len(b)}  ver=0x{ver:08X}  sections_start={start}  parsed_end={end}")
    print("sections:", ", ".join(types) if types else "(none)")

    meshes=[]
    dummies=[]
    for _,t,_,sec in secs:
        if t=="SFXO":
            try:
                meshes.append(sfxo_brief(sec))
            except Exception:
                meshes.append(("<parse-failed>","",0,0,0))
        elif t=="DMMY":
            dummies = dmmy_names(sec)

    if meshes:
        print("meshes:")
        for (n,p,sp,nv,nf) in meshes:
            print(f"  - {n}  parent='{p}'  save_parent={sp}  verts={nv}  faces={nf}")
    else:
        print("meshes: (none)")

    if dummies:
        print("dummies:", ", ".join(dummies))
    else:
        print("dummies: (none)")

    mesh_names=set(m[0] for m in meshes if m[0])
    has_flagpole = "flagpole" in mesh_names
    has_flagmesh = "FlagMesh" in mesh_names
    has_prop = "$prop_flag" in (dummies or [])
    print("FLAG_CONTRACT:",
          f"flagpole={has_flagpole}",
          f"FlagMesh={has_flagmesh}",
          f"$prop_flag={has_prop}")

if __name__=="__main__":
    for p in sys.argv[1:]:
        analyze(p)
