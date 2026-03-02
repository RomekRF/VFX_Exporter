import struct, sys, math

def u32(b,o): return struct.unpack_from("<I",b,o)[0], o+4
def i32(b,o): return struct.unpack_from("<i",b,o)[0], o+4
def u8(b,o):  return b[o], o+1
def f32(b,o): return struct.unpack_from("<f",b,o)[0], o+4

def cstr(b,o):
    e=b.index(0,o)
    return b[o:e].decode("latin1","replace"), e+1

def vec3(b,o):
    x,o=f32(b,o); y,o=f32(b,o); z,o=f32(b,o)
    return (x,y,z), o

def quat(b,o):
    x,o=f32(b,o); y,o=f32(b,o); z,o=f32(b,o); w,o=f32(b,o)
    return (x,y,z,w), o

def flags(raw):
    return {
        "raw": raw,
        "facing": bool(raw & 0x1),
        "morph": bool(raw & 0x4),
        "dump_uvs": bool(raw & 0x100),
        "facing_rod": bool(raw & 0x800),
    }

def skip_face_vertices(b,o,n):
    for _ in range(n):
        o += 4+4+4+4
        naf,o = i32(b,o)
        o += naf*4
    return o

def skip_frames_v46(b,o,num_frames,num_verts,num_faces,fl,is_keyframed):
    for fi in range(num_frames):
        if fl["morph"] or fi==0:
            o += 12+12
            o += num_verts*6
            if (fl["facing"] or fl["facing_rod"]):
                o += 8
            if fl["facing_rod"] and fi==0:
                o += 12
            o += (3*num_faces)*8
        if (not fl["morph"]) and (not is_keyframed):
            o += 12+16+12
    return o

def read_vec3_kf(b,o):
    t,o = i32(b,o)
    v,o = vec3(b,o)
    it,o = vec3(b,o)
    ot,o = vec3(b,o)
    return (t,v,it,ot), o

def read_quat_kf(b,o):
    t,o = i32(b,o)
    q,o = quat(b,o)
    ten,o = f32(b,o); con,o = f32(b,o); bias,o = f32(b,o); ei,o = f32(b,o); eo,o = f32(b,o)
    return (t,q,ten,con,bias,ei,eo), o

def dump(path):
    b=open(path,"rb").read()
    if b[:4] != b"VSFX": raise SystemExit("Bad magic")
    ver = struct.unpack_from("<I", b, 4)[0]
    print(f"\n=== {path} ===")
    print(f"version=0x{ver:08X}")
    if ver != 0x00040006:
        print("NOTE: dumper expects v4.6 (0x40006).")
        return

    off = 128
    while off < len(b):
        st = b[off:off+4].decode("latin1","replace")
        size = struct.unpack_from("<I", b, off+4)[0]
        body = b[off+8:off+8+(size-4)]
        off += 4+size

        if st != "SFXO":  # mesh sections
            continue

        o=0
        name,o = cstr(body,o)
        parent,o = cstr(body,o)
        _,o = u8(body,o)          # save_parent
        nv,o = i32(body,o)
        nf,o = i32(body,o)
        o += nf*96                # faces (fixed size in v4.6)
        fps,o = i32(body,o)
        stime,o = f32(body,o); etime,o = f32(body,o); nframes,o = i32(body,o)
        nm,o = i32(body,o)
        o += nm*4                 # material indices
        bc,o = vec3(body,o); br,o = f32(body,o)
        fr,o = u32(body,o)
        fl = flags(fr)
        nfv,o = i32(body,o)
        o = skip_face_vertices(body,o,nfv)
        ik,o = u8(body,o)
        is_keyframed = (ik != 0)

        o = skip_frames_v46(body,o,nframes,nv,nf,fl,is_keyframed)

        print(f"\n[MESH] {name}  parent='{parent}'  morph={fl['morph']}  keyframed={is_keyframed}")
        if not is_keyframed:
            continue

        piv_t,o = vec3(body,o)
        piv_r,o = quat(body,o)
        piv_s,o = vec3(body,o)

        nt,o = i32(body,o)
        t0 = None
        for i in range(nt):
            k,o = read_vec3_kf(body,o)
            if i==0: t0=k

        nr,o = i32(body,o)
        r0 = None
        for i in range(nr):
            k,o = read_quat_kf(body,o)
            if i==0: r0=k

        ns,o = i32(body,o)
        s0 = None
        for i in range(ns):
            k,o = read_vec3_kf(body,o)
            if i==0: s0=k

        sc = None
        if piv_s and s0:
            sc = (piv_s[0]*s0[1][0], piv_s[1]*s0[1][1], piv_s[2]*s0[1][2])

        print(f"  pivot_translation = {piv_t}")
        print(f"  pivot_scale       = {piv_s}")
        print(f"  key0_translation  = {t0[1] if t0 else None}")
        print(f"  key0_scale        = {s0[1] if s0 else None}")
        print(f"  effective_scale   = {sc}")

if __name__=="__main__":
    for p in sys.argv[1:]:
        dump(p)
