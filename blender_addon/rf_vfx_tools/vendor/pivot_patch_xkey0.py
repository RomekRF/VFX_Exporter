import argparse, struct

def i32(b,o): return struct.unpack_from("<i",b,o)[0], o+4
def u32(b,o): return struct.unpack_from("<I",b,o)[0], o+4
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

def read_vec3_kf_value_only(b,o):
    _,o = i32(b,o)    # time
    v,o = vec3(b,o)   # value
    o += 12+12        # inTan + outTan
    return v, o

def read_quat_kf_value_only(b,o):
    _,o = i32(b,o)    # time
    q,o = quat(b,o)   # value
    o += 4*5          # ten/cont/bias/easeIn/easeOut
    return q, o

def keyframed_info_v46(vfx_bytes):
    b=vfx_bytes
    if b[:4] != b"VSFX":
        raise SystemExit("Bad magic (expected VSFX).")
    ver = struct.unpack_from("<I", b, 4)[0]
    if ver != 0x00040006:
        raise SystemExit(f"pivot_patch_xkey0 supports v4.6 only (0x00040006). Got 0x{ver:08X}")

    off = 128
    out = {}
    while off < len(b):
        stype, slen = struct.unpack_from("<II", b, off)
        body_off = off + 8
        body_len = slen - 4
        sec_end  = body_off + body_len
        if sec_end > len(b) or slen < 8:
            break

        if stype == 0x4F584653:  # 'SFXO'
            body = b[body_off:sec_end]
            o=0
            name,o = cstr(body,o)
            parent,o = cstr(body,o)
            _,o = u8(body,o)
            nv,o = i32(body,o)
            nf,o = i32(body,o)
            o += nf*96
            _,o = i32(body,o)                    # fps
            _,o = f32(body,o); _,o = f32(body,o); nframes,o = i32(body,o)
            nm,o = i32(body,o)
            o += nm*4
            _,o = vec3(body,o); _,o = f32(body,o)
            fr,o = u32(body,o)
            fl = flags(fr)
            nfv,o = i32(body,o)
            o = skip_face_vertices(body,o,nfv)
            ik,o = u8(body,o)
            is_keyframed = (ik != 0)

            o = skip_frames_v46(body,o,nframes,nv,nf,fl,is_keyframed)

            if is_keyframed:
                pivot_abs = body_off + o
                piv_t,o2 = vec3(body,o)
                piv_r,o2 = quat(body,o2)
                piv_s_abs = body_off + o2
                piv_s,o2 = vec3(body,o2)

                # Translation keys
                nt,o3 = i32(body,o2)
                key0_t = None
                key0_t_abs = None
                for ki in range(nt):
                    _time,o3 = i32(body,o3)
                    if ki == 0:
                        key0_t_abs = body_off + o3
                    v,o3 = vec3(body,o3)
                    if ki == 0:
                        key0_t = v
                    o3 += 12+12  # inTan + outTan

                # Rotation keys
                nr,o4 = i32(body,o3)
                key0_r = None
                key0_r_abs = None
                for ki in range(nr):
                    _time,o4 = i32(body,o4)
                    if ki == 0:
                        key0_r_abs = body_off + o4
                    q,o4 = quat(body,o4)
                    if ki == 0:
                        key0_r = q
                    o4 += 4*5  # tcb

                # Scale keys
                ns,o5 = i32(body,o4)
                key0_s = None
                key0_s_abs = None
                for ki in range(ns):
                    _time,o5 = i32(body,o5)
                    if ki == 0:
                        key0_s_abs = body_off + o5
                    s,o5 = vec3(body,o5)
                    if ki == 0:
                        key0_s = s
                    o5 += 12+12  # inTan + outTan

                out[name] = {
                    "parent": parent,
                    "pivot_abs": pivot_abs,
                    "piv_t": piv_t,
                    "piv_r": piv_r,
                    "piv_s": piv_s,
                    "piv_s_abs": piv_s_abs,
                    "key0_t": key0_t,
                    "key0_t_abs": key0_t_abs,
                    "key0_r": key0_r,
                    "key0_r_abs": key0_r_abs,
                    "key0_s": key0_s,
                    "key0_s_abs": key0_s_abs,
                }

        off = sec_end
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Patch keyframed mesh pivot + key0 TRS from TEMPLATE into INPUT and write OUT. v4.6 only."
    )
    ap.add_argument("--template", required=True, help="Template/original .vfx (source of pivot + key0 scale).")
    ap.add_argument("--in", dest="invfx", required=True, help="Input .vfx to patch (e.g. TRUEEXPORT output).")
    ap.add_argument("--out", required=True, help="Output patched .vfx.")
    ap.add_argument("--mesh", action="append", default=[], help="Optional mesh name(s) to patch. Default: all common keyframed meshes.")
    args = ap.parse_args()

    tmpl_b = open(args.template,"rb").read()
    in_b   = bytearray(open(args.invfx,"rb").read())

    td = keyframed_info_v46(tmpl_b)
    nd = keyframed_info_v46(in_b)

    common = sorted(set(td.keys()) & set(nd.keys()))
    if args.mesh:
        want = set(args.mesh)
        common = [m for m in common if m in want]

    if not common:
        raise SystemExit("No matching keyframed meshes found to patch.")

    patched = 0
    for name in common:
        t = td[name]
        n = nd[name]

        # Pivot TRS
        off = n["pivot_abs"]
        struct.pack_into("<3f", in_b, off, *t["piv_t"])
        struct.pack_into("<4f", in_b, off+12, *t["piv_r"])
        if n.get("piv_s_abs") is not None and t.get("piv_s") is not None:
            struct.pack_into("<3f", in_b, n["piv_s_abs"], *t["piv_s"])

        # Key0 TRS values (if present)
        if n.get("key0_t_abs") is not None and t.get("key0_t") is not None:
            struct.pack_into("<3f", in_b, n["key0_t_abs"], *t["key0_t"])
        if n.get("key0_r_abs") is not None and t.get("key0_r") is not None:
            struct.pack_into("<4f", in_b, n["key0_r_abs"], *t["key0_r"])
        if n.get("key0_s_abs") is not None and t.get("key0_s") is not None:
            struct.pack_into("<3f", in_b, n["key0_s_abs"], *t["key0_s"])

        patched += 1
        print(f"[PATCH] {name} pivot+key0 TRS copied from template")

    open(args.out,"wb").write(in_b)
    print(f"Wrote: {args.out} patched_meshes={patched}")

if __name__=="__main__":
    main()
