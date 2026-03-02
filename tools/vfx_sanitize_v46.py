import struct, sys, os

V46 = 0x00040006

def u32(b,o): return struct.unpack_from("<I",b,o)[0], o+4
def i32(b,o): return struct.unpack_from("<i",b,o)[0], o+4

def cstr(b,o):
    e = b.index(0,o)
    return b[o:e].decode("latin1","replace"), e+1

def parse_sfxo_counts(body: bytes):
    o=0
    name,o = cstr(body,o)
    parent,o = cstr(body,o)
    save_parent_ofs = o  # <-- byte in BODY
    save_parent = body[o]; o += 1
    nv,o = i32(body,o)
    nf,o = i32(body,o)
    # v4.6 face block is 96 bytes each
    o += nf * 96

    fps,o = i32(body,o)
    _s,o = struct.unpack_from("<f", body, o)[0], o+4
    _e,o = struct.unpack_from("<f", body, o)[0], o+4
    nframes,o = i32(body,o)

    nmats,o = i32(body,o)
    o += nmats * 4

    # bbox-ish (3f + f)
    o += 16

    _flags,o = u32(body,o)
    nfv,o = i32(body,o)

    # skip face-vertex records (variable)
    for _ in range(nfv):
        o += 16
        naf,o = i32(body,o)
        o += naf * 4

    is_keyframed = (body[o] != 0)

    return dict(
        name=name, parent=parent,
        nv=nv, nf=nf, nfv=nfv, nframes=nframes, nmats=nmats,
        is_keyframed=is_keyframed,
        save_parent_ofs=save_parent_ofs
    )

def iter_sections(vfx_bytes: bytes):
    b=vfx_bytes
    off=128
    while off+8 <= len(b):
        stype = struct.unpack_from("<I", b, off)[0]
        size  = struct.unpack_from("<I", b, off+4)[0]
        if size < 8:
            break
        body_start = off+8
        body_end   = off+4+size
        if body_end > len(b):
            break
        t = stype.to_bytes(4,"little").decode("latin1","replace")
        yield off, t, size, body_start, body_end
        off = off + 4 + size

def sanitize(inp: str, outp: str):
    b = bytearray(open(inp,"rb").read())
    if b[:4] != b"VSFX":
        raise SystemExit("Not VSFX")
    ver = struct.unpack_from("<I", b, 4)[0]
    if ver != V46:
        raise SystemExit(f"Only v4.6 supported here. Got 0x{ver:08X}")

    # read existing header
    unk = list(struct.unpack_from("<30I", b, 8))

    mesh_count = 0
    dmmy_count = 0
    matl_count = 0
    total_faces = 0
    total_fv = 0
    sum_frames = 0
    sum_matrefs = 0
    keyfr = 0

    # also force save_parent=0 for every mesh (matches stock RF)
    for off, t, size, bs, be in iter_sections(b):
        if t == "SFXO":
            info = parse_sfxo_counts(bytes(b[bs:be]))
            mesh_count += 1
            total_faces += info["nf"]
            total_fv    += info["nfv"]
            sum_frames  += info["nframes"]
            sum_matrefs += info["nmats"]
            if info["is_keyframed"]:
                keyfr += 1

            # patch save_parent byte
            save_abs = bs + info["save_parent_ofs"]
            b[save_abs] = 0

        elif t == "DMMY":
            dmmy_count += 1
        elif t == "MATL":
            matl_count += 1

    # Patch only the indices we have confidently mapped from real RF files:
    # unk[02]=mesh_count, unk[04]=dummy_count, unk[09]=matl_count
    # unk[13]=total_faces, unk[14]=sum(mesh.num_materials), unk[15]=total_face_vertices
    # unk[16]=total_faces*3 (adjacency total), unk[17]=sum(num_frames), unk[18]=mesh_count
    # unk[20..23]=keyframed mesh count (must match when keyframed meshes exist)
    unk[2]  = mesh_count
    unk[4]  = dmmy_count
    unk[9]  = matl_count
    unk[13] = total_faces
    unk[14] = sum_matrefs
    unk[15] = total_fv
    unk[16] = total_faces * 3
    unk[17] = sum_frames
    unk[18] = mesh_count
    unk[20] = keyfr
    unk[21] = keyfr
    unk[22] = keyfr
    unk[23] = keyfr

    struct.pack_into("<30I", b, 8, *[u & 0xFFFFFFFF for u in unk])
    open(outp,"wb").write(b)

    print("OK: wrote", outp)
    print(f"counts: meshes={mesh_count} dummies={dmmy_count} matl={matl_count} faces={total_faces} faceVerts={total_fv} framesSum={sum_frames} keyfr={keyfr}")

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("usage: vfx_sanitize_v46.py in.vfx out.vfx")
        raise SystemExit(2)
    sanitize(sys.argv[1], sys.argv[2])
