import sys, struct

# Usage:
#   python vfxstudio/tools/dump_scene_root_blocks.py <file.vfx>

def read_cstr(buf, off):
    end = buf.find(b"\x00", off)
    if end == -1:
        return "", off
    s = buf[off:end]
    try:
        txt = s.decode("ascii","replace")
    except Exception:
        txt = ""
    # accept printable
    if all(32 <= b <= 126 for b in s):
        return txt, end+1
    return "", off

def main():
    if len(sys.argv) < 2:
        print("usage: python vfxstudio/tools/dump_scene_root_blocks.py <file.vfx>")
        return 2

    sys.path.insert(0, ".")
    from vfxstudio.libvfx.libvfx.reader import read_vfx

    path = sys.argv[1]
    v = read_vfx(path)

    # Scene Root appears in chunk index 2 for CTFflag-blue, but we search by names
    target = None
    for c in v.chunks:
        body = (getattr(c,"body",None) or getattr(c,"data",b"") or b"")
        if not body:
            continue
        p = 0
        n1, p = read_cstr(body, p)
        n2, p = read_cstr(body, p)
        if n2 == "Scene Root":
            target = (n1, n2, p, body[p:])
            break

    if not target:
        print("Scene Root record not found.")
        return 1

    n1, n2, poff, payload = target
    print(f"SceneRoot: name1={n1!r} name2={n2!r} payload_offset={poff} payload_len={len(payload)}")

    starts = [57,153,249,345,441]
    for bi,s in enumerate(starts):
        block = payload[s:s+96]
        if len(block) < 96:
            print(f"block {bi}: start {s} incomplete (len={len(block)})")
            continue
        m = struct.unpack("<16f", block[:64])
        tail = struct.unpack("<8f", block[64:96])
        print(f"\n=== block {bi} start={s} ===")
        for r in range(4):
            row = m[r*4:(r+1)*4]
            print("  " + " ".join(f"{x: .6f}" for x in row))
        print("tail8: " + " ".join(f"{x: .6f}" for x in tail))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
