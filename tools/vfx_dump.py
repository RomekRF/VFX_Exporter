import struct, sys, os

SECTION_ENUM = {
    0x4F584653: ("mesh","sfxo"),
    0x4C54414D: ("material","matl"),
    0x54524150: ("particle","part"),
    0x54474C41: ("light","algt"),
    0x50524157: ("spacewarp","warp"),
    0x454E4843: ("chain","chne"),
    0x444F4D4D: ("mmod","mmod"),
    0x41524D43: ("camera","cmra"),
    0x594D4D44: ("dummy","dmmy"),
}

HDR_FIELDS_V46 = [
  "flags","end_frame","num_meshes","num_lights","num_dummies","num_particle_systems","num_spacewarps","num_cameras","num_selsets",
  "num_materials","num_mix_frames","num_self_illumination_frames","num_opacity_frames",
  "num_faces","num_mesh_material_indices","num_vertex_normals","num_adjacent_faces","num_mesh_frames","num_uv_frames",
  "num_mesh_transform_frames","num_mesh_transform_keyframe_lists","num_mesh_translation_keys","num_mesh_rotation_keys","num_mesh_scale_keys",
  "num_light_frames","num_dummy_frames","num_part_sys_frames","num_spacewarp_frames","num_camera_frames","num_selset_objects"
]

def cstr(buf, off):
    end = buf.index(0, off)
    return buf[off:end].decode("latin1","replace"), end+1

def fourcc(u):
    return struct.pack("<I", u).decode("latin1","replace")

def dump(path):
    b = open(path,"rb").read()
    if len(b) < 128:
        print(f"{path}: TOO SMALL ({len(b)} bytes)")
        return

    sig, ver = struct.unpack_from("<II", b, 0)
    sigs = sig.to_bytes(4,"little", signed=False)
    print("\n" + "="*80)
    print(os.path.basename(path))
    print(f"size={len(b)}  sig={sigs!r}  ver=0x{ver:08X}")

    unk = struct.unpack_from("<30I", b, 8)
    if ver == 0x00040006:
        for k,v in zip(HDR_FIELDS_V46, unk):
            print(f"  {k:34s} {v}")
    else:
        print("  (non-4.6 header; raw unknown[30] shown)")
        for i,v in enumerate(unk):
            print(f"  unk[{i:02d}] = {v}")

    off = 128
    si = 0
    while off < len(b):
        if off+8 > len(b):
            print(f"  [TRUNC] at {off}")
            break
        stype, slen = struct.unpack_from("<II", b, off)
        body_len = slen - 4
        sec_start = off
        body_off = off + 8
        sec_end = off + 8 + body_len
        name = SECTION_ENUM.get(stype, (f"UNKNOWN({fourcc(stype)})", fourcc(stype)))[0]
        print(f"  sec[{si:02d}] {name:8s} type=0x{stype:08X} len={slen}  bytes={sec_end-sec_start}  @0x{sec_start:X}")
        body = b[body_off:sec_end]

        # Light parse for mesh/material/dummy (enough to spot the problem)
        try:
            if name == "mesh":
                o=0
                mname,o = cstr(body,o)
                parent,o = cstr(body,o)
                save_parent = body[o]; o+=1
                numv, = struct.unpack_from("<i", body, o); o+=4
                numf, = struct.unpack_from("<i", body, o); o+=4
                # skip faces (v4.6 face struct size = 76 bytes)
                o += numf * 76
                fps, = struct.unpack_from("<i", body, o); o+=4
                start_time, = struct.unpack_from("<f", body, o); o+=4
                end_time, = struct.unpack_from("<f", body, o); o+=4
                num_frames, = struct.unpack_from("<i", body, o); o+=4
                nmat, = struct.unpack_from("<i", body, o); o+=4
                mats = []
                for _ in range(nmat):
                    mi, = struct.unpack_from("<i", body, o); o+=4
                    mats.append(mi)
                print(f"        mesh='{mname}' parent='{parent}' v={numv} f={numf} fps={fps} frames={num_frames} mats={mats}")

            elif name == "material":
                o=0
                mtype, = struct.unpack_from("<i", body, o); o+=4
                fps, = struct.unpack_from("<i", body, o); o+=4
                additive = body[o]; o+=1
                tex,o = cstr(body,o)
                sf, = struct.unpack_from("<i", body, o); o+=4
                pr, = struct.unpack_from("<f", body, o); o+=4
                at, = struct.unpack_from("<i", body, o); o+=4
                print(f"        material type={mtype} fps={fps} add={additive} tex='{tex}' start={sf} rate={pr} anim={at}")

            elif name == "dummy":
                o=0
                dname,o = cstr(body,o)
                parent,o = cstr(body,o)
                save_parent = body[o]; o+=1
                # pos (3f) + orient (4f)
                o += 12 + 16
                nfr, = struct.unpack_from("<i", body, o); o+=4
                print(f"        dummy='{dname}' parent='{parent}' frames={nfr}")

        except Exception as e:
            print(f"        (parse note) {e}")

        off = sec_end
        si += 1

if __name__ == "__main__":
    for p in sys.argv[1:]:
        dump(p)
