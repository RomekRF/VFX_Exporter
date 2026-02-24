from __future__ import annotations
import argparse
import os
import sys
from rich.console import Console
from rich.table import Table

# Import libvfx via relative path add (keeps this repo simple initially)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
LIBVFX_ROOT = os.path.join(REPO_ROOT, "libvfx")
if LIBVFX_ROOT not in sys.path:
    sys.path.insert(0, LIBVFX_ROOT)

from libvfx.summary import summarize_vfx

console = Console()

def cmd_summary(args: argparse.Namespace) -> int:
    s = summarize_vfx(args.vfx)
    console.print(f"[bold]File:[/bold] {s.path}")
    console.print(f"[bold]Size:[/bold] {s.file_size} bytes")
    console.print(f"[bold]Version:[/bold] {s.version_hex}")
    console.print(f"[bold]End frame:[/bold] {s.end_frame}")
    console.print(f"[bold]Dummies:[/bold] {s.dummy_count}   [bold]Particles:[/bold] {s.particle_count}")

    mt = Table(title="Meshes (from --debug-frames backend)")
    mt.add_column("Name", overflow="fold")
    mt.add_column("Morph", justify="center")
    mt.add_column("FPS", justify="right")
    mt.add_column("Frames", justify="right")
    mt.add_column("Flags", justify="right")

    if s.meshes:
        for m in s.meshes:
            mt.add_row(m.name, str(m.morph), str(m.fps), str(m.frames), m.flags_hex)
    else:
        mt.add_row("(none found)", "-", "-", "-", "-")
    console.print(mt)

    t = Table(title="Textures / Materials (scan)")
    t.add_column("Texture name", overflow="fold")
    if s.materials:
        for m in s.materials:
            t.add_row(m.texture or m.name)
    else:
        t.add_row("(none found)")
    console.print(t)
    return 0

def cmd_to_gltf(args: argparse.Namespace) -> int:
    # For now, call your existing EXE as the backend if present.
    # This is intentional: GUI will later call THIS CLI, which will call libvfx or legacy exe as needed.
    exe = args.exe
    if not os.path.exists(exe):
        raise SystemExit(f"Missing converter EXE: {exe}")

    out = args.out
    if not out.lower().endswith((".gltf", ".glb")):
        raise SystemExit("--out must end with .gltf or .glb (use .gltf for now)")

    # Use your existing converter flags
    import subprocess
    cmd = [exe, "--gltf", args.vfx]
    if args.debug_frames:
        cmd.insert(1, "--debug-frames")

    console.print(f"[bold]Running:[/bold] {' '.join(cmd)}")
    subprocess.check_call(cmd)
    console.print("[green]Done.[/green] (Converter writes next to input; we will wire --out next.)")
    return 0

def cmd_to_vfx(args: argparse.Namespace) -> int:
    console.print("[yellow]to-vfx is scaffolded but not implemented yet.[/yellow]")
    console.print("Next step: implement VFX writer in libvfx with lossless + authoring modes.")
    return 2

def cmd_verify_roundtrip(args: argparse.Namespace) -> int:
    console.print("[yellow]verify-roundtrip is scaffolded but not implemented yet.[/yellow]")
    console.print("Next step: parse->write and binary-compare in lossless mode.")
    return 2

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vfxcli")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="Print info about a VFX file")
    s.add_argument("vfx")
    s.set_defaults(fn=cmd_summary)

    g = sub.add_parser("to-gltf", help="Convert VFX to glTF (currently uses existing vfx2obj.exe backend)")
    g.add_argument("vfx")
    g.add_argument("--exe", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "dist", "vfx2obj.exe"))
    g.add_argument("--out", default="out.gltf")
    g.add_argument("--debug-frames", action="store_true")
    g.set_defaults(fn=cmd_to_gltf)

    v = sub.add_parser("to-vfx", help="Convert glTF to VFX (coming next)")
    v.add_argument("gltf")
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_to_vfx)

    r = sub.add_parser("verify-roundtrip", help="Read->write and compare (lossless mode; coming next)")
    r.add_argument("vfx")
    r.set_defaults(fn=cmd_verify_roundtrip)

    return p

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args))

if __name__ == "__main__":
    raise SystemExit(main())

