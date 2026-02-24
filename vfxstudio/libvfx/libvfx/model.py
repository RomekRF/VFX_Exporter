from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VfxMaterial:
    name: str
    texture: Optional[str] = None

@dataclass
class VfxMeshInfo:
    name: str
    morph: bool
    frames: int
    fps: int
    flags_hex: str

@dataclass
class VfxSummary:
    path: str
    file_size: int
    version_hex: str
    end_frame: int
    meshes: List[VfxMeshInfo]
    materials: List[VfxMaterial]
    dummy_count: int
    particle_count: int
