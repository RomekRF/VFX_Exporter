import json, sys

def main(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        g = json.load(f)
    nodes = g.get('nodes') or []
    scenes = g.get('scenes') or []
    if not scenes:
        raise SystemExit('No scenes in glTF')

    s0 = scenes[0]
    roots = list(s0.get('nodes') or [])
    if not roots:
        # If scene has no roots, assume all nodes are roots
        roots = list(range(len(nodes)))

    # If already single root named __VFX_ROOT__, do nothing
    if len(roots) == 1:
        r = roots[0]
        if r < len(nodes) and isinstance(nodes[r], dict) and nodes[r].get('name') == '__VFX_ROOT__':
            print('Already rootified.')
            return

    root_idx = len(nodes)
    root_node = {'name':'__VFX_ROOT__', 'children': roots}
    nodes.append(root_node)

    s0['nodes'] = [root_idx]
    g['nodes'] = nodes
    g['scenes'][0] = s0

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(g, f, ensure_ascii=False)
    print(f'Rootified: scene0.nodes -> [{root_idx}]  (children={len(roots)})')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python tools/gltf_rootify.py file.gltf')
    main(sys.argv[1])
