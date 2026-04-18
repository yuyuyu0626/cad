import argparse
import os
import itertools
import numpy as np
import trimesh


PERMUTATIONS = {
    'xyz': (0, 1, 2),
    'xzy': (0, 2, 1),
    'yxz': (1, 0, 2),
    'yzx': (1, 2, 0),
    'zxy': (2, 0, 1),
    'zyx': (2, 1, 0),
}


def load_single_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f'No mesh geometry found in: {path}')
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'Unsupported mesh type: {type(mesh)}')
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError('Input mesh is empty.')
    return mesh.copy()



def get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    vc = None
    try:
        vc = mesh.visual.vertex_colors
    except Exception:
        vc = None

    if vc is None or len(vc) != len(mesh.vertices):
        vc = np.tile(np.array([[200, 200, 200, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    else:
        vc = np.asarray(vc)
        if vc.shape[1] == 3:
            vc = np.concatenate([vc.astype(np.uint8), np.full((vc.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        else:
            vc = vc[:, :4].astype(np.uint8)
    return vc



def write_ascii_ply(path: str, vertices: np.ndarray, normals: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar alpha\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for v, n, c in zip(vertices, normals, colors):
            f.write(
                f'{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} '
                f'{n[0]:.8f} {n[1]:.8f} {n[2]:.8f} '
                f'{int(c[0])} {int(c[1])} {int(c[2])} {int(c[3])}\n'
            )
        for face in faces:
            f.write(f'3 {int(face[0])} {int(face[1])} {int(face[2])}\n')



def main() -> None:
    parser = argparse.ArgumentParser(description='Convert OBJ/STL/PLY mesh to BOP-style centered ASCII PLY.')
    parser.add_argument('--input_mesh', required=True, help='Input mesh path: .obj / .stl / .ply')
    parser.add_argument('--output_ply', required=True, help='Output BOP PLY path, e.g. models/obj_000001.ply')
    parser.add_argument('--scale', type=float, default=1.0, help='Optional extra uniform scale. Default keeps original mesh scale.')
    parser.add_argument('--target_extent', type=float, nargs=3, default=None,
                        help='Optional target bbox extent. Use ONLY when you intentionally want to rescale the mesh.')
    parser.add_argument('--axis_order', choices=sorted(PERMUTATIONS.keys()), default='xyz',
                        help='Optional axis permutation before centering. Example: xzy swaps Y and Z.')
    args = parser.parse_args()

    mesh = load_single_mesh(args.input_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # axis permutation
    perm = PERMUTATIONS[args.axis_order]
    vertices = vertices[:, perm]

    # preserve original scale unless user explicitly asks otherwise
    vertices *= args.scale

    if args.target_extent is not None:
        cur_extent = vertices.max(axis=0) - vertices.min(axis=0)
        target_extent = np.asarray(args.target_extent, dtype=np.float64)
        valid = cur_extent > 1e-9
        if not np.any(valid):
            raise ValueError('Current mesh extent is degenerate.')
        uniform_scale = float(np.median(target_extent[valid] / cur_extent[valid]))
        vertices *= uniform_scale
        print(f'[INFO] Applied target-extent uniform scale: {uniform_scale:.8f}')

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    vertices -= bbox_center

    mesh.vertices = vertices
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(normals) != len(vertices):
        normals = np.zeros_like(vertices)
    colors = get_vertex_colors(mesh)

    output_ply = os.path.abspath(args.output_ply)
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    write_ascii_ply(output_ply, vertices, normals, colors, faces)

    final_min = vertices.min(axis=0)
    final_max = vertices.max(axis=0)
    final_extent = final_max - final_min
    print('[OK] Wrote:', output_ply)
    print('[INFO] Final bbox min:', final_min.tolist())
    print('[INFO] Final bbox max:', final_max.tolist())
    print('[INFO] Final bbox extent:', final_extent.tolist())
    print('[INFO] Final bbox center:', (0.5 * (final_min + final_max)).tolist())


if __name__ == '__main__':
    main()
