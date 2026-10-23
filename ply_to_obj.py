"""Convert PLY geometry to OBJ without reconstruction or coordinate scaling."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def convert_ply_to_obj(source, output=None, *, points_only=False,
                       vertex_colors=False, overwrite=False):
    source = Path(source)
    output = Path(output) if output is not None else source.with_suffix('.obj')
    if source.suffix.lower() != '.ply' or output.suffix.lower() != '.obj':
        raise ValueError('Input must be .ply and output must be .obj.')
    if not source.is_file():
        raise FileNotFoundError(f'Input not found: {source}')
    if source.resolve() == output.resolve():
        raise ValueError('Output must not replace the input.')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Output exists: {output}; use --overwrite to replace it.')

    if points_only:
        geometry = o3d.io.read_point_cloud(str(source))
        vertices = np.asarray(geometry.points)
        faces = np.empty((0, 3), dtype=np.int64)
        normals = np.asarray(geometry.normals)
        colors = np.asarray(geometry.colors)
    else:
        geometry = o3d.io.read_triangle_mesh(str(source))
        vertices = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.triangles)
        normals = np.asarray(geometry.vertex_normals)
        colors = np.asarray(geometry.vertex_colors)
        if len(faces) == 0:
            raise ValueError('PLY has no faces. Reconstruct a mesh first, or use '
                             '--points-only for an OBJ point cloud (not a surface).')
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError('PLY must contain nonempty, finite XYZ vertices.')
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('PLY contains invalid face indices.')
    has_normals = len(normals) == len(vertices) and np.isfinite(normals).all()
    has_colors = vertex_colors and len(colors) == len(vertices) and np.isfinite(colors).all()

    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwriting even after the check.
    with output.open('w' if overwrite else 'x', encoding='utf-8', newline='\n') as stream:
        stream.write('# PLY to OBJ: original coordinates and units preserved\n')
        if has_colors:
            stream.write('# RGB uses the optional v x y z r g b extension (not a texture)\n')
        for index, vertex in enumerate(vertices):
            values = np.concatenate((vertex, colors[index])) if has_colors else vertex
            stream.write('v ' + ' '.join(format(float(value), '.17g') for value in values) + '\n')
        if has_normals and not points_only:
            for normal in normals:
                stream.write('vn ' + ' '.join(format(float(value), '.17g') for value in normal) + '\n')
        if points_only:
            for index in range(1, len(vertices) + 1):
                stream.write(f'p {index}\n')
        else:
            for face in faces:
                indices = [int(index) + 1 for index in face]
                refs = [f'{index}//{index}' for index in indices] if has_normals else map(str, indices)
                stream.write('f ' + ' '.join(refs) + '\n')
    return {'output': output, 'vertices': len(vertices), 'faces': len(faces),
            'vertex_colors': bool(has_colors)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input PLY mesh (ASCII or binary).')
    parser.add_argument('--out', type=Path, help='Default: input path with .obj extension.')
    parser.add_argument('--points-only', action='store_true', help='Export points only; never creates a surface.')
    parser.add_argument('--vertex-colors', action='store_true', help='Optional RGB vertex extension; some viewers ignore it.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output OBJ.')
    args = parser.parse_args(argv)
    try:
        result = convert_ply_to_obj(args.input, args.out, points_only=args.points_only,
                                    vertex_colors=args.vertex_colors, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f"Saved: {result['output']} | vertices={result['vertices']} faces={result['faces']}")
    if result['vertex_colors']:
        print('RGB is a vertex-color extension; viewer support varies. No texture/MTL is generated.')
    if args.points_only:
        print('Point cloud only: this OBJ has no surface faces.')


if __name__ == '__main__':
    main()
