import blenderproc as bproc
# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
s2_p1_gen_pbr_data.py is used to generate PBR data.  
The original script is adapted from BlenderProc2.  
Project link: https://github.com/DLR-RM/BlenderProc

Usage:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    
Arguments (example: s2_p1_gen_pbr_data.sh 0 42 ... ):
    Arg 1 (`GPU_ID`): GPU index. Set to 0 for the first GPU.
    Arg 2 (`SCENE_NUM`): Number of scenes; total images generated = 1000 * 42.
    Arg 3 (`cc0textures`): Path to the cc0textures material library.
    Arg 4 (`dataset_path`): Path to the dataset.
    Arg 4 (`s2_p1_gen_pbr_data`): Path to the s2_p1_gen_pbr_data.py script.
    
------------------------------------------------------    

s2_p1_gen_pbr_data.py 用于生成 PBR 数据，原始脚本改编自 BlenderProc2。  
项目链接: https://github.com/DLR-RM/BlenderProc

运行方法:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

参数说明 (以 s2_p1_gen_pbr_data.sh 0 42 ... 为例):
    参数 1 (`GPU_ID`): GPU 的编号。设置为 0 表示使用第一块显卡。
    参数 2 (`SCENE_NUM`): 场景数量，对应生成的图像数 = 1000 * 42
    参数 3 (`cc0textures`): cc0textures 材质库的路径。
    参数 3 (`dataset_path`): 数据集的路径。
    参数 3 (`s2_p1_gen_pbr_data`): s2_p1_gen_pbr_data.py的路径。
'''

import os
import bpy
import argparse
import numpy as np
from tqdm import tqdm
from kasal.utils.io_json import load_json2dict, write_dict2json
import json
import mathutils
from bpy_extras.object_utils import world_to_camera_view
from collections import defaultdict

PERMUTATIONS = {
    'xyz': (0, 1, 2),
    'xzy': (0, 2, 1),
    'yxz': (1, 0, 2),
    'yzx': (1, 2, 0),
    'zxy': (2, 0, 1),
    'zyx': (2, 1, 0),
}

def get_3d_bbox(obj):
    """
    获取 Blender 物体的 8 个 3D 顶点 (世界坐标系)
    """
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    return bbox_corners

def project_3d_to_2d(cam, bbox_corners, resolution_x, resolution_y):
    """
    将 3D 顶点投影到 2D 图像平面，返回像素坐标
    """
    pts_2d = []
    for corner in bbox_corners:
        # 使用 Blender 内置的投影函数
        co_2d = world_to_camera_view(bpy.context.scene, cam, corner)
        
        # 将归一化坐标转换为像素坐标
        # 注意：Blender 的 Y 轴方向（从下到上）与 OpenCV/PyTorch（从上到下）是反的
        u = round(co_2d.x * resolution_x)
        v = round((1.0 - co_2d.y) * resolution_y)
        
        # 限定在图像范围内
        u = max(0, min(u, resolution_x - 1))
        v = max(0, min(v, resolution_y - 1))
        pts_2d.append([u, v])
    return pts_2d


def load_flat_cc0textures_512(cc_textures_path: str):
    texture_root = os.path.join(cc_textures_path, 'cc0textures')
    if not os.path.isdir(texture_root):
        raise FileNotFoundError(f'cc0textures-512 folder not found: {texture_root}')

    grouped_files = defaultdict(dict)
    for filename in os.listdir(texture_root):
        lower = filename.lower()
        if not lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
        stem, _ = os.path.splitext(filename)
        if '_' not in stem:
            continue
        tex_id, tex_kind = stem.rsplit('_', 1)
        grouped_files[tex_id][tex_kind.lower()] = os.path.join(texture_root, filename)

    materials = []
    for tex_id, tex_files in grouped_files.items():
        if 'color' not in tex_files:
            continue

        mat = bproc.material.create(f'cc0_{tex_id}')
        blender_mat = mat.blender_obj
        blender_mat.use_nodes = True
        nodes = blender_mat.node_tree.nodes
        links = blender_mat.node_tree.links
        principled = nodes.get("Principled BSDF")
        output = nodes.get("Material Output")
        if principled is None or output is None:
            continue

        for node in list(nodes):
            if node.name not in {"Principled BSDF", "Material Output"}:
                nodes.remove(node)

        color_node = nodes.new(type='ShaderNodeTexImage')
        color_node.image = bpy.data.images.load(tex_files['color'], check_existing=True)
        color_node.location = (-800, 250)
        links.new(color_node.outputs['Color'], principled.inputs['Base Color'])

        if 'roughness' in tex_files:
            roughness_node = nodes.new(type='ShaderNodeTexImage')
            roughness_node.image = bpy.data.images.load(tex_files['roughness'], check_existing=True)
            roughness_node.image.colorspace_settings.name = 'Non-Color'
            roughness_node.location = (-800, 0)
            links.new(roughness_node.outputs['Color'], principled.inputs['Roughness'])

        if 'normal' in tex_files:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.image = bpy.data.images.load(tex_files['normal'], check_existing=True)
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
            normal_tex_node.location = (-800, -250)

            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_map_node.location = (-500, -250)
            links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])

        materials.append(mat)

    return materials


def import_textured_obj(filepath: str) -> bpy.types.Object:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    existing_names = {obj.name for obj in bpy.data.objects}
    bpy.ops.object.select_all(action='DESELECT')
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.obj':
        try:
            bpy.ops.wm.obj_import(filepath=filepath)
        except Exception:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(
            f'Unsupported render model format: {filepath}. '
            f'Supported formats are .obj, .glb, .gltf, .fbx'
        )

    imported = [obj for obj in bpy.data.objects if obj.name not in existing_names and obj.type == 'MESH']
    if not imported:
        raise RuntimeError(f'No mesh objects imported from render model: {filepath}')

    if len(imported) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
        active_name = bpy.context.view_layer.objects.active.name
        imported = [bpy.data.objects[active_name]]

    obj = bpy.data.objects[imported[0].name]
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    return obj


def _rotation_matrix_from_euler_deg(rotation_deg):
    euler = mathutils.Euler(tuple(np.deg2rad(rotation_deg).tolist()), 'XYZ')
    return np.asarray(euler.to_matrix(), dtype=np.float64)


def align_render_template_to_model(
    blender_obj: bpy.types.Object,
    target_extent: np.ndarray,
    axis_order: str = 'auto',
    base_scale: float = 1.0,
    rotation_deg = (0.0, 0.0, 0.0),
):
    mesh = blender_obj.data
    vertices = np.asarray([v.co[:] for v in mesh.vertices], dtype=np.float64)
    if vertices.size == 0:
        raise ValueError('Imported OBJ mesh has no vertices.')

    def evaluate_perm(perm_name: str):
        perm = PERMUTATIONS[perm_name]
        verts_perm = vertices[:, perm] * float(base_scale)
        cur_extent = verts_perm.max(axis=0) - verts_perm.min(axis=0)
        valid = cur_extent > 1e-9
        if not np.any(valid):
            raise ValueError('Imported OBJ mesh extent is degenerate.')
        uniform_scale = float(np.median(target_extent[valid] / cur_extent[valid]))
        aligned_extent = cur_extent * uniform_scale
        score = float(np.linalg.norm(aligned_extent - target_extent))
        return score, perm, uniform_scale

    if axis_order == 'auto':
        candidates = [(name, *evaluate_perm(name)) for name in PERMUTATIONS.keys()]
        best_name, _, best_perm, uniform_scale = min(candidates, key=lambda x: x[1])
        print(f'[INFO] Auto-selected render axis order: {best_name}')
    else:
        _, best_perm, uniform_scale = evaluate_perm(axis_order)
        best_name = axis_order

    verts_aligned = vertices[:, best_perm] * float(base_scale) * float(uniform_scale)
    rot = _rotation_matrix_from_euler_deg(rotation_deg)
    verts_aligned = (rot @ verts_aligned.T).T
    bbox_min = verts_aligned.min(axis=0)
    bbox_max = verts_aligned.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    verts_aligned -= bbox_center

    for vertex, coord in zip(mesh.vertices, verts_aligned):
        vertex.co = mathutils.Vector(coord.tolist())
    mesh.update()
    bpy.context.view_layer.update()
    print(f'[INFO] Render template aligned with axis_order={best_name}, uniform_scale={uniform_scale:.8f}')
    final_extent = verts_aligned.max(axis=0) - verts_aligned.min(axis=0)
    print(f'[INFO] Render template final extent: {final_extent.tolist()}')


def _resolve_object(obj_or_name):
    if obj_or_name is None:
        return None
    if isinstance(obj_or_name, str):
        return bpy.data.objects.get(obj_or_name)
    name = getattr(obj_or_name, 'name', None)
    if name is None:
        return None
    return bpy.data.objects.get(name)


def duplicate_render_template(template_obj, name: str) -> bpy.types.Object:
    template = _resolve_object(template_obj)
    if template is None:
        raise RuntimeError(f'Render template object is no longer available when creating proxy: {name}')

    proxy = template.copy()
    proxy.data = template.data.copy()
    proxy.animation_data_clear()
    proxy.name = name
    bpy.context.collection.objects.link(proxy)
    proxy.hide_render = False
    proxy.hide_set(False)
    return proxy


def cleanup_render_proxies(render_proxies):
    for proxy in render_proxies:
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)


def cleanup_render_template(template_obj):
    template = _resolve_object(template_obj)
    if template is None:
        return
    mesh = template.data
    bpy.data.objects.remove(template, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh, do_unlink=True)

if __name__ == '__main__':
    
    # Retrieve the GPU ID.
    # 获取 GPU 的编号。
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('gpu_id', type=int, help='')
    parser.add_argument('cc0textures', type=str, help='')
    parser.add_argument('--num-scenes', type=int, default=50, help='Number of scenes to generate in this run.')
    parser.add_argument('--bop-num-worker', type=int, default=1,
                        help='Number of worker processes used by write_bop() when calculating GT masks/info/coco.')
    parser.add_argument('--num-objects', type=int, default=4,
                        help='Number of object instances spawned per scene.')
    parser.add_argument('--render-model-obj', type=str, default=None,
                        help='Optional textured render model used only for RGB appearance. '
                             'Supports .obj/.glb/.gltf/.fbx. BOP PLY is still used for GT/labels.')
    parser.add_argument('--render-axis-order', choices=['auto'] + sorted(PERMUTATIONS.keys()), default='auto',
                        help='Axis permutation applied to the textured OBJ before centering/scaling. "auto" matches models_info extent.')
    parser.add_argument('--render-scale', type=float, default=1.0,
                        help='Extra uniform scale applied to the textured OBJ before automatic extent matching.')
    parser.add_argument('--render-rotation-deg', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        help='Optional additional XYZ rotation in degrees applied to the textured OBJ after scaling/permutation.')
    parser.add_argument('--render-model-id', type=int, default=None,
                        help='Model id used to align the textured OBJ to models_info. Defaults to the only model id when possible.')
    args = parser.parse_args()
    gpu_id = int(args.gpu_id)

    # Retrieve the folder path of the current dataset.
    # 获取当前数据集的文件夹路径。
    current_dir = os.path.abspath(os.getcwd())

    # Retrieve the name of the dataset.
    # 获取数据集的名称。
    dataset_name = os.path.basename(current_dir) 

    # BlenderProc requires the path to the parent directory of the dataset.
    # BlenderProc 需要传入数据集的父级目录路径。
    bop_parent_path = os.path.dirname(current_dir)

    # Load the 3D model information of the dataset.
    # 加载数据集的 3D 模型信息。
    models_info = load_json2dict(os.path.join(current_dir, 'models', 'models_info.json'))

    if not os.path.exists(os.path.join(current_dir, 'camera.json')):
        write_dict2json(os.path.join(current_dir, 'camera.json'), 
                            {
                            "cx": 325.2611083984375,
                            "cy": 242.04899588216654,
                            "depth_scale": 0.1,
                            "fx": 572.411363389757,
                            "fy": 573.5704328585578,
                            "height": 480,
                            "width": 640
                            }
                        )
    
    # Retrieve the list of 3D model IDs from the dataset.
    # 获取数据集中 3D 模型的 ID 列表。
    models_ids = []
    for key in models_info:
        models_ids.append(int(key))
    models_ids = np.array(models_ids)

    render_target_extent = None
    render_model_id = args.render_model_id
    if args.render_model_obj:
        if render_model_id is None:
            if len(models_ids) != 1:
                raise ValueError('Please pass --render-model-id when dataset contains multiple models.')
            render_model_id = int(models_ids[0])
        if str(int(render_model_id)) not in models_info:
            raise KeyError(f'No models_info entry found for render_model_id={render_model_id}')
        render_target_extent = np.asarray(
            [
                models_info[str(int(render_model_id))]['size_x'],
                models_info[str(int(render_model_id))]['size_y'],
                models_info[str(int(render_model_id))]['size_z'],
            ],
            dtype=np.float64,
        )
    # Print the parent path and name of the dataset.
    # 打印数据集的父级路径和名称。
    print('-*' * 10)
    print('-*' * 10)
    print('bop_parent_path', bop_parent_path)
    print('dataset_name', dataset_name)
    print('-*' * 10)
    print('-*' * 10)

    # Retrieve the path to the cc0textures assets.
    # 获取 cc0textures 的路径。
    cc_textures_path = args.cc0textures

    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    num_scenes = int(args.num_scenes)

    # Create the rendering scene.
    # 创建渲染场景。
    bproc.init()
    bproc.loader.load_bop_intrinsics(bop_dataset_path = bop_dataset_path)
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # Load all texture images from the cc_textures directory.
    # 加载 cc_textures 目录中的所有纹理图。
    # Prefer the project-specific 512 loader when available, otherwise support the
    # flattened cc0textures-512 layout used by this project.
    if os.path.basename(cc_textures_path) == 'cc0textures-512' and hasattr(bproc.loader, 'load_512_ccmaterials'):
        cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    elif os.path.basename(cc_textures_path) == 'cc0textures-512':
        cc_textures = load_flat_cc0textures_512(cc_textures_path)
    else:
        cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)

    
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.06, -0.06, 0.0], [-0.03, -0.03, 0.0])
        max = np.random.uniform([0.03, 0.03, 0.12], [0.06, 0.06, 0.20])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    # Set the GPU ID.
    # 设置 GPU 的编号。
    bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids = [gpu_id])

    for i in tqdm(range(num_scenes)):
        
        rand_s = np.random.rand()
        
        # Bin-picking selection mode.
        # bin-picking 的挑选模式。
        
        # idx_l = np.random.choice(models_ids, size=2, replace=True)
        # obj_ids = []
        # for _ in range(15):
        #     obj_ids.append(int(idx_l[0]))
        #     obj_ids.append(int(idx_l[1]))
        
        # Multi-class object picking mode.
        # 多类别物体的挑选模式。
        
        if rand_s > 0.5:
            idx_l = np.random.choice(models_ids, size=max(1, int(args.num_objects)), replace=True)
        else:
            idx_l = np.random.choice(models_ids, size=min(models_ids.shape[0], max(1, int(args.num_objects))), replace=False)
        obj_ids = []
        for idx_i in idx_l:
            obj_ids.append(int(idx_i))
        
        # Load objects into BlenderProc.
        # 将物体加载到 BlenderProc 中。
        target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = bop_dataset_path, 
                                                    mm2m = True,
                                                    obj_ids = obj_ids,
                                                    )
        render_proxies = []
        render_template = None
        render_template_name = None
        
        # Set object materials and poses, then render 20 frames.
        # 设置物体的材质和位姿，并渲染 20 帧图像。
        
        for obj in (target_bop_objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
        sampled_target_bop_objs = target_bop_objs
        for obj in (sampled_target_bop_objs):      
            mat = obj.get_materials()[0]     
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            # Blender 4.x renamed the Principled BSDF "Specular" socket.
            if "Specular" in mat.blender_obj.node_tree.nodes["Principled BSDF"].inputs:
                mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            else:
                mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)
                
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                        max_simulation_time=10,
                                                        check_object_interval=1,
                                                        substeps_per_frame = 20,
                                                        solver_iters=25)

        if args.render_model_obj:
            render_template = import_textured_obj(args.render_model_obj)
            align_render_template_to_model(
                render_template,
                target_extent=render_target_extent,
                axis_order=args.render_axis_order,
                base_scale=args.render_scale,
                rotation_deg=args.render_rotation_deg,
            )
            render_template.hide_render = True
            render_template.hide_set(True)
            render_template_name = render_template.name
            for proxy_idx, bp_obj in enumerate(sampled_target_bop_objs):
                proxy = duplicate_render_template(render_template_name, f'render_proxy_{i:06d}_{proxy_idx:03d}')
                proxy.matrix_world = bp_obj.blender_obj.matrix_world.copy()
                render_proxies.append(proxy)
                bp_obj.blender_obj.hide_render = True
        else:
            for bp_obj in sampled_target_bop_objs:
                bp_obj.blender_obj.hide_render = False

        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

        cam_poses = 0
        while cam_poses < 20:
            location = bproc.sampler.shell(center = [0, 0, 0],
                                    radius_min = 0.25,
                                    radius_max = 0.8,
                                    elevation_min = 10,
                                    elevation_max = 80)
            poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=int(round(0.6 * len(obj_ids))), replace=False))
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1
        # 开始渲染
        data = bproc.renderer.render()
        
        # [原有逻辑] 保存 BOP 格式 (保留它以免破坏原始结构)
        bproc.writer.write_bop(bop_parent_path,
                            target_objects = sampled_target_bop_objs,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10,
                            num_worker = int(args.bop_num_worker))
                            
        # ==============================================================
        # [新增逻辑：自定义角点提取与 JSON 保存]
        # ==============================================================
        
        # 确定保存我们自定义标签的文件夹 (例如与 BOP 的 train_pbr 放在同级)
        custom_label_dir = os.path.join(bop_dataset_path, "custom_keypoints")
        os.makedirs(custom_label_dir, exist_ok=True)
        
        scene = bpy.context.scene
        cam = scene.camera
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y

        # 遍历这一批次刚刚渲染出的所有帧 (cam_poses 是相机位姿的数量，这里是 20)
        for frame_idx in range(cam_poses):
            # 将 Blender 时间轴设置到当前帧，这样物体的位姿和相机位置才会更新
            scene.frame_set(frame_idx)
            
            frame_data = {
                "image_name": f"rgb/{frame_idx:06d}.jpg", # 对应 bop 保存的图片路径规则
                "objects": []
            }

            # 遍历当前帧场景中的目标物体
            for bp_obj in sampled_target_bop_objs:
                # 获取底层的 Blender bpy 对象
                blender_obj = bp_obj.blender_obj
                
                # 获取 8 个 3D 角点 (世界坐标)
                bbox_3d = get_3d_bbox(blender_obj)
                
                # 投影到 2D 像素平面
                bbox_2d_pixels = project_3d_to_2d(cam, bbox_3d, res_x, res_y)
                
                frame_data["objects"].append({
                    "obj_id": int(bp_obj.get_cp("category_id")),
                    "keypoints_2d": bbox_2d_pixels
                })
            
            # 为当前帧保存 JSON，名称规则：场景号_帧号_keypoints.json
            # 注意：i 是外层的场景循环变量
            json_filename = os.path.join(custom_label_dir, f"scene_{i:06d}_frame_{frame_idx:06d}_keypoints.json")
            with open(json_filename, 'w') as f:
                json.dump(frame_data, f, indent=4)
        # ==============================================================
        
        for obj in (sampled_target_bop_objs):    
            obj.blender_obj.hide_render = False
            obj.disable_rigidbody()  
            obj.hide(True)
        if render_proxies:
            cleanup_render_proxies(render_proxies)
        if render_template_name is not None:
            cleanup_render_template(render_template_name)
    pass
