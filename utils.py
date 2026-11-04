import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
import ollama
from prompts import BOUNDING_BOX_GENERATOR_SYSTEM_PROMPT
import json

# =============================================================================
# 3D to multi-view 2D
# =============================================================================
#TODO: Clean the code
def _remove_platform_only(tmesh):
    """
    Remove small platforms for some objaverse models
    :param tmesh: Raw mesh
    :return: mesh without platforms
    """
    # Если на входе сцена, собираем все меши
    if isinstance(tmesh, trimesh.Scene):
        geoms = [g for g in tmesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return tmesh
        components = []
        for g in geoms:
            components.extend(g.split(only_watertight=False))
    elif isinstance(tmesh, trimesh.Trimesh):
        components = tmesh.split(only_watertight=False)
    else:
        return tmesh

    if len(components) <= 1:
        return tmesh

    cat_parts = []
    for comp in components:
        extents = comp.extents
        # Платформа характеризуется почти нулевой высотой по Y и малым числом полигонов
        is_platform = (extents[1] < 0.05 * max(extents[0], extents[2])) and (len(comp.faces) < 100)

        if not is_platform:
            cat_parts.append(comp)

    # Объединяем обратно все части кота с сохранением текстур/цветов
    if cat_parts:
        return trimesh.util.concatenate(cat_parts)
    return tmesh

def _convert_trimesh_to_o3d(tmesh):
    """
    Convert Trimesh to Open3D TriangleMesh by preserving the colors
    :param tmesh: trimesh mesh
    :return: o3d mesh
    """

    tmesh = _remove_platform_only(tmesh)


    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)

    # Color transmition
    if hasattr(tmesh.visual, 'to_color'):
        color_visual = tmesh.visual.to_color()
        colors = color_visual.vertex_colors[:, :3] / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    elif hasattr(tmesh.visual, 'vertex_colors') and tmesh.visual.vertex_colors is not None:
        colors = tmesh.visual.vertex_colors[:, :3] / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def render_6_views(mesh_input, output_dir="rendered_views", width=1008, height=1008, camera_distance_multiplyer = 2.2):
    """
    Render mesh from 6 views (front, back, left, right, top, bottom)
    :param mesh_input: path to the .obj mesh / trimesh / open3d
    :param output_dir: output directory
    :param width: render width
    :param height: render height
    :param camera_distance_multiplyer: coefitient that determines the distance between camera and object
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert mesh to o3d mesh
    if isinstance(mesh_input, str):
        mesh = o3d.io.read_triangle_mesh(mesh_input)
    elif isinstance(mesh_input, trimesh.Trimesh):
        mesh = _convert_trimesh_to_o3d(mesh_input)
    elif isinstance(mesh_input, trimesh.Scene):
        tmesh = mesh_input.dump(concatenate=True)
        mesh = _convert_trimesh_to_o3d(tmesh)
    else:
        mesh = mesh_input

    mesh.compute_vertex_normals()

    # Calculate center and camera distance
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_extent()
    dist = extent * camera_distance_multiplyer  # Camera distance

    # Define 6 views
    views = {
        "front":  (center + np.array([0, 0, dist]),   np.array([0, 1, 0])),
        "back":   (center + np.array([0, 0, -dist]),  np.array([0, 1, 0])),
        "left":   (center + np.array([-dist, 0, 0]),  np.array([0, 1, 0])),
        "right":  (center + np.array([dist, 0, 0]),   np.array([0, 1, 0])),
        "top":    (center + np.array([0, dist, 0]),   np.array([0, 0, -1])),
        "bottom": (center + np.array([0, -dist, 0]),  np.array([0, 0, 1]))
    }

    # Open3d configuration

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.scene.set_background(np.array([0.2, 0.2, 0.2, 1.0]))

    # Calculate camera boundaries without image extensions
    half_size = (extent / 2.0) * 1.2  # 1.2 дает 20% свободного места по краям
    aspect = width / height           # Учитываем пропорции картинки (чтобы не сплющило)

    for name, (eye, up) in views.items():
        # Set the camera
        renderer.scene.camera.look_at(center, eye, up)
        # Enable orthographic camera projection to disable perspective and distorion
        renderer.scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho,
            -half_size * aspect, half_size * aspect,  # left, right
            -half_size, half_size,                    # bottom, top
            0.1, dist * 5.0                           # near, far
        )

        # Render images and store
        img = renderer.render_to_image()
        file_path = os.path.join(output_dir, f"{name}.png")
        o3d.io.write_image(file_path, img)

    del renderer

def show_rendered_grid(input_dir="rendered_views"):
    """
    Shows a grid of 6 rendered views for given path
    :param input_dir: directory where multiviews are stored
    :return:
    """
    views = ["front", "back", "left", "right", "top", "bottom"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, view in enumerate(views):
        path = os.path.join(input_dir, f"{view}.png")
        if os.path.exists(path):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(view.capitalize(), fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# VLM-based bounding box generation
# ==============================================================================
def generate_bounding_box_for_image(ollama_host, image_path, user_prompt, model_name = "qwen2.5vl:7b", temperature = 0.1):
    """
    Generates the upper left and bottom right coordinates of bounding box for requested by user object on given image
    :param ollama_host: ip address of ollama
    :param image_path: path to the image. Note: image size should be multiple of 28
    :param user_prompt: object that should be bounded
    :param model_name: name of the vlm model, qwen2.5vl:7b by default
    :param temperature: temperature for vlm predictions
    :return: array of 4 numbers [x1, y1, x2, y2], where x1, y1 - top left coordinate of bounding box and x2, y2 - bottom right coordinate of bounding box
    """
    client = ollama.Client(host=ollama_host)
    system_prompt = (BOUNDING_BOX_GENERATOR_SYSTEM_PROMPT)
    response = client.chat(
        model=model_name,
        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': f"{user_prompt}",
                'images': [image_path]
            }
        ],
        format='json',
        options={'temperature': temperature}
    )
    content = response['message']['content']
    return content



# ==============================================================================
# Image Segmentation
# ==============================================================================
def download_SAM_weights() -> str:
    """
    Download SAM weights.
    :return: path to the SAM weights
    """

    # prepare the folder for weights
    weights_dir = "./weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_filename = "sam_vit_h_4b8939.pth"
    weights_path = os.path.join(weights_dir, weights_filename)

    # download the model weights
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(weights_path):
        # if weights are not installed yet
        torch.hub.download_url_to_file(url, weights_path)
        print("Download of SAM weight complete")
    else:
        print("./weights are already installed")

    return weights_path

def segment_image(image_path: str, x1, y1, x2, y2):
    """
    Generate a mask for specific object in the image inside the bounding box
    :param image_path: path to the image
    :param x1: upper left x coordinate
    :param y1: upper left y coordinate
    :param x2: lower right x coordinate
    :param y2: lower right y coordinate
    :return: boolean mask for specific object in the image
    """
    SAM_weights = download_SAM_weights()
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load SAM to RAM
    sam = sam_model_registry[model_type](checkpoint=SAM_weights)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # setup the current image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device, dtype=torch.float32)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    predictor.set_torch_image(input_image_torch, image.shape[:2])

    # form the box
    input_box = np.array([x1, y1, x2, y2])

    # start segmentation for given image on the bounding box
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False # output one mask instead of many masks
    )

    # masks — numpy array of booleans (1, H, W)
    return masks[0]

def visualize_segmentation(image_path: str, mask, x1, y1, x2, y2):
    """
    Visualize the segmentation mask on the image including bounding box
    :param image_path:
    :param mask:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    image = cv2.imread(image_path)
    print("Image size (H, W, C):", image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)  # original image

    # apply the mask
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = [1.0, 0.0, 0.0, 0.5]  # Color of the mask
        plt.imshow(overlay)

    # Show bounding box
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      edgecolor='yellow', facecolor='none', linewidth=2))

    plt.title("Output of SAM including bounding box")
    plt.axis('off')
    plt.show()


import numpy as np
from PIL import Image
import trimesh.visual


def project_masks_to_3d(mesh, masks_dict, width=800, height=800):
    """
    Projects 2D segmentation masks onto a 3D mesh using back-projection.

    :param mesh: trimesh.Trimesh or Open3D TriangleMesh object
    :param masks_dict: Dictionary mapping view names to masks, e.g., {'front': mask_array, ...}
    :param width: Image width used during rendering
    :param height: Image height used during rendering
    :return: (colored_mesh, mask_submesh)
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals

    # Universal center and extent extraction for both Open3D and Trimesh
    if hasattr(mesh, 'get_axis_aligned_bounding_box'):
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()
    else:
        # For trimesh.Trimesh
        center = mesh.bounds.mean(axis=0)     # AABB center
        extent = float(mesh.extents.max())    # AABB max extent

    dist = extent * 2.2
    half_size = (extent / 2.0) * 1.2
    aspect = width / height

    # Define camera origins and up-vectors for 6 orthogonal views
    views = {
        "front":  (center + np.array([0, 0, dist]),   np.array([0, 1, 0])),
        "back":   (center + np.array([0, 0, -dist]),  np.array([0, 1, 0])),
        "left":   (center + np.array([-dist, 0, 0]),  np.array([0, 1, 0])),
        "right":  (center + np.array([dist, 0, 0]),   np.array([0, 1, 0])),
        "top":    (center + np.array([0, dist, 0]),   np.array([0, 0, -1])),
        "bottom": (center + np.array([0, -dist, 0]),  np.array([0, 0, 1]))
    }

    vertex_masked = np.zeros(len(vertices), dtype=bool)
    rel_verts = vertices - center

    for name, (eye, up) in views.items():
        if name not in masks_dict:
            continue

        mask_img = masks_dict[name]
        if isinstance(mask_img, Image.Image):
            mask_arr = np.array(mask_img) > 128
        else:
            mask_arr = mask_img > 0

        # Construct camera coordinate basis
        z_cam = (eye - center) / np.linalg.norm(eye - center)
        x_cam = np.cross(up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)

        # 1. Back-face culling (verify normal orientation relative to the camera)
        visible_mask = np.dot(normals, z_cam) > 0.1

        # 2. Project 3D vertices to 2D pixel coordinates
        x_proj = np.dot(rel_verts, x_cam)
        y_proj = np.dot(rel_verts, y_cam)

        u_ndc = x_proj / (half_size * aspect)
        v_ndc = y_proj / half_size

        px = np.clip(np.round((u_ndc + 1.0) / 2.0 * (width - 1)).astype(int), 0, width - 1)
        py = np.clip(np.round((1.0 - v_ndc) / 2.0 * (height - 1)).astype(int), 0, height - 1)

        # 3. Sample mask at projected pixel locations
        sampled_mask = mask_arr[py, px]
        vertex_masked |= (visible_mask & sampled_mask)

    # --- 3D Model Coloring ---
    colored_mesh = mesh.copy()
    colors = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
    colors[vertex_masked] = [0, 255, 0, 255]  # Green for masked vertices

    # --- Extract Masked 3D Submesh ---
    masked_faces = vertex_masked[mesh.faces].any(axis=1)
    mask_submesh = mesh.submesh([masked_faces], append=True)

    return colored_mesh, mask_submesh