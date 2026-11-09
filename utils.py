import os
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import ollama
from prompts import BOUNDING_BOX_GENERATOR_SYSTEM_PROMPT
import json
import trimesh.visual

import numpy as np
import trimesh
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
    """Convert Trimesh to Open3D TriangleMesh by preserving the colors"""

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


def render_n_views(mesh_input, n_views=6, output_dir="rendered_views", width=1008, height=1008, views=None):
    """
    Generates N images evenly distributed on a sphere around the mesh.
    Returns a list of tuples formatted as [(eye_pos, file_path), ...].

    Parameters:
        - mesh_input: file path (.obj), trimesh.Trimesh, trimesh.Scene, or o3d.geometry.TriangleMesh
        - n_views: number of viewpoints to generate if views is None
        - output_dir: directory to save rendered frames
        - width, height: resolution of output images
        - views: dict of viewpoints formatted as {"name": (eye_np_array, up_np_array)}.
                 If None, n_views camera positions will be calculated using Fibonacci Sphere algorithm.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert input object to Open3D TriangleMesh
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

    # Automatically calculate center and camera offset distance
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_extent()
    dist = extent * 2.2  # Camera offset distance

    # Generate N evenly spaced camera views using Fibonacci Sphere algorithm
    if views is None:
        views = {}
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

        for i in range(n_views):
            # Calculate y-coordinate along the sphere axis from 1 to -1
            y = 1.0 - (i / float(n_views - 1)) * 2.0 if n_views > 1 else 0.0
            radius = np.sqrt(1.0 - y * y)  # Radius at height y
            theta = phi * i  # Golden angle increment

            # Calculate direction on sphere
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            direction = np.array([x, y, z])
            eye = center + direction * dist

            # Set correct up-vector (avoid singular vector alignment near poles)
            if np.abs(y) > 0.99:
                up = np.array([0.0, 0.0, -1.0]) if y > 0 else np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])

            views[f"view_{i + 1:03d}"] = (eye, up)

    # Setup Open3D offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.scene.set_background(np.array([0.2, 0.2, 0.2, 1.0]))

    half_size = (extent / 2.0) * 1.2
    aspect = width / height

    rendered_results = []

    for name, (eye, up) in views.items():
        # 1. Point the camera
        renderer.scene.camera.look_at(center, eye, up)

        # 2. Set orthographic projection
        renderer.scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho,
            -half_size * aspect, half_size * aspect,
            -half_size, half_size,
            0.1, dist * 5.0
        )

        # Render and save image
        img = renderer.render_to_image()
        file_path = os.path.join(output_dir, f"{name}.png")
        o3d.io.write_image(file_path, img)

        # Append tuple (camera_position, file_path) to results
        rendered_results.append((eye, file_path))

    del renderer  # Free resources

    return rendered_results


import os
import math
import matplotlib.pyplot as plt
from PIL import Image


def show_rendered_grid(input_dir="rendered_views", n_views=None, cols=4):
    """
    Displays a grid of N rendered view images.

    Parameters:
        - input_dir: Directory where rendered view images are stored
        - n_views: Maximum number of images to display (if None, displays all found images)
        - cols: Number of columns in the grid layout (default is 4)
    """
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    # Collect and sort all image files in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No rendered images found in directory {input_dir}.")
        return

    # Limit the number of images if n_views is specified
    if n_views is not None:
        image_files = image_files[:n_views]

    total_views = len(image_files)

    # Automatically calculate rows and columns count
    cols = min(total_views, cols)
    rows = math.ceil(total_views / cols)

    # Create Matplotlib figure with dynamic size based on grid dimensions
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    # Flatten axes array to ensure consistent 1D indexing
    if total_views == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Draw each rendered image onto the grid
    for i, file_name in enumerate(image_files):
        path = os.path.join(input_dir, file_name)
        img = Image.open(path)

        axes[i].imshow(img)
        title = os.path.splitext(file_name)[0].replace('_', ' ').capitalize()
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    # Turn off remaining empty subplots if total_views is not divisible by cols
    for j in range(total_views, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# VLM-based bounding box generation
# ==============================================================================
def generate_bounding_box_for_image(
    ollama_host,
    image_path,
    user_prompt,
    model_name="qwen3-vl:8b",
    temperature=0.1,
):
    """Detect a requested part and return a JSON string with PIXEL coordinates.

    Output: {"is_found": bool, "bbox_2d": [xmin, ymin, xmax, ymax] | None}.
    Qwen3-VL returns coordinates normalized to [0, 1000]; Qwen2.5-VL
    is prompted for absolute pixel coordinates. A missing target returns
    is_found=False. Invalid model output raises an exception instead of
    silently being treated as a missing target.
    """
    import math

    with Image.open(image_path) as image:
        width, height = image.size

    name = model_name.lower()
    is_qwen3 = "qwen3" in name
    is_qwen25 = "qwen2.5" in name
    if not (is_qwen3 or is_qwen25):
        raise ValueError(
            f"Unknown coordinate convention for model {model_name!r}. "
            "Specify the model's bbox coordinate system before using it."
        )

    if is_qwen3:
        system_prompt = BOUNDING_BOX_GENERATOR_SYSTEM_PROMPT
    else:
        # The shared project prompt requests normalized coordinates.
        # Qwen2.5-VL is instead prompted for absolute image pixels.
        system_prompt = BOUNDING_BOX_GENERATOR_SYSTEM_PROMPT.replace(
            "Coordinates must be integers normalized to [0, 1000],",
            f"Coordinates must be absolute pixel integers for a {width}x{height} image,"
        )

    client = ollama.Client(host=ollama_host)
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(user_prompt), "images": [str(image_path)]},
        ],
        format="json",
        options={"temperature": temperature},
    )

    content = response["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"{model_name} returned an empty response for {image_path!s}; "
            f"done_reason={response.get('done_reason')!r}"
        )

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{model_name} returned invalid JSON for {image_path!s}: {content[:500]!r}"
        ) from exc

    if not isinstance(result, dict) or type(result.get("is_found")) is not bool:
        raise ValueError(f"Invalid is_found from {model_name}: {result!r}")

    if not result["is_found"]:
        return json.dumps({"is_found": False, "bbox_2d": None})

    bbox = result.get("bbox_2d")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox_2d from {model_name}: {bbox!r}")

    try:
        if any(isinstance(v, bool) for v in bbox):
            raise ValueError("Boolean coordinate")
        coords = [float(v) for v in bbox]
        if not all(math.isfinite(v) for v in coords):
            raise ValueError("Non-finite coordinate")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bbox coordinates from {model_name}: {bbox!r}") from exc

    if is_qwen3:
        # The model reports normalized coordinates, not image pixels.
        if not all(0 <= v <= 1000 for v in coords):
            raise ValueError(f"Qwen3 bbox outside normalized [0, 1000] range: {bbox!r}")
        coords = [
            coords[0] * width / 1000,
            coords[1] * height / 1000,
            coords[2] * width / 1000,
            coords[3] * height / 1000,
        ]

    x1, y1, x2, y2 = coords
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Inverted or empty bbox from {model_name}: {bbox!r}")

    # Clamp round-off / slightly out-of-frame detections to image dimensions.
    x1 = max(0, min(width - 1, round(x1)))
    y1 = max(0, min(height - 1, round(y1)))
    x2 = max(0, min(width - 1, round(x2)))
    y2 = max(0, min(height - 1, round(y2)))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Bounding box collapsed after pixel conversion: {bbox!r}")

    return json.dumps({"is_found": True, "bbox_2d": [x1, y1, x2, y2]})


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

def project_masks_to_3d(mesh, masks_data, width=1008, height=1008, min_views=1, normal_thresh=0.3):
    """
    Projects 2D masks onto a 3D trimesh model using camera parameters with Z-Buffer depth testing.

    Parameters:
        mesh: trimesh.Trimesh object
        masks_data: list or dict containing mask and camera positions.
        width, height: Resolution of the rendered views
        min_views: Minimum view hits required to mark vertex (set to 2 for strict voting)
        normal_thresh: Threshold for back-face culling (0.3 eliminates tangential angles)

    Returns:
        colored_mesh: trimesh.Trimesh with green highlighted masked vertices
        mask_submesh: trimesh.Trimesh submesh of the masked region
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = mesh.extents.max()

    half_size = (extent / 2.0) * 1.2
    aspect = width / height

    vertex_hit_count = np.zeros(len(vertices), dtype=int)
    rel_verts = vertices - center

    # Unify masks_data into a list of tuples: (mask, eye, up)
    items = []
    if isinstance(masks_data, dict):
        for val in masks_data.values():
            if isinstance(val, dict):
                mask = val["mask"]
                eye = np.array(val["eye"], dtype=float)
                up = np.array(val["up"], dtype=float) if "up" in val and val["up"] is not None else None
            elif isinstance(val, (tuple, list)):
                mask = val[0]
                eye = np.array(val[1], dtype=float)
                up = np.array(val[2], dtype=float) if len(val) > 2 else None
            items.append((mask, eye, up))
    elif isinstance(masks_data, (list, tuple)):
        for val in masks_data:
            if isinstance(val, dict):
                mask = val["mask"]
                eye = np.array(val["eye"], dtype=float)
                up = np.array(val["up"], dtype=float) if "up" in val and val["up"] is not None else None
            elif isinstance(val, (tuple, list)):
                mask = val[0]
                eye = np.array(val[1], dtype=float)
                up = np.array(val[2], dtype=float) if len(val) > 2 else None
            items.append((mask, eye, up))

    for mask_img, eye, up in items:
        # Convert mask to boolean numpy array
        if isinstance(mask_img, Image.Image):
            mask_arr = np.array(mask_img) > 128
        else:
            mask_arr = mask_img > 0

        # Camera orientation basis
        z_cam = (eye - center) / np.linalg.norm(eye - center)

        # Auto-detect 'up' vector if missing
        if up is None:
            if np.abs(z_cam[1]) > 0.99:  # Top or Bottom view
                up = np.array([0.0, 0.0, -1.0]) if z_cam[1] > 0 else np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])

        x_cam = np.cross(up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)

        # 1. Back-face culling (усилен с 0.1 до 0.3)
        visible_mask = np.dot(normals, z_cam) > normal_thresh

        # 2. Project 3D coordinates to NDC & pixels
        x_proj = np.dot(rel_verts, x_cam)
        y_proj = np.dot(rel_verts, y_cam)

        u_ndc = x_proj / (half_size * aspect)
        v_ndc = y_proj / half_size

        px = np.clip(np.round((u_ndc + 1.0) / 2.0 * (width - 1)).astype(int), 0, width - 1)
        py = np.clip(np.round((1.0 - v_ndc) / 2.0 * (height - 1)).astype(int), 0, height - 1)

        # 3. Z-Buffer Occlusion Check (ПРОВЕРКА НА ПЕРЕКРЫТИЕ ГЛУБИНЫ)
        depths = np.dot(rel_verts, z_cam)
        z_buffer = np.full((height, width), fill_value=np.inf)
        vis_indices = np.where(visible_mask)[0]

        # Находим наименьшую глубину (ближайшую поверхность) для каждого пикселя
        for idx in vis_indices:
            r_y, r_x = py[idx], px[idx]
            if depths[idx] < z_buffer[r_y, r_x]:
                z_buffer[r_y, r_x] = depths[idx]

        # Вершина передняя, если её глубина совпадает с минимальной глубиной в пикселе
        eps = extent * 0.03  # Допуск по глубине
        is_front_surface = np.abs(depths - z_buffer[py, px]) < eps

        # 4. Sample mask
        sampled_mask = mask_arr[py, px]

        # Накапливаем совпадения только для передних видимых вершин
        vertex_hit_count += (visible_mask & is_front_surface & sampled_mask).astype(int)

    # Применяем фильтр по количеству ракурсов
    vertex_masked = vertex_hit_count >= min_views

    # --- Color 3D Mesh ---
    colored_mesh = mesh.copy()
    colors = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
    colors[vertex_masked] = [0, 255, 0, 255]
    colored_mesh.visual.vertex_colors = colors

    # --- Extract Mask Submesh ---
    masked_faces = vertex_masked[mesh.faces].any(axis=1)
    mask_submesh = mesh.submesh([masked_faces], append=True)

    return colored_mesh, mask_submesh
