import torch
from torch.utils.data import Dataset
import objaverse
import trimesh
import os

class ObjaverseDataset(Dataset):
    def __init__(self, uids: list = None, category: str = None, limit: int = 20):
        """
        Dataset to load from objaverse dataset
        :param uids: ids of samples
        :param category: category of sample if uids is None
        :param limit: How many samples to load
        """
        self.uids = uids
        self.category = category
        self.limit = limit
        # if ids are not provided
        if uids is None:
            lvis_annotations = objaverse.load_lvis_annotations()
            # either take from specified category
            if category and category in lvis_annotations:
                self.uids = lvis_annotations[category][:limit]
            # or take first limit samples
            else:
                all_ids = objaverse.load_uids()
                self.uids = all_ids[:limit]
        # load paths for objects
        self.object_paths = objaverse.load_objects(uids=self.uids)
        # again extract ids from objects to protect against false uids from user
        # and to synchronize object paths and their uids
        self.uid_list = list(self.object_paths.keys())

    def __len__(self):
        return len(self.uid_list)

    def __getitem__(self, index):
        uid = self.uid_list[index]
        file_path = self.object_paths[uid]
        # load 3d .glm object in RAM
        mesh = trimesh.load(file_path, force = 'mesh')

        # mesh normalization
        mesh.vertices -= mesh.center_mass
        max_dim = max(mesh.extents)
        if max_dim > 0:
            mesh.vertices /= max_dim

        # for backup store mesh in .obj format
        obj_dir = "./data_cache/obj_models"
        os.makedirs(obj_dir, exist_ok=True)
        obj_path = os.path.join(obj_dir, f"{uid}.obj")

        if not os.path.exists(obj_path):
            mesh.export(obj_path)

        return {
            "uid": uid,
            "mesh": mesh,  # actual object
            "obj_path": obj_path,  # path to file in cache
            "vertices": torch.tensor(mesh.vertices, dtype=torch.float32),
            "faces": torch.tensor(mesh.faces, dtype=torch.long)
        }



