import torch
from torch.utils.data import Dataset
import objaverse


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
        # and their ids ???
        self.uid_list = list(self.object_paths.keys())

    def __len__(self):
        return len(self.uid_list)

    # def __getitem__(self, id):


