import os
import torch

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