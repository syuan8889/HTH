from safetensors.torch import save_file
import torch
# if raise error that ".bin" model files not safe, run this script.
state_dict = torch.load("/HTH/clip-models/clip_b_ft/pytorch_model.bin", map_location="cpu", weights_only=False)
save_file(state_dict, "/HTH/clip-models/clip_b_ft/pytorch_model.safetensors")
