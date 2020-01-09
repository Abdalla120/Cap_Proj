import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


import numpy as np

from src.modeling.run_model_single import (
    load_model, load_inputs, process_augment_inputs, batch_to_tensor
)
import src.utilities.pickling as pickling

shared_parameters = {
    "device_type": "gpu",
    "gpu_number": 0,
    "max_crop_noise": (100, 100),
    "max_crop_size_noise": 100,
    "batch_size": 1,
    "seed": 0,
    "augmentation": True,
    "use_hdf5": True,
}
random_number_generator = np.random.RandomState(shared_parameters["seed"])

image_heatmaps_parameters = shared_parameters.copy()
image_heatmaps_parameters["view"] = "R-CC"
image_heatmaps_parameters["use_heatmaps"] = True
image_heatmaps_parameters["model_path"] = "models/ImageHeatmaps__ModeImage_weights.p"
model, device = load_model(image_heatmaps_parameters)

model_input = load_inputs(
    image_path="sample_single_output/cropped.png",
    metadata_path="sample_single_output/cropped_metadata.pkl",
    use_heatmaps=True,
    benign_heatmap_path="sample_single_output/benign_heatmap.hdf5",
    malignant_heatmap_path="sample_single_output/malignant_heatmap.hdf5",
)
batch = [
    process_augment_inputs(
        model_input=model_input,
        random_number_generator=random_number_generator,
        parameters=image_heatmaps_parameters,
    ),
]
tensor_batch = batch_to_tensor(batch, device)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x = tensor_batch[0].cpu().numpy()

axes[0].imshow(x[0], cmap="gray")
axes[0].set_title("Image")

axes[1].imshow(x[1], cmap=LinearSegmentedColormap.from_list("benign", [(0, 0, 0), (0, 1, 0)]))
axes[1].set_title("Benign Heatmap")

axes[2].imshow(x[2], cmap=LinearSegmentedColormap.from_list("malignant", [(0, 0, 0), (1, 0, 0)]))
axes[2].set_title("Malignant Heatmap")
plt.show()
