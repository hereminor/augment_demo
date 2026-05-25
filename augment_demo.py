from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# CONFIGURATIONS
SEED = 67

# Only available for previous version lol
# Affine rotate
AFFINE_ROTATE_LIMIT = (-15, 15)

# Affine scale
AFFINE_SCALE_LIMIT = (0.9, 1.1)

# Affine translate_percent
AFFINE_TRANSLATE_LIMIT = (-0.05, 0.05)

# Affine shear
AFFINE_SHEAR_LIMIT = (-5, 5)

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# GridDistortion
GRID_STEPS = 5
GRID_DISTORT_LIMIT = (-0.3, 0.3)

# ElasticTransform
ELAS_ALPHA = 120
ELAS_SIGMA = 6
ELAS_ALPHA_AFFINE = 3.6

# RandomBrightnessContrast
BRIGHT_LIMIT = (-0.2, 0.2)
BRIGHT_CONTRAST_LIMIT = (-0.2, 0.2)

# RandomGamma
GAMMA_LIMIT = (40, 160)

boxes = [
    (0,0,512,512),
    (512,0,1024,512),
    (0,512,512,1024),
    (512,512,1024,1024)
]

aug_list = [
    "none",                 # 0: None
    "Augmentation_01",      # 1: The Strict Orientation Baseline
    "Augmentation_02",      # 2: The "Capillary Popper"
    "Augmentation_03",      # 3: Subtle Contrast & Orientation
    "Augmentation_04",      # 4: The Saccade Simulator
    "Augmentation_05",      # 5: Vessel Caliber Variation
    "Augmentation_06",      # 6: The Topology Stress Test
    "Augmentation_07",      # 7: High Contrast + Saccade Simulator
    "Augmentation_08",      # 8: The Conservative Comprehensive
    "Augmentation_09",      # 9: AggressiveGeometric, Raw Pixels
    "Augmentation_10"       # 10: The Original "Kitchen Sink"
]

# interpolation=cv2.INTER_LINEAR,
# mask_interpolation=cv2.INTER_NEAREST,
# border_mode=cv2.BORDER_REFLECT_101,
# fill=0,
# fill_mask=0,

aug_dict = {
    "none": [[A.HorizontalFlip(p=0.0)], "None"],
    "Augmentation_01": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
        ], "The Strict Orientation Baseline"],
    "Augmentation_02": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CLAHE(
            clip_limit=3.0,
            tile_grid_size=(8,8),
            p=1.0)
        ], "The \"Capillary Popper\""],
    "Augmentation_03": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CLAHE(
            clip_limit=1.5,
            tile_grid_size=(8,8),
            p=0.5)
        ], "Subtle Contrast & Orientation"],
    "Augmentation_04": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        translate_percent=(-0.05, 0.05),
        rotate=(-10, 10),
        scale=(1.0, 1.0),
        shear=(0, 0),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.75)
        ], "The Saccade Simulator"],
    "Augmentation_05": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        scale=(0.95, 1.05),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.5)
        ], "Vessel Caliber Variation"],
    "Augmentation_06": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        shear=(-5, 5),
        p=0.5)
        ], "The Topology Stress Test"],
    "Augmentation_07": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CLAHE(
            clip_limit=3.0,
            tile_grid_size=(8,8),
            p=1.0),
        A.Affine(
        translate_percent=(-0.05, 0.05),
        rotate=(-10, 10),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.75)
        ], "High Contrast + Saccade Simulator"],
    "Augmentation_08": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        translate_percent=(-0.05, 0.05),
        rotate=(-10, 10),
        scale=(1.0, 1.0),
        shear=(0, 0),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.3),
        A.CLAHE(
            clip_limit=1.5,
            tile_grid_size=(8,8),
            p=0.5)
        ], "The Conservative Comprehensive"],
    "Augmentation_09": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.1, 0.1),
        rotate=(-15, 15),
        shear=(0, 0),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.8)
        ], "AggressiveGeometric, Raw Pixels"],
    "Augmentation_10": [[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
        scale=AFFINE_SCALE_LIMIT,
        translate_percent=AFFINE_TRANSLATE_LIMIT,
        rotate=AFFINE_ROTATE_LIMIT,
        shear=AFFINE_SHEAR_LIMIT,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=0.75,
        ),
        A.CLAHE(
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
        p=0.5)
        ], "The Original \"Kitchen Sink\""]
}

def process(index_list, dataset, augmentation="none"):
    aug_name = aug_dict[augmentation][1]
    Path(f"processed/{dataset}/{augmentation}").mkdir(parents=True, exist_ok=True)

    print(50*"=")
    print(f"Process: {aug_name}")
    print(50*"=")

    for index in index_list:
        print(f"Processing image {index}")
        if dataset != 'drac':
            raw_im = Image.open(f"dataset_{dataset}/raw/{index}.jpg")
        else:
            raw_im = Image.open(f"dataset_{dataset}/raw/{index}.png")
        gt_im = Image.open(f"dataset_{dataset}/gt/{index}.jpg")

        raw_patches = []
        gt_patches = []
        print(f"Cropping image")
        for j, i in enumerate(boxes):
            print(f"Augmenting box #{j+1}")
            aug_im, aug_ma = apply_aug(raw_im.crop(i), gt_im.crop(i), augmentation)
            raw_patches.append(aug_im)
            gt_patches.append(aug_ma)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for j in range(4):
            x, y = j % 2, j // 2
            axes[y,x].imshow(raw_patches[j], cmap="gray")
            axes[y,x].set_title(f"Raw ({x},{y})")
            axes[y,x+2].imshow(gt_patches[j], cmap="gray")
            axes[y,x+2].set_title(f"GT ({x},{y})")

        for ax in axes.flatten():
            ax.axis("off")

        print(f"Saving combined images")
        fig.savefig(f"processed/{dataset}/{augmentation}/{index}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        if augmentation != "none":
            def_image = Image.open(f"processed/{dataset}/None/{index}.png")
            temp_image = Image.open(f"processed/{dataset}/{augmentation}/{index}.png")
            sequence = [def_image, temp_image]

            sum_fig = plt.figure(layout='constrained')
            sum_fig.suptitle(aug_name)
            sub_fig = sum_fig.subfigures(2,1)

            def_fig = sub_fig[0].subplots()
            def_fig.imshow(def_image)
            def_fig.axis("off")

            aug_fig = sub_fig[1].subplots()
            aug_fig.imshow(temp_image)
            aug_fig.axis("off")

            print("Saving summary image")
            sum_fig.savefig(f"processed/{dataset}/{augmentation}/{index}_summary.png", dpi=300, bbox_inches="tight")
            plt.close(sum_fig)

            print("Saving gif")
            sequence[0].save(
                f"processed/{dataset}/{augmentation}/{index}.gif",
                append_images=sequence[1:],
                duration=500,
                loop=0,
            )

def apply_aug(im, ma, augmentation):
    transform_temp, transform_name = aug_dict[augmentation]
    transform = A.Compose(transform_temp, seed=SEED)

    im_array = np.asarray(im, dtype=np.uint8)
    ma_array = np.asarray(ma, dtype=np.uint8)

    result = transform(image=im_array, mask=ma_array)

    return Image.fromarray(result["image"]), Image.fromarray(result["mask"])

def main():
    Path("processed/tu").mkdir(parents=True, exist_ok=True)
    Path("processed/drac").mkdir(parents=True, exist_ok=True)

    print(f"SEED: {SEED}")

    index_list = [8,13,20]
    for augmentation in aug_list:
        process(index_list, "tu", augmentation)

main()