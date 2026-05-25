from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# CONFIGURATIONS

SEED = 67

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
    "none",         # 0: None
    "hflip",        # 1: Horizontalflip
    "a_scale",      # 2: Affine scale
    "a_trans",      # 3: Affine translate_percent
    "a_rot",        # 4: Affine rotate
    "a_shear",      # 5: Affine shear
    "clahe",        # 6: CLAHE
    "griddist",     # 7: GridDistortion
    "elastrans",    # 8: ElasticTransform
    "randbright",   # 9: RandomBrightnessContrast
    "randgamma"     # 10: RandomGamma
]

# interpolation=cv2.INTER_LINEAR,
# mask_interpolation=cv2.INTER_NEAREST,
# border_mode=cv2.BORDER_REFLECT_101,
# fill=0,
# fill_mask=0,

aug_dict = {
    "none": [A.HorizontalFlip(p=0.0), "None"],
    "hflip": [A.HorizontalFlip(p=1.0), "HorizontalFlip"],
    "a_scale": [A.Affine(
        scale=AFFINE_SCALE_LIMIT,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=1.0
    ), "Affine_scale"],
    "a_trans": [A.Affine(
        translate_percent=AFFINE_TRANSLATE_LIMIT,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=1.0
    ), "Affine_translate_percent"],
    "a_rot": [A.Affine(
        rotate=AFFINE_ROTATE_LIMIT,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=1.0
    ), "Affine_rotate"],
    "a_shear": [A.Affine(
        shear=AFFINE_SHEAR_LIMIT,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        fill=0,
        fill_mask=0,
        p=1.0
    ), "Affine_shear"],
    "clahe": [A.CLAHE(
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
        p=1.0
    ), "CLAHE"],
    "griddist": [A.GridDistortion(num_steps=GRID_STEPS, distort_limit=GRID_DISTORT_LIMIT, p=1.0), "GridDistortion"],
    "elastrans": [A.ElasticTransform(alpha=ELAS_ALPHA, sigma=ELAS_SIGMA, alpha_affine=ELAS_ALPHA_AFFINE, p=1.0), "ElasticTransform"],
    "randbright": [A.RandomBrightnessContrast(brightness_limit=BRIGHT_LIMIT,contrast_limit=BRIGHT_CONTRAST_LIMIT, p=1.0), "RandomBrightnessContrast"],
    "randgamma": [A.RandomGamma(gamma_limit=GAMMA_LIMIT, p=1.0), "RandomGamma"]
}

def process(index_list, dataset, augmentation="none"):
    aug_name = aug_dict[augmentation][1]
    Path(f"processed/{dataset}/{aug_name}").mkdir(parents=True, exist_ok=True)

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
        fig.savefig(f"processed/{dataset}/{aug_name}/{index}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        if augmentation != "none":
            def_image = Image.open(f"processed/{dataset}/None/{index}.png")
            temp_image = Image.open(f"processed/{dataset}/{aug_name}/{index}.png")
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
            sum_fig.savefig(f"processed/{dataset}/{aug_name}/{index}_summary.png", dpi=300, bbox_inches="tight")
            plt.close(sum_fig)

            print("Saving gif")
            sequence[0].save(
                f"processed/{dataset}/{aug_name}/{index}.gif",
                append_images=sequence[1:],
                duration=500,
                loop=0,
            )

def apply_aug(im, ma, augmentation):
    transform_temp, transform_name = aug_dict[augmentation]
    transform = A.Compose([transform_temp], seed=SEED)

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