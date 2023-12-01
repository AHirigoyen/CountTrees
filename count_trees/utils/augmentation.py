import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(augment):
    if augment:
        transform = A.Compose([
            # A.OneOf([
            #     A.RandomCrop(width=200, height=200, p=0.3),
            #     A.RandomCrop(width=300, height=300, p=0.3),
            #     A.RandomCrop(width=100, height=100, p=0.3),
            # ], p=1),
            A.ShiftScaleRotate(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.6),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"]))
    else:
        # Include any basic non-augmentation transforms here if necessary
        transform = A.Compose([ToTensorV2()],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=["category_ids"]))

    return transform
