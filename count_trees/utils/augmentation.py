import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(augment):
        """This is the new transform"""
        if augment:
            trans = A.Compose([
                A.RandomCrop(width=380, height=380),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
            
            n = 5
            transforms = [trans for _ in range(n)]
            transform = A.Compose(transforms)
            transform = ToTensorV2()
            
        else:
            transform = ToTensorV2()
            
        return transform