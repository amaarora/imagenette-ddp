import albumentations
from torchvision import transforms


IMG_SIZE = 160

Config = dict(
    DATA_DIR="../data/imagenette2-160",
    TRAIN_DATA_DIR="../data/imagenette2-160/train",
    TEST_DATA_DIR="../data/imagenette2-160/val",
    DEVICE="cuda",
    MODEL="efficientnet_b0",
    PRETRAINED=False,
    LR=1e-3,
    EPOCHS=10,
    IMG_SIZE=IMG_SIZE,
    BS=64,
    TRAIN_AUG=transforms.Compose(
        [
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    TEST_AUG=transforms.Compose(
        [
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
)
