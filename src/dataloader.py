from torch.utils.data import DataLoader
from configs import config
from src.dataset import CustomDataset
from torchvision.transforms import ToTensor,Compose,Normalize,RandomAffine,ColorJitter


# vì mỗi ảnh đối tượng và nhãn khác nhau  không cùng kích thước mà hàm default_collate mặc định
# sẽ chỉ biết chia dựa trên kích thước bằng nhau nên ta phải tự xây dựng
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def hard_hat_workers_dataloader(root,batch_size,num_workers=config.num_worker):
    transform_train = Compose([
        # bỏ vì khi xoay ảnh chúng ta cần phải mang nhãn xoay theo
        # RandomAffine(
        #     degrees=(-5, 5),
        #     translate=(0.15, 0.15),
        #     scale=(0.85, 1.15),
        #     shear=10
        # ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        ToTensor(),
        # bỏ vì trong model faster cnn đã có chuẩn hóa
        # Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    transform_val = Compose([
        ToTensor(),
        # bỏ vì trong model faster cnn đã có chuẩn hóa
        # Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    #data train
    train_dataset = CustomDataset(root=root,is_train=True,transform=transform_train)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # data val
    val_dataset = CustomDataset(root=root,is_train=False,transform=transform_val)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader

