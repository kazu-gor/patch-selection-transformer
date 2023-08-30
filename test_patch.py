from random import sample
import torch
import pandas as pd

from lib.dataloader import get_loader


def target_class(targets):
    return torch.where(
        torch.einsum("ijklm->ij", targets) > 0,
        torch.tensor(1),
        torch.tensor(0),
    )

def main():
    # define dataset root path
    image_root_train = '../../dataset/TrainDataset/images/'
    image_root_val = '../../dataset/ValDataset/images/'
    image_root_test = '../../dataset/TestDataset/images/'

    gt_root_train = '../../dataset/TrainDataset/masks/'
    gt_root_val = '../../dataset/ValDataset/masks/'
    gt_root_test = '../../dataset/TestDataset/masks/'

    # define dataloader
    train_loader = get_loader(image_root_train,
                              gt_root_train,
                              shuffle=False,
                              batchsize=1,
                              trainsize=352,
                              patch_size=44,
                              phase='val')
    val_loader = get_loader(image_root_val,
                            gt_root_val,
                            shuffle=False,
                            batchsize=1,
                            trainsize=352,
                            patch_size=44,
                            phase='val')
    test_loader = get_loader(image_root_test,
                             gt_root_test,
                             shuffle=False,
                             batchsize=1,
                             trainsize=352,
                             patch_size=44,
                             phase='val')

    dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Load each dataloaders_dict
    num_images = 0
    idx_s1 = 0
    idx_s2 = 0
    idx_s3 = 0
    # num_target_s1 = 0
    # num_target_s2 = 0
    # num_target_s3 = 0
    num_target_s1 = list()
    num_target_s2 = list()
    num_target_s3 = list()
    for phase, data_loader in dataloaders_dict.items():
        for i, (samples, target_s1, target_s2, target_s3) in enumerate(data_loader):
            print(f"{phase = }, {i = }")
            num_images += 1

            # target_s1 = torch.einsum("ijklm->ij", target_s1)
            # target_s2 = torch.einsum("ijklm->ij", target_s2)
            # target_s3 = torch.einsum("ijklm->ij", target_s3)

            target_s1 = target_class(target_s1)
            target_s2 = target_class(target_s2)
            target_s3 = target_class(target_s3)

            print(samples)
            print(target_s1.view(8, 8))
            print(target_s2.view(4, 4))
            print(target_s3.view(2, 2))

            idx_s1 += target_s1.shape[1]
            idx_s2 += target_s2.shape[1]
            idx_s3 += target_s3.shape[1]

            num_target_s1.append(torch.sum(target_s1).item())
            num_target_s2.append(torch.sum(target_s2).item())
            num_target_s3.append(torch.sum(target_s3).item())

    df_num_target_s1 = pd.DataFrame(num_target_s1)
    df_num_target_s2 = pd.DataFrame(num_target_s2)
    df_num_target_s3 = pd.DataFrame(num_target_s3)

    print(f"{df_num_target_s1.describe() = }")
    print(f"{df_num_target_s2.describe() = }")
    print(f"{df_num_target_s3.describe() = }")

    print(f"{df_num_target_s1.value_counts() = }")
    print(f"{df_num_target_s2.value_counts() = }")
    print(f"{df_num_target_s3.value_counts() = }")

    # Find the mean and distribution of each num_target
    print(f"{num_images = }")
    print(f"{idx_s1 = }")
    print(f"{idx_s2 = }")
    print(f"{idx_s3 = }")
    print(f"{sum(num_target_s1) / num_images = }")
    print(f"{sum(num_target_s2) / num_images = }")
    print(f"{sum(num_target_s3) / num_images = }")


if __name__ == '__main__':
    main()
