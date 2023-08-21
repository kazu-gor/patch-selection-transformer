import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, alpha=0.55, image_size=352):
        super(Criterion, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss()
        self.image_size = image_size

    def forward(self, preds, targets):
        # iwasaki comment [2023-08-08 11:37:08] plane labels
        targets_labels = torch.where(
            torch.einsum("ijklm->ij", targets) > 0,
            torch.tensor(1).cuda(),
            torch.tensor(0).cuda(),
        )
        preds = torch.sigmoid(preds)
        label_loss = self.bce(preds, targets_labels.float())
        label_loss = label_loss.sum() / torch.numel(label_loss)

        # # iwasaki comment [2023-08-08 11:37:08] area labels
        patch_pixel = targets.shape[-1] ** 2
        calcification_area = torch.einsum('ijklm->ij', targets)
        perc_of_calc = calcification_area / float(patch_pixel)
        non_zero_indices = torch.nonzero(perc_of_calc)
        # # iwasaki comment [2023-08-08 15:10:08] slicing nonzero values
        target_sliced = perc_of_calc[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        pred_sliced = preds[non_zero_indices[:, 0], non_zero_indices[:, 1]]

        area_loss = self.mse(pred_sliced, target_sliced)

        loss = label_loss * self.alpha + area_loss * (1 - self.alpha)
        return loss.to(torch.half)

        # targets_area = torch.einsum("ijklm->ij", targets)
        # # print(f"{targets_area.size() = }")
        # # print(f"{preds.size() = }")
        # num_row_patch = preds.size()[-1] ** 0.5
        # # print(f"{num_row_patch = }")
        # num_pixel_per_patch = (self.image_size / num_row_patch) ** 2
        # # print(f"{num_pixel_per_patch = }")
        # percentage_of_total_target = targets_area / num_pixel_per_patch
        # # print(f"{percentage_of_total_target = }")
        loss = label_loss + percentage_of_total_target
        # # print(f"{loss.size() = }")

        # return loss.sum() / torch.numel(loss)
