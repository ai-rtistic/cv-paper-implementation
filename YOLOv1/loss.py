import torch
import torch.nn as nn
from utils import intersection_over_union

# batch size: 64 in paper
# print shape at each row



class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):  # S : split size (grid), B : bbox num at each grid, C : number of classes
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.S = S
        self.B = B
        self.C = C
        self.lamda_noobj = 0.5
        self.lamda_coord = 5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        # prdiction: (64, 1470) -> (64, 7, 7, 30)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # iou_b1: (64, 7, 7, 1),  predictions[..., 21:25]: (64, 7, 7, 4)
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # ious: (2, 64, 7, 7, 1),   iou_b1.unsqueeze(0): (1, 64, 7, 7, 1)
        iou_maxes, best_box = torch.max(ious, dim=0)
        # iou_maxes: (64, 7, 7, 1),  best_box: (64, 7, 7, 1) -> max location(index)
        exists_box = target[..., 20].unsqueeze(3)   # identity_obj_i  
        # exists_box: (64, 7, 7, 1),  target[..., 20]: (64, 7, 7)


        ## box coordinate loss

        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30]) + (1-best_box) * predictions[..., 21:25]
        ) # best_box=1 if b2 have max iou,   best_box=0 if b1 have max iou
        # box_predictions: (64, 7, 7, 4)
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        # change w, h to sqrt w, h
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),  #  (64, 7, 7, 4) -> (64*7*7, 4)
            torch.flatten(box_targets, end_dim=-2)
        )

        ## object loss
        pred_box = (
            best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21]
        )  # pred_box: (64, 7, 7, 1)

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),   # (64, 7, 7, 1) -> (64*7*7*1)
            torch.flatten(exists_box * target[..., 20:21])   
        )


        ## no object loss
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21] , start_dim=1),  # (64, 7, 7, 1) -> (64, 7*7*1)
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26] , start_dim=1),  # (64, 7, 7, 1) -> (64, 7*7*1)
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        ## class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),  # (64, 7, 7, 20) -> (64*7*7, 20)
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )


        loss = (
            self.lamda_coord * box_loss
            + object_loss
            + self.lamda_noobj * no_object_loss
            + class_loss
        )

        return loss



        



