import torch.nn.functional as F
def iou(pred, target):
    ious = []
    for cls in range(34):
        # Complete this function
        pred = F.softmax(pred,dim=1)
        pred = pred.cuda()
        inter = pred * target
        inter = inter.view(2,34,-1).sum(2)
        union= pred + target - (pred*target)
        union = union.view(2, 34,-1).sum(2)

#         loss = inter/union
#         intersection =(pred[:,cls,:,:] & target[:,cls,:,:]).float().sum((1, 2))
#         union = (pred[:,cls,:,:] | target[:,cls,:,:]).float().sum((1, 2))  
        if not union.nelement()==0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(inter/union)            
    return ious


def pixel_acc(pred, target):
    #Complete this function
    pass