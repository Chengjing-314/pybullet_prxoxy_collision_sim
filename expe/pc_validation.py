import os
import torch

path =  "/home/chengjing/Desktop/save_new_ratio"
worlds = 1
cams = 5


tacc, tprc, trecall, TPR, TNR = 0, 0, 0, 0, 0


for world in range(worlds):
        wpath = os.path.join(path, "world_" + str(world))
        gt_label = torch.load(os.path.join(wpath, "collision_label.pt"))
        for cam in range(cams):
            cpath = os.path.join(wpath, "cam_" + str(cam))
            cl = torch.load(os.path.join(cpath, "pc_collision_label.pt"))
            acc = torch.sum(gt_label == cl) /  gt_label.size(dim = 0)
            prec = torch.sum(cl[gt_label == 1] == 1) / (torch.sum(cl[gt_label == 1] == 1) + torch.sum(cl[gt_label == -1] == 1))
            recall = torch.sum(cl[gt_label == 1] == 1) / (torch.sum(cl[gt_label == 1] == 1) + torch.sum(cl[gt_label == 1] == -1))
            tpr = torch.sum(cl[gt_label == 1] == 1) / len(cl[gt_label == 1])
            tnr = torch.sum(cl[gt_label == -1] == -1) / len(cl[gt_label == -1])
            tacc += acc
            tprc += prec
            trecall += recall
            TPR += tpr
            TNR += tnr
            # print("label acc: {}".format(acc))
            # print("label precision: {}".format(prec))
            # print("label recall: {}".format(recall))
            
            
tacc, tprc, trecall = tacc.item(), tprc.item(), trecall.item()

print("avg label acc: {}".format(tacc / (worlds * cams)))
print("avg label precision: {}".format(tprc / (worlds * cams)))
print("avg label recall: {}".format(trecall / (worlds * cams)))
print("avg label TPR: {}".format(TPR / (worlds * cams)))
print("avg label TNR: {}".format(TNR / (worlds * cams)))