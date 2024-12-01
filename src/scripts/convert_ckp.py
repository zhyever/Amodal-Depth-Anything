import torch

update_ckp = {}
dav2_ckp = torch.load('./work_dir/ckp/depth_anything_v2_vitg.pth', map_location='cpu')
for k,v in dav2_ckp.items():
    if 'pretrained' in k:
        update_ckp[k] = v
torch.save(update_ckp, './work_dir/ckp/depth_anything_v2_vitg_backbone.pth')
