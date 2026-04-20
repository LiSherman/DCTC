import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from module.unet_condiction import UNet,UNet_Anchor
from scipy.ndimage import zoom
import shutil
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism
import glob
from dataloaders.dataset import BaseDataSets

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import SimpleITK as sitk
set_determinism(123)

train_dir = ""
test_dir = ""

image_size = [224,224]

logdir = "../logs_acdc/"
method ='3L'

num_modality = 1
num_classes = 3
save_result = True

max_epoch = 8000
batch_size = 32
val_every = 200

env = "pytorch"
num_gpus = 1
root_path = '../../data/ACDC'
snapshot_path = "../logs_acdc/{}/".format(method)
print(snapshot_path)
test_save_path = os.path.join(snapshot_path,'output')
print(test_save_path)
device = "cuda:0"
for root, dirs, files in os.walk(snapshot_path):
        for file in files:
            if file.startswith('unet_best_model'):
                checkpoint_path =  os.path.join(root, file)

print(checkpoint_path)

class DiffUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()


        betas = get_named_beta_schedule("linear", 1000)
        self.model = UNet_Anchor(in_chns=num_classes, class_num=num_classes).to(device)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            return self.model(x, t=step, image=image)

        elif pred_type == "ddim_sample":
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (args.batch_size-args.labeled_bs, num_classes, image_size[0], image_size[1]),
                                                                model_kwargs={"image": image})
            sample_out = sample_out["pred_xstart"]
            return sample_out
        elif pred_type == "ddim_sample_val":
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, num_classes, image_size[0], image_size[1]),
                                                                model_kwargs={"image": image})
            sample_out = sample_out["pred_xstart"]
            return sample_out
        elif pred_type == "ddim_single":
            sample_out = self.sample_diffusion.ddim_sample_loop_single(self.model, step, (1, num_classes, image_size[0], image_size[0]),
                                                                model_kwargs={"image": image})
            sample_out = sample_out["pred_xstart"]
            return sample_out

from medpy import metric
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

class ACDCTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[image_size[0], image_size[1]],
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.model = DiffUnet()

        if checkpoint_path is not None:
            print("-" * 60)
            print("加载预训练模型   ===>   ", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path,map_location=device), strict=False)
            print("-" * 60)
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label
    

    def validation_ds(self, ):

        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        db_val = BaseDataSets(base_dir=root_path, split="test")
        val_loader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
        self.model.eval()
        if save_result:
            if os.path.exists(test_save_path):
                shutil.rmtree(test_save_path)
            os.makedirs(test_save_path)
        f = open(test_save_path+'/performance.txt', 'a')
        for idx, batch in enumerate(val_loader):
            output_list, target_list = [], []
            image, label = batch["image"],batch["label"]
            case = batch["idx"][0]
            image, label = image.squeeze(0).cpu().detach(
                ).numpy(), label.squeeze(0).cpu().detach().numpy()

            prediction = np.zeros_like(label)
            
            for ind in range(image.shape[0]):
                
                output_label = np.zeros(image_size)
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                slice = zoom(slice, (image_size[0] / x, image_size[0] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(
                    0).unsqueeze(0).float().to(device)
                with torch.no_grad():

                    output = self.model(image=input, pred_type="ddim_sample_val")
                    output = torch.sigmoid(output)

                    output = (output > 0.5).float().cpu().numpy()
                    output_label[output[0,0]==1] = 1
                    output_label[output[0,1]==1] = 2
                    output_label[output[0,2]==1] = 3
                    output_label = zoom(output_label, (x / image_size[0], y / image_size[0]), order=0)
                    prediction[ind,:] = output_label

            if np.sum(prediction == 1)==0:
                first_metric = 0,0,0,0
            else:
                first_metric = calculate_metric_percase(prediction == 1, label == 1)

            if np.sum(prediction == 2)==0:
                second_metric = 0,0,0,0
            else:
                second_metric = calculate_metric_percase(prediction == 2, label == 2)

            if np.sum(prediction == 3)==0:
                third_metric = 0,0,0,0
            else:
                third_metric = calculate_metric_percase(prediction == 3, label == 3)
            avg_dice =(first_metric[0]+second_metric[0]+third_metric[0])/3.
            f.writelines(f'idx:{idx} performance {avg_dice} \n')
            if save_result:
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                img_itk.SetSpacing((1, 1, 10))
                prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
                prd_itk.SetSpacing((1, 1, 10))
                lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
                lab_itk.SetSpacing((1, 1, 10))
                
                sitk.WriteImage(prd_itk, test_save_path + f"/{case}_pred.nii.gz")
                sitk.WriteImage(img_itk, test_save_path + f"/{case}_img.nii.gz")
                sitk.WriteImage(lab_itk, test_save_path + f"/{case}_gt.nii.gz")
            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)
            print(
                f"dice   ===>   RV is {first_metric[0]},   Myo is {second_metric[0]},   LV is {third_metric[0]},   avg_dice is {avg_dice}"
            )
        idx = idx+1
        avg_metric = [first_total / idx, second_total /
                  idx, third_total / idx]
        print((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)
        
        f.writelines('metric is {} \n'.format(avg_metric))
        f.writelines('average metric is {}\n'.format((avg_metric[0]+avg_metric[1]+avg_metric[2])/3))
        f.close()


class ACDCTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.annotation_lines = []

        for file in os.listdir(os.path.join(data_dir)):
            if file.endswith("_gt_.nii.gz"):
                filename = file.split("_gt")[0]
                self.annotation_lines.append(filename)

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        """Get the images"""
        name = self.annotation_lines[index] + '.nii.gz'

        img_path = os.path.join(self.data_dir, name)

        mask_name = self.annotation_lines[index] + '_gt_.nii.gz'
        mask_path = os.path.join(self.data_dir, mask_name)

        image = nib.load(img_path).get_fdata()
        label = nib.load(mask_path).get_fdata()

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            state = torch.get_rng_state()
            torch.set_rng_state(state)
            sample = self.transform(sample)

        sample["file"] = self.annotation_lines[index]

        return sample


if __name__ == "__main__":
    trainer = ACDCTrainer(env_type=env,
                          max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          logdir=logdir,
                          val_every=val_every,
                          num_gpus=num_gpus,
                          master_port=17751,
                          training_script=__file__)

    trainer.validation_ds()
