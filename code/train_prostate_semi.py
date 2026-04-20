import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom
from module.unet_condiction import UNet,UNet_Anchor
import SimpleITK as sitk

from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism

from dataloaders.dataset import (BaseDataSets, RandomGenerator, WeakStrongAugment, Normalize, TwoStreamBatchSampler)

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import argparse
import shutil
from scipy.signal import argrelextrema
from utils import losses, ramps
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/Prostate', help='Name of Experiment')
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--base_lr", type=float, default=0.001, help="learning_rate")
parser.add_argument("--exp", type=str, default="Semi", help="method")
parser.add_argument("--gpu", type=str, default="cuda:0", help="method")
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
set_determinism(123)
args = parser.parse_args()

train_dir = ""
eval_dir = ""

image_size = [224, 224]

logdir = f"../logs_prostate/{args.labeled_num}L_{args.exp}"
print(logdir)
model_save_path = os.path.join(logdir, "model")
code_save_path = os.path.join(logdir, "code")

num_modality = 1
base_lr = args.base_lr
num_classes = 2

max_epoch = 5000
batch_size = args.batch_size
val_every = 10

env = "pytorch"
num_gpus = 1

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, 
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27,"3":130, "4": 53, "7":299}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
labeled_slice = patients_to_slices("Prostate", args.labeled_num)

device = args.gpu
db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        Normalize(True),
        RandomGenerator(image_size,2,False)
    ]))

db_val = BaseDataSets(base_dir=args.root_path, split="val")

total_slices = len(db_train)
labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
print("Total silices is: {}, labeled slices is: {}".format(
    total_slices, labeled_slice))
labeled_idxs = list(range(0, labeled_slice))
unlabeled_idxs = list(range(labeled_slice, total_slices))
batch_sampler = TwoStreamBatchSampler(
labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_top_peaks(D):
    A = 1    
    k0 = (1 * np.pi) / 5 
    c = 0.1 + 0.2 * D  
    b = 0      
    x = np.linspace(0, 10, 1000)  

    k = k0 + c * x
    f = A * np.sin(k * (x - b))

    peaks = argrelextrema(f, np.greater)[0]

    rounded_peaks = np.round(x[peaks]).astype(int)

    unique_peaks = set(rounded_peaks) 
    top_3_peaks = sorted(unique_peaks, reverse=True)[:3]
    if top_3_peaks[0]==10:
        top_3_peaks[0]=9

    return top_3_peaks
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def create_model(ema=False):
        model = UNet_Anchor(in_chns=num_classes, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
class DiffUnet(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()


        betas = get_named_beta_schedule("linear", 1000)
        self.model = model
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
            sample_out = self.sample_diffusion.ddim_sample_loop_single(self.model, step, (args.batch_size-args.labeled_bs, num_classes, image_size[0], image_size[1]),
                                                                model_kwargs={"image": image})
            sample_out = sample_out["pred_xstart"]
            return sample_out



class ACDCTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device, val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[224, 224],
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.device = device
        self.model = DiffUnet(create_model()).to(self.device)
        self.ema_model = DiffUnet(create_model(ema=True)).to(self.device)

        self.best_mean_dice = 0.0
        self.iter_num = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.base_lr, weight_decay=1e-3)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=100,
                                                       max_epochs=max_epochs)

        self.dice_loss = losses.DiceLoss(num_classes)

    def training_step(self, batch):

        ##  Labeled sample
        image, label, onehot_label = self.get_input(batch)
        L_image, L_label = image[:args.labeled_bs], onehot_label[:args.labeled_bs]
        x_start = L_label
        self.iter_num += 1
        # Perform noise addition and denoising for labeled data
        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=L_image, pred_type="denoise")
        L_output = torch.softmax(pred_xstart, dim=1)

        loss_bce = self.ce(L_output, label[:args.labeled_bs].long()) 
        loss_dice = self.dice_loss(L_output, label[:args.labeled_bs].unsqueeze(1))
        loss_mse = self.mse(L_output, L_label)

        # Process unlabeled data with time-step ensemble
            
        time_steps = get_top_peaks(1-loss_dice.item())
        U_image = image[args.labeled_bs:]

        step1_output = self.model(image = U_image, pred_type = 'ddim_single',step = time_steps[0])
        step2_output = self.model(image = U_image, pred_type = 'ddim_single',step = time_steps[1])
        step3_output = self.model(image = U_image, pred_type = 'ddim_single',step = time_steps[2])
        avg_output = (step1_output + step2_output + step3_output) / len(time_steps)
        avg_output = torch.softmax(avg_output,dim=1)
        consistency_weight = get_current_consistency_weight(self.epoch)
        loss_un = (self.mse(torch.softmax(step1_output,dim=1), avg_output) + self.mse(torch.softmax(step2_output,dim=1), avg_output) + self.mse(torch.softmax(step3_output,dim=1), avg_output))/3.*consistency_weight
        self.writer.add_scalar('info/loss_un', loss_un, self.iter_num)
        loss = loss_dice + loss_bce + loss_mse + loss_un

        self.log("train_loss", loss, step=self.global_step)

        for param_group in self.optimizer.param_groups:
                lr_ = param_group['lr'] 
        self.writer.add_scalar('info/lr', lr_, self.iter_num)
        self.writer.add_scalar('info/total_loss', loss, self.iter_num)
        self.writer.add_scalar('info/loss_ce', loss_bce, self.iter_num)
        self.writer.add_scalar('info/loss_dice', loss_dice, self.iter_num)
        self.writer.add_scalar('info/loss_mse', loss_mse, self.iter_num)


        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        one_label = batch["onehot_label"]

        # label = label.float()
        one_label = one_label.float()
        return image, label, one_label


    def validation_ds(self, ):
        Pros_dice = 0.
        cnt = 0
        test_save_path = './outputs/'

        val_loader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
        for idx, batch in enumerate(val_loader):
            output_list, target_list = [], []
            image, label = batch["image"],batch["label"]
            image, label = image.squeeze(0).cpu().detach(
                ).numpy(), label.squeeze(0).cpu().detach().numpy()

            prediction = np.zeros_like(label)
            
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                slice = zoom(slice, (image_size[0] / x, image_size[0] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(
                    0).unsqueeze(0).float().to(device)
                output = self.model(image=input, pred_type="ddim_sample_val")
                output = torch.softmax(output,dim=1)

                output = (torch.argmax(output.detach(),dim=1)).float().cpu().numpy().squeeze(0)#.squeeze(0)
                output = zoom(output, (x / image_size[0], y / image_size[0]), order=0)

                output_list.append(output)

            prediction = np.array(output_list)

            Pros = dice(prediction==1, label ==1)

            print(
                f"dice   ===>   Prostate is {Pros}"
            )

            Pros_dice += Pros
            cnt += 1

        Pros_avg = Pros_dice / cnt

        print("Average")
        print(
            f"dice   ===>   RV is {Pros_avg}"#
        )

        return [Pros_avg]

    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      self.iter_num, round(mean_dice.item(), 4)))
            torch.save(self.model.state_dict(), save_mode_path)

        


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
    snapshot_path = logdir
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    trainer = ACDCTrainer(env_type=env,
                          max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          logdir=logdir,
                          val_every=val_every,
                          num_gpus=num_gpus,
                          master_port=17751,
                          training_script=__file__)

    trainer.train(train_dataset=db_train, val_dataset=db_val)
