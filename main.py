from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler

from unet import Unet
from dataset import MyDataset
from dataset import MyDataset_unlabeled
from dataset import MyDataset_test
from MeanTeacher import *


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换
# image转换
x_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask转换
y_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()
])

def train(batch_size):
    student_model = Unet(3, 1).to(device)
    teacher_model = Unet(3,1).to(device)
    epochs = 50
    min_lr = 1e-4
    batch_size = batch_size
    optimizer = optim.Adam(student_model.parameters())
    # 学习率调节
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=epochs,
                                                   eta_min=min_lr)
    liver_dataset_labeled = MyDataset("altrasound/train_labeled",transform=x_transforms,target_transform=y_transforms)
    liver_dataset_unlabeled = MyDataset_unlabeled("altrasound/train_unlabeled",transform=x_transforms)
    dataloaders_labeled = DataLoader(liver_dataset_labeled, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders_unlabeled = DataLoader(liver_dataset_unlabeled, batch_size=batch_size, shuffle=True, num_workers=0)
    trainer = Trainer(student_model, teacher_model, optimizer, device)
    trainer.loop_train(epochs, dataloaders_labeled, dataloaders_unlabeled, scheduler)

def check(stu_ckpt, t_ckpt):
    student_model = Unet(3, 1).to(device)
    teacher_model = Unet(3,1).to(device)
    liver_dataset_test = MyDataset_test("altrasound/val",transform=x_transforms,target_transform=y_transforms)
    dataloaders_test = DataLoader(liver_dataset_test, batch_size=1)
    optimizer = optim.Adam(student_model.parameters())
    trainer = Trainer(student_model, teacher_model, optimizer, device)
    trainer.test(student_model, teacher_model, stu_ckpt, t_ckpt, dataloaders_test)

if __name__ == '__main__':
# 设置训练时的batch_size
    batch_size = 4
    train(batch_size)

# 训练后进行测试
    check_times = 50
    for t in range(check_times):
        stu_ckpt = 'student_weights_%d.pth' % t
        t_ckpt = 'teacher_weights_%d.pth' % t
        print('##### check iteration %d #####' % (t + 1))
        check(stu_ckpt, t_ckpt)
        print("")



