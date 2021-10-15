from torch.utils.data import Dataset
import PIL.Image as Image
import os



def make_dataset(root):
    imgs=[]
    #训练集中的n张图片
    n = 10
    for i in range(n):
        root1 = root + '/img'
        root2 = root + '/gt'
        img = os.path.join(root1, "%d.jpg" % i)
        mask = os.path.join(root2, "%d.png" % i)
        imgs.append((img, mask))
    return imgs

def make_test_dataset(root):
    imgs=[]
    #训练集中的n张图片
    n = 5
    for i in range(n):
        root1 = root + '/img'
        root2 = root + '/gt'
        img = os.path.join(root1, "%d.jpg" % i)
        mask = os.path.join(root2, "%d.png" % i)
        imgs.append((img, mask))
    return imgs

def make_dataset_unlabeled(root):
    imgs = []
    n = 40
    for i in range(n):
        root1 = root + '/img'
        img = os.path.join(root1, "%d.jpg" % (i + 10))
        imgs.append(img)
    return imgs

class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class MyDataset_unlabeled(Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset_unlabeled(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path)
        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x

    def __len__(self):
        return len(self.imgs)

class MyDataset_test(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_test_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)