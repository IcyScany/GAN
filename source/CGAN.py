import pickle
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelBinarizer
from torchvision.utils import save_image
from fid_score.fid_score import FidScore
from pylab import *

datapath = '../cifar10'
torch.manual_seed(0)


# 数据加载，其中只加载马的图片数据，标签为"7"
def dataloader(datapath):
    datapath = datapath + r'/cifar-10-batches-py/data_batch_'
    batches_num = 5
    global dataset
    dataset = []
    for i in range(batches_num):
        filename = datapath + str(i + 1)
        with open(filename, 'rb') as f:
            print('loading ' + filename)
            data = pickle.load(f, encoding='bytes')
            for k in range(len(data[b'labels'])):
                if data[b'labels'][k] == 7:         # 判断标签是否为马
                    image_data = data[b'data'][k].reshape(3, 32, 32)
                    # numpy转tensor，以及标准化及归一化
                    image_data = transforms.ToTensor()(image_data.transpose(1, 2, 0))     # 转换为tensor，并除255归一化到[0,1]之间
                    image_data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_data)
                    dataset.append(image_data)

    dataset = torch.utils.data.DataLoader(MyDataset(), batch_size=64,
                                          shuffle=True, num_workers=2)

    return dataset


# 设置dataset类
class MyDataset(Dataset):
    def __init__(self):
        self.data = dataset
        self.label = torch.ones(5000, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def Gnet():
    channel_num = 3  # 图片的通道数
    n_classes = 10  # 类别数
    latent_size = 64    # 噪声向量的维度
    n_g_feature = 64    # 生成器的深度
    # 生成器                             #(N,latent_size, 1,1)
    gnet = nn.Sequential(
        # 输入 (64, 1, 1)
        nn.ConvTranspose2d(latent_size + n_classes, 4 * n_g_feature, kernel_size=4, bias=False),
        nn.BatchNorm2d(4 * n_g_feature),
        nn.ReLU(),
        # (256, 4, 4)
        nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(2 * n_g_feature),
        nn.ReLU(),
        # (128, 8, 8)
        nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(n_g_feature),
        nn.ReLU(),
        # (64, 16, 16)
        nn.ConvTranspose2d(n_g_feature, channel_num, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
        # 输出 (3, 32, 32)
        )

    return gnet


def Dnet():
    channel_num = 3  # 图片的通道数
    n_classes = 10  # 类别数
    n_d_feature = 64    # 判别器的深度

    # 对抗网络             #(N,channel_num, 32,32)
    dnet = nn.Sequential(
        nn.Conv2d(channel_num + n_classes, n_d_feature, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        # (64, 16, 16)
        nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(2 * n_d_feature),
        nn.LeakyReLU(0.2, inplace=True),
        # (128, 8, 8)
        nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(4 * n_d_feature),
        nn.LeakyReLU(0.2, inplace=True),
        # (256, 4, 4)
        nn.Conv2d(n_d_feature * 4, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
        nn.Flatten(),  # (N,1)
        nn.Sigmoid()
        )

    return dnet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# 将标签进行one-hot编码
def to_categrical(y: torch.FloatTensor):
    lb = LabelBinarizer()
    lb.fit(list(range(0, 10)))
    y_one_hot = lb.transform(y.cpu())
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor.cuda()


# 样本和one-hot标签进行连接，以此作为条件生成
def concanate_data_label(data, y):  # data （N,nc, 128,128）
    y_one_hot = to_categrical(y)  # (N,1)->(N,n_classes)
    con = torch.cat((data, y_one_hot), 1)

    return con


def Train(traindata):
    gnet = Gnet().cuda()
    dnet = Dnet().cuda()

    # apply()初始化网络权重
    dnet.apply(weights_init)
    gnet.apply(weights_init)

    # 定义损失函数
    loss = torch.nn.BCELoss()
    Dloss = []
    Gloss = []

    # 优化器
    doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_label = 1.0    # 真实标签
    fake_label = 0.0    # 假标签
    nz = 64    # 噪声向量的维度

    batch_size = 64
    fixed_noises = torch.randn(64, nz, 1, 1).cuda()

    epoch_num = 100
    all_time = 0
    # 固定生成器，训练判别器
    for epoch in range(epoch_num):
        stime = time.time()
        for step, (data, target) in enumerate(traindata):
            data = data.cuda()
            target = target.cuda()
            # 拼接真实数据和标签
            target1 = to_categrical(target).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
            target2 = target1.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
            data = torch.cat((data, target2), dim=1)  # 将标签与数据拼接

            label = torch.full((data.size(0), 1), real_label).cuda()

            # （1）训练判别器
            # 训练真实图片
            dnet.zero_grad()
            output = dnet(data)
            loss_D1 = loss(output, label)
            loss_D1.backward()

            # 训练假照片,拼接噪声和标签
            noise_z = torch.randn(data.size(0), nz, 1, 1).cuda()
            noise_z = torch.cat((noise_z, target1), dim=1)  # (N,nz+n_classes,1,1)
            # 拼接假数据和标签
            fake_data = gnet(noise_z)
            fake_data = torch.cat((fake_data, target2), dim=1)  
            label = torch.full((data.size(0), 1), fake_label).cuda()

            output = dnet(fake_data.detach())
            loss_D2 = loss(output, label)
            loss_D2.backward()

            # 更新判别器
            doptimizer.step()

            # （2）训练生成器
            gnet.zero_grad()
            label = torch.full((data.size(0), 1), real_label).cuda()
            output = dnet(fake_data.cuda())
            lossG = loss(output, label)
            lossG.backward()

            # 更新生成器
            goptimizer.step()

            if step == 0:
                # 每循环一次输出模型训练结果
                thetime = time.time() - stime
                all_time = all_time + thetime

                Dloss.append(loss_D1.item() + loss_D2.item())
                Gloss.append(lossG.item())

                target3 = to_categrical(torch.full((64, 1), 7)).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
                fake = torch.cat((fixed_noises, target3), dim=1)  # (N,nz+n_classes,1,1)
                fake = gnet(fake)

                # 保存图片
                fake = (fake * 0.5 + 0.5)
                save_image(fake, f'../Images/CGAN/output/output_{epoch + 1}.png')

                print(f'循环: {epoch + 1}/{epoch_num}, Time: {thetime}')
                print(f'对抗网络损失: {loss_D1 + loss_D2}, 生成网络损失: {lossG}\n')

    # 保存模型
    torch.save(gnet, f'../Model/CGAN/Gnet.pt')
    torch.save(dnet, f'../Model/CGAN/Dnet.pt')

    # 绘制loss
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    x_axis = list(range(1, 101))

    plt.plot(x_axis, Gloss, '-', color='r', label='Gloss')
    plt.plot(x_axis, Dloss, '-', color='b', label='Dloss')

    plt.legend(loc="upper right")
    plt.xlabel('迭代次数')
    plt.ylabel('loss')
    plt.title('CGAN')

    plt.savefig('../CGAN_loss.jpg')  # 保存该图片

    # 生成1000张图片
    for i in range(1000):
        noises = torch.randn(1, nz, 1, 1).cuda()
        target3 = to_categrical(torch.full((1, 1), 7)).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
        noises = torch.cat((noises, target3), dim=1)  # (N,nz+n_classes,1,1)
        generate_image = gnet(noises)
        generate_image = (generate_image * 0.5 + 0.5)

        save_image(generate_image, f"../Images/CGAN/Generate/{i}.png")

    # 计算FID
    fid = FidScore(['../Images/Real', '../Images/CGAN/Generate'], torch.device('cuda:0'), batch_size=1)
    fid_score = fid.calculate_fid_score()
    print(f'训练完成，总耗时:{all_time}\n生成1000张图片\n')
    print("计算FID分数\nFID_core: ", fid_score)


if __name__ == '__main__':
    trainloader = dataloader(datapath)
    Train(trainloader)

