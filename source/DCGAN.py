import pickle
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
import torch.nn.init as init
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


# 生成网络,由四层转置卷积构成
def Gnet():
    latent_size = 64
    channel_num = 3
    n_g_feature = 64    # 隐层大小为64
    gnet = nn.Sequential(
        # 输入 (64, 1, 1)
        nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
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


# 对抗网络
def Dnet():
    n_d_feature = 64
    channel_num = 3
    # 输入 (3, 32, 32)
    dnet = nn.Sequential(
        nn.Conv2d(channel_num, n_d_feature, kernel_size=4, stride=2, padding=1),
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
        nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
    )

    return dnet


def weights_init(m):  # 初始化权重值的函数
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


def Train(traindata):
    gnet = Gnet().cuda()
    dnet = Dnet().cuda()

    # apply()初始化网络权重
    gnet.apply(weights_init)
    dnet.apply(weights_init)

    # 损失函数
    loss = nn.BCEWithLogitsLoss()
    Gloss = []
    Dloss = []

    # 优化器
    # Adam优化器,学习率n=0.002，动量参数=0.5
    goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 随机噪声图片
    batch_size = 64
    fixed_noises = torch.randn(batch_size, 64, 1, 1).cuda()

    # Trainging
    epoch_num = 100     # 循环次数
    all_time = 0
    for epoch in range(epoch_num):
        stime = time.time()
        for step, data in enumerate(traindata):
            if step == len(traindata) - 1:      # 剔除最后一张(16,3,32,32)
                continue
            inputs, _ = data
            inputs = inputs.cuda()

            # 训练对抗网络
            real_labels = torch.ones(batch_size).cuda()      # 真实数据对应标签为1(64,)
            real_preds = dnet(inputs)                        # 判别真实数据(64,1,1,1)

            real_outputs = real_preds.reshape(-1)  # (64,)
            dloss_real = loss(real_outputs, real_labels).cuda()     # 真实数据损失

            noises = torch.randn(batch_size, 64, 1, 1).cuda()  # 随机噪声(64,64,1,1)
            fake_images = gnet(noises)                         # 噪声图片(64,3,32,32)
            fake_labels = torch.zeros(batch_size).cuda()       # 假数据对应标签0
            fake = fake_images.detach()                        # 梯度计算不回溯到生成网络,可用于加快训练速度
            fake_preds = dnet(fake)                            # 鉴别假数据
            fake_outputs = fake_preds.view(-1)
            dloss_fake = loss(fake_outputs, fake_labels)       # 假数据损失

            dloss = dloss_real + dloss_fake     # 总损失
            dloss = dloss.cuda()
            dnet.zero_grad()                    # 梯度归零
            dloss.backward()
            doptimizer.step()

            # 训练生成网络
            labels = torch.ones(batch_size).cuda()
            preds = dnet(fake_images)         # 鉴别假数据
            outputs = preds.view(-1)
            gloss = loss(outputs, labels)     # 真数据看到的损失
            gloss = gloss.cuda()
            gnet.zero_grad()
            gloss.backward()
            goptimizer.step()

        # 每循环一次输出模型训练结果
        thetime = time.time() - stime
        all_time = all_time + thetime

        Dloss.append(dloss.item())
        Gloss.append(gloss.item())

        fake = gnet(fixed_noises)
        fake = (fake * 0.5 + 0.5)       # 还原标准化的图片
        save_image(fake, f'../Images/DCGAN/output/output_{epoch+1}.png')

        print(f'循环: {epoch+1}/{epoch_num}, Time: {thetime}')
        print(f'对抗网络损失: {dloss}, 生成网络损失: {gloss}\n')

    # 保存模型
    torch.save(gnet, f'../Model/DCGAN/Gnet.pt')
    torch.save(dnet, f'../Model/DCGAN/Dnet.pt')

    # 绘制loss
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    x_axis = list(range(1, 101))

    plt.plot(x_axis, Gloss, '-', color='r', label='Gloss')
    plt.plot(x_axis, Dloss, '-', color='b', label='Dloss')

    plt.legend(loc="upper right")
    plt.xlabel('迭代次数')
    plt.ylabel('loss')
    plt.title('DCGAN')

    plt.savefig('../DCGAN_loss.jpg')  # 保存该图片

    # 生成1000张图片
    for i in range(1000):
        noises = torch.randn(1, 64, 1, 1).cuda()    # 生成图像大小与原图像一致
        generate_image = gnet(noises)
        generate_image = (generate_image * 0.5 + 0.5)    # 还原标准化的图片
        save_image(generate_image, f"../Images/DCGAN/Generate/{i}.png")

    # 计算FID
    fid = FidScore(['../Images/Real', '../Images/DCGAN/Generate'], torch.device('cuda:0'), batch_size=1)
    fid_score = fid.calculate_fid_score()
    print(f'训练完成，总耗时:{all_time}\n生成1000张图片\n')
    print("计算FID分数\nFID_core: ", fid_score)


if __name__ == '__main__':
    trainloader = dataloader(datapath)
    Train(trainloader)
