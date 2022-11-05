def train_Klotskinet():
    import torch
    import os
    import torchvision
    import torch.nn as nn
    from tqdm import tqdm
    from torchvision.models.resnet import resnet50
    from network_prototype.resnet import Resnet50
    from torchvision import transforms
    from torch.autograd import Variable
    from torchvision.transforms import Resize

    from utils.get_image_path import get_train_list
    from utils.get_image_path import get_test_list
    from utils.image_split import img_split_r_negative
    from utils.get_save_time import get_save_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义超参数
    HyperParameters = {
        'lr': 0.001,
        'epochs': 5,
    }
    class_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                  "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    torch_resize = Resize([400, 400])

    # 加载model
    os.environ['TORCH_HOME'] = 'pretrain_weights/'
    resnet50 = resnet50(pretrained=True).to(device)
    model = Resnet50(2).to(device)
    # 读取参数
    pretrained_dict = resnet50.state_dict()
    model_dict = model.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr = HyperParameters['lr'])
    # 定义损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    base_img_train_path = r"raw_dataset/VOCdevkitTrain/VOC2007/JPEGImages/"
    base_xml_train_path = r"raw_dataset/VOCdevkitTrain/VOC2007/Annotations/"
    base_img_test_path = r"raw_dataset/VOCdevkitTest/VOC2007/JPEGImages/"
    base_xml_test_path = r"raw_dataset/VOCdevkitTest/VOC2007/Annotations/"

    for i in range(0, HyperParameters['epochs']):
        for j in range(0, len(class_list)):
            class_name = class_list[j]
            filename_list = get_train_list(class_name)

            for file_name, true_label in tqdm(filename_list, desc='Epoch: {}/{}||Train||class: {}/{}|'.format(i+1, HyperParameters['epochs'], j+1, len(class_list)) ):
                img_path = base_img_train_path + file_name + ".jpg"
                xml_path = base_xml_train_path + file_name + ".xml"
                try:
                    tiles = img_split_r_negative(true_label, img_path, xml_path)
                    tiles = Variable(tiles, requires_grad=True)
                except:
                    tiles = []
                true_label = torch.tensor([0]) if true_label == -1 else torch.tensor([1])
                true_label = true_label.to(device)


                if len(tiles) >= 1:
                    output = model(tiles).to(device)
                    max_conf = output[int(torch.argmax(output) / 2)].unsqueeze(dim=0)
                    loss = criterion(max_conf, true_label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # max_conf_tile = tiles[0].to(device).unsqueeze(dim=0)
                    # max_conf_tile = torch_resize(max_conf_tile)
                    # max_confidence = torch.tensor(-10).to(device)
                    # for tile in tiles:
                    #     t = tile.to(device).unsqueeze(dim=0)
                    #     t = torch_resize(t)
                    #     t_output = model(t).to(device)
                    #     tile_confidence = t_output[0][torch.argmax(t_output)]
                    #     if tile_confidence > max_confidence:
                    #         max_confidence = tile_confidence
                    #         max_conf_tile = t
                    # output = model(max_conf_tile).to(device)
                    # loss = criterion(output, true_label)
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
        total_nums = 0
        right_nums = 0
        for j in range(0, len(class_list)):
            class_name = class_list[j]
            filename_list = get_test_list(class_name)
            for file_name, true_label in tqdm(filename_list, desc='Epoch: {}/{}||Train||class: {}/{}|'.format(i+1, HyperParameters['epochs'], j+1, len(class_list)) ):
                img_path = base_img_test_path + file_name + ".jpg"
                xml_path = base_xml_test_path + file_name + ".xml"
                try:
                    tiles = img_split_r_negative(true_label, img_path, xml_path)
                except:
                    tiles = []
                if len(tiles) >= 1:
                    total_nums = total_nums + 1
                    max_conf_tile = tiles[0].to(device).unsqueeze(dim=0)
                    max_conf_tile = torch_resize(max_conf_tile)
                    max_confidence = torch.tensor(-10).to(device)
                    for tile in tiles:
                        t = tile.to(device).unsqueeze(dim=0)
                        t = torch_resize(t)
                        t_output = model(t).to(device)
                        tile_confidence = t_output[0][torch.argmax(t_output)]
                        if tile_confidence > max_confidence:
                            max_confidence = tile_confidence
                            max_conf_tile = t
                output = model(max_conf_tile).to(device)
                true_label = torch.tensor([0]) if true_label == -1 else torch.tensor([1])
                y = torch.tensor([0]) if output[0][0] > output[0][1] else torch.tensor([1])
                if y.equal(true_label):
                    right_nums = right_nums + 1
        print(right_nums, total_nums)
        print("Epoch:{}, Acc:{}, Time:{}\n".format(i+1, right_nums/(total_nums + 0.1), get_save_time()))
        torch.save(model.state_dict(), 'network_weights/resnet50_{}_{}.pth'.format(right_nums/(total_nums + 0.1), get_save_time()))

if __name__ == '__main__':
    train_Klotskinet()