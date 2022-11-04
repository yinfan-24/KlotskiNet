def BD2A(k, shita):

    import torch
    import torch.nn as nn
    from network_prototype.resnet import Resnet50
    from torchvision import transforms
    from torchvision.transforms import Resize
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载已经完成训练的模型
    model = Resnet50(2)

    model.load_state_dict(torch.load(r'newwork_weights/resnet50_2022_10_24_23_26_38.pth'))
    model.eval()

    # 其余参数
    file_path = r"raw_dataset/VOCdevkitTrain/VOC2007/ImageSets/Main/aeroplane_train_test.txt"
    base_Train_img_path = r"raw_dataset/VOCdevkitTrain/VOC2007/JPEGImages/"
    base_Train_xml_path = r"raw_dataset/VOCdevkitTrain/VOC2007/Annotations/"
    torch_resize = Resize([200, 200])
    Up = torch.tensor([]).cuda()
    Un = torch.tensor([]).cuda()
    U = torch.tensor([]).cuda()
    file_name_dict = {}


    # 遍历Train文件，获取文件名
    fo_test = open(file_path, "r+", encoding="utf-8")

    id = 1
    for line in fo_test.readlines():
        print(id)
        id += 1
        filename, true_label = line.split()

        true_label = int(true_label)

        img_path = base_Train_img_path + filename + ".jpg"
        xml_path = base_Train_xml_path + filename + ".xml"
        tiles = img_seg(true_label, img_path, xml_path)

        if len(tiles) >= 1:
            max_conf_tile = tiles[0].unsqueeze(dim=0)
            max_conf_tile = torch_resize(max_conf_tile)
            max_confidence = torch.tensor(-10)
            for tile in tiles:
                t = tile.unsqueeze(dim=0)
                t = torch_resize(t)
                t_output = model(t)

                tile_confidence = t_output[0][torch.argmax(t_output)]
                if tile_confidence > max_confidence:
                    max_confidence = tile_confidence
                    max_conf_tile = t

            if true_label == -1:
                u = model.get_latent_features(max_conf_tile).cuda()
                try:
                    Un = torch.cat([Un, u])
                    U = torch.cat([U, u])
                    file_name_dict[str(len(U))] = filename
                except:
                    Un = u
                    U = u
                    file_name_dict[str(len(U))] = filename
            elif true_label == 1:
                u = model.get_latent_features(max_conf_tile).cuda()
                try:
                    Up = torch.cat([Up, u])
                    U = torch.cat([U, u])
                    file_name_dict[str(len(U))] = filename
                except:
                    Up = u
                    U = u
                    file_name_dict[str(len(U))] = filename
            else:
                print(true_label)
                print("error")

    print(Un.size(), Up.size(), U.size())
    Un_mean = torch.mean(Un, 0)
    Up_mean = torch.mean(Up, 0)
    Sp = (Up - Up_mean).T @ (Up - Up_mean)
    Spn = (Un - Up_mean).T @ (Un - Up_mean)
    # print(Sp.size(), Spn.size())
    Sp_inverse = Sp.inverse()
    direction_matrix = (Sp_inverse @ Spn).cpu().detach().numpy()
    # a 是特征值的集合， b是特征向量的集合
    a, b = np.linalg.eig(direction_matrix)
    # 挑‘出k个方向
    fy_k = b[np.argpartition(a, -k)[-k:]]


    shita_k = int(len(U) * shita)
    baised_files = []
    for fy_np in fy_k:
        fy = torch.from_numpy(fy_np).type(torch.FloatTensor).cuda()
        fy = fy.unsqueeze(dim=0)
        # print(U.type(), fy.type())
        project_u = (U @ fy.T).squeeze(dim=1)
        project_u = project_u.cpu().detach().numpy()
        filename_list_index = np.argpartition(project_u, -shita_k)[-shita_k:]
        for i in filename_list_index:
            baised_files.append(file_name_dict[str(i+1)])

    baised_files = list(set(baised_files))
    print(baised_files)