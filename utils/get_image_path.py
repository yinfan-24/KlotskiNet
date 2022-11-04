def get_train_list(class_name):
    import random

    base_train_path = r"raw_dataset/VOCdevkitTrain/VOC2007/ImageSets/Main/"
    class_train_path = base_train_path + class_name + "_train.txt"
    fo = open(class_train_path, "r+", encoding="utf-8")
    negative_list = []
    positive_list = []
    for line in fo.readlines():
        filename, true_label = line.split()
        true_label = int(true_label)
        negative_list.append(filename) if true_label == -1 else positive_list.append(filename)
    fo.close()

    tmp_list = positive_list
    while len(positive_list) < len(negative_list):
        positive_list.append(tmp_list[random.randrange(len(tmp_list))])

    rst_list = []
    for i in range(0, min(len(positive_list), len(negative_list))):
        rst_list.append((positive_list[i], 1))
        rst_list.append((negative_list[i], -1))
    return rst_list


def get_test_list(class_name):

    base_test_path = r"raw_dataset/VOCdevkitTest/VOC2007/ImageSets/Main/"
    class_test_path = base_test_path + class_name + "_test.txt"
    fo = open(class_test_path, "r+", encoding="utf-8")
    negative_list = []
    positive_list = []
    for line in fo.readlines():
        filename, true_label = line.split()
        true_label = int(true_label)
        negative_list.append(filename) if true_label == -1 else positive_list.append(filename)
    fo.close()

    rst_list = []
    for i in range(0, len(negative_list)):
        rst_list.append((negative_list[i], -1))
    for i in range(0, len(positive_list)):
        rst_list.append((positive_list[i], -1))
    return rst_list