# Relate to Algorithm 1, remove positive tiles
# input：image path（string）、xml path（string）、label of this image（1/-1）
# output：R-（tensor）
def img_split_r_negative(label_type, img_path, xml_path):
    from PIL import Image
    from utils.read_xml import read_voc_xml
    from torchvision.transforms import transforms
    import torch
    from torchvision.transforms import Resize

    torch_resize = Resize([400, 400])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path)
    label_list = read_voc_xml(xml_path)
    width, height = img.size
    # get 3 * 5 tiles
    w = width / 5
    h = height / 3
    i = 0
    j = 0
    rst_tensor = []

    while i + w <= width:
        while (j + h <= height):
            new_img = img.crop((i, j, i + w, j + h))
            new_img = transforms.ToTensor()(new_img).to(device)
            new_img = torch_resize(new_img).unsqueeze(dim=0)
            if label_type == -1:
                try:
                    rst_tensor = torch.cat((rst_tensor, new_img))
                except:
                    rst_tensor = new_img
            # if the tile contains foreground, remove it
            else:
                is_foreground = False
                for label in label_list:
                    label_minx, labe_miny, label_maxx, label_maxy = label
                    if label_minx <= i <= label_maxx and labe_miny <= j <= label_maxy:
                        is_foreground = True
                        break
                    if label_minx <= i + w <= label_maxx and labe_miny <= j <= label_maxy:
                        is_foreground = True
                        break
                    if label_minx <= i <= label_maxx and labe_miny <= j + h <= label_maxy:
                        is_foreground = True
                        break
                    if label_minx <= i + h <= label_maxx and labe_miny <= j + w <= label_maxy:
                        is_foreground = True
                        break
                if not is_foreground:
                    try:
                        rst_tensor = torch.cat((rst_tensor, new_img))
                    except:
                        rst_tensor = new_img
            j = j + h
        i += w
        j = 0
    return rst_tensor


def img_seg(img_path):
    from PIL import Image
    import torch
    from torchvision.transforms import transforms
    from torchvision.transforms import Resize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_resize = Resize([400, 400])
    img = Image.open(img_path)
    width, height = img.size
    w = width / 5
    h = height / 3
    i = 0
    j = 0
    rst_tensor = []
    while i + w <= width:
        while j + h <= height:
            new_img = img.crop((i, j, i + w, j + h))
            new_img = transforms.ToTensor()(new_img).to(device)
            new_img = torch_resize(new_img).unsqueeze(dim=0)
            try:
                rst_tensor = torch.cat((rst_tensor, new_img))
            except:
                rst_tensor = new_img

            j = j + h
        i += w
        j = 0

    return rst_tensor
