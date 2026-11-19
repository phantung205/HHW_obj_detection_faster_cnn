from torch.utils.data import Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
from configs import config
import torch

class CustomDataset(Dataset):
    def __init__(self,root,is_train=True,transform=None):
        # khai báo các biến để có thể dùng trong class
        self.transform = transform

        if is_train:
            mode = "train"
        else:
            mode = "valid"

        self.root_mode = os.path.join(root,mode)


        # tạo list chưa tên file ảnh và nhãn ko chứa đuôi
        self.files = []

        for file_name in os.listdir(self.root_mode):
            # kiểm tra xem có phải ảnh ko
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # kiểm tra xem đọc đc ảnh ko
            image_path = os.path.join(self.root_mode,file_name)
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                continue

            # kiểm tra thêm xem có file xml nhãn tương  ứng nữa ko
            label_name = os.path.splitext(file_name)[0] + ".xml"
            xml_path = os.path.join(self.root_mode, label_name)
            if not os.path.exists(xml_path):
                continue

            self.files.append(file_name)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]

        # đọc ảnh
        image_path = os.path.join(self.root_mode,file)
        image = Image.open(image_path).convert("RGB")

        # đọc file xml
        xml_name = os.path.splitext(file)[0] + ".xml"
        xml_path = os.path.join(self.root_mode,xml_name)
        boxes, labels = self.parse_xml(xml_path)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


    def parse_xml(self,xml_path):
        # đọc file xml và chuyển nó thành cấu trúc cây đối tượng
        tree = ET.parse(xml_path)
        # lấy ra thẻ gốc
        root = tree.getroot()

        boxes, labels = [], []

        # tìm lấy ra tất cả các thẻ có tên object
        for obj in root.findall("object"):
            # lấy ra tên class
            name = obj.find("name").text.strip()

            label = config.categories.index(name) + 1

            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)