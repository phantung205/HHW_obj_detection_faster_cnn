import os
import cv2
import xml.etree.ElementTree as ET
from configs import config


dir_train = os.path.join(config.dir_data_processed, "train")


image_name = "000001_jpg.rf.f695422118aba8250f5a102143d82e26.jpg"

image_path = os.path.join(dir_train, image_name)

# lấy ra file xml
xml_name = os.path.splitext(image_name)[0] + ".xml"
xml_path = os.path.join(dir_train, xml_name)


# đọc ảnh
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Không đọc được ảnh!")

# đọc file xml
tree = ET.parse(xml_path)
root = tree.getroot()


# duyện qua các đối tượng
for obj in root.findall("object"):

    # Lấy tên class
    class_name = obj.find("name").text.strip()

    # Lấy tọa độ bounding box
    b = obj.find("bndbox")

    xmin = int(b.find("xmin").text)
    ymin = int(b.find("ymin").text)
    xmax = int(b.find("xmax").text)
    ymax = int(b.find("ymax").text)

    print(class_name, xmin, ymin, xmax, ymax)

    # Chọn màu theo class
    if class_name == "head":
        color = (0, 0, 255)
    elif class_name == "helmet":
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    # Vẽ bounding box theo kiểu xmin,ymin ,xmax,ymax xem đúng ko
    cv2.rectangle(image,(xmin, ymin),(xmax, ymax),color,2)

    # Hiển thị tên class
    cv2.putText(image,class_name,(xmin, ymin - 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)


cv2.imshow("Check Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()