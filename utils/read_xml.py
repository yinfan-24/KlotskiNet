# Read the annotation xml file of the VOC dataset, get the annotated box and the category of the image

# input：xml path（string）
# output：the list of the marker box（list[list[int]]）
def read_voc_xml(xml_path):
    from xml.dom import minidom

    doc = minidom.parse(xml_path)
    bndbox = doc.getElementsByTagName("object")
    box_list = []
    for box in bndbox:
        xmin = int(box.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = int(box.getElementsByTagName("ymin")[0].firstChild.data)
        xmax = int(box.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = int(box.getElementsByTagName("ymax")[0].firstChild.data)
        box_list.append([xmin, ymin, xmax, ymax])
    return box_list