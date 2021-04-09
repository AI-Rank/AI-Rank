import os
import random

class_id_to_img_map = {}
with open("val_list.txt", "r") as fin, open("val_list_1k.txt", "w") as fout:
    lines = fin.readlines()
    for line in lines:
        class_id = int(line.split(" ")[1])
        if class_id not in class_id_to_img_map:
            class_id_to_img_map[class_id] = []
        class_id_to_img_map[class_id].append(line)

    os.system("mkdir val_list_1k")
    out_arr = []
    for class_id in class_id_to_img_map:
        lines = random.sample(class_id_to_img_map[class_id], 1)
        for line in lines:
            out_arr.append(line)
            fout.write(line)
            strline = line.split(" ")[0]
            print("strline: ", strline)
            strline2 = strline.split("/")[1]
            print("strline2: ", strline2)
            # strline2 is image_name
            ori = "val/" + strline2
            now = "val_list_1k/" + strline2
            print("ori: ", ori)
            print("now: ", now)
            command = "cp ./" + ori + "  ./" + now
            print("command: ", command)
            os.system(command)

