import os
import json
import shutil


# old_image_dir = os.path.join(basedir, "imgs")
# new_image_dir = os.path.join(basedir, "new_images")
# 
# old_mask_dir = os.path.join(basedir, "val_masks_new")
# new_mask_dir = os.path.join(basedir, "new_masks")
# 
# if not os.path.exists(new_image_dir):
#     os.makedirs(new_image_dir)
#     os.makedirs(new_mask_dir)

data = {'test':{}}
count = 0
for split in ["train", "val"]:
    with open("./Models/utils/dataloader/annotation_new_{split}.txt", "r") as f:
        for line in f.readlines():
            image_id, sentence = int(line.split()[0]), line.split()[1:]
            command = " ".join(sentence)
            # old_image_filename = os.path.join(old_image_dir, f'img_{split}_{image_id}.jpg')
            # old_mask_filename = os.path.join(old_mask_dir, f'gt_img_ann_{split}_{image_id}.png')

            # new_image_filename = os.path.join(new_image_dir, f'img_test_{count}.jpg')
            # new_mask_filename = os.path.join(new_mask_dir, f'gt_img_ann_test_{count}.png')
            data['test'][count] = {"command": command, "img": f'img_test_{count}.jpg'}
            # shutil.copy(old_image_filename, new_image_filename)
            # shutil.copy(old_mask_filename, new_mask_filename)

            count += 1

json_file = json.dumps(data)

with open("talk2car_test.json", "w") as f:
    f.write(json_file)

print(data['test'][0])
print(len(data.keys()))

