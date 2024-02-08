import pandas
import os
import shutil
data_frame = pandas.read_csv('HAM10000_metadata.csv')
target_types = ["mel"]
filtered_dataframe = data_frame[data_frame['dx'].isin(target_types)]
cancer_images_ids = filtered_dataframe['image_id'].values

# ids = data_frame[data_frame['image_id'].isin(result)]
# types = set(ids['dx'].values) 

for folder_name in ("ham10000_images_part_1", "ham10000_images_part_2"):
  # print(len(os.listdir(folder_name)))
  for image in os.listdir(folder_name):
    id = image.split(".")[0]
    if id in cancer_images_ids:
      shutil.move(os.path.join(folder_name, image), os.path.join("cancer_dataset", "cancer"))
    else:
      shutil.move(os.path.join(folder_name, image), os.path.join("cancer_dataset","nocancer"))


# shutil.move(src, dst, copy_function=copy2)
