import os
dir_path = 'data'
img_names = os.listdir(dir_path)
for old_name, i in zip(img_names, range(len(img_names))):
    os.rename(os.path.join(dir_path, old_name), os.path.join(dir_path,'data_{0}.png'.format(i)))
print(img_names)