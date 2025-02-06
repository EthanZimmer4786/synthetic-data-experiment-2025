import os

path = './CIFAKE Dataset/train/FAKE/'

### Fix files missing class label ###
# for i in range(len(os.listdir(path))):
#     file = os.listdir(path)[i]
#     if(not('(' in file)):
#         os.rename(path + file, path + file.split('.').pop(0) + ' (1).jpg')

### Reformat names to "_"'s and reduce class indicator by 1 ###
# for file in os.listdir(path):
#     if(' ' in file):
#         category = int(file.split('(').pop(1).replace(').jpg', ''))
#         print(file.split('(').pop(0).replace(' ', '_') + str(category - 1) + '.jpg')
#         os.rename(path + file, path + file.split('(').pop(0).replace(' ', '_') + str(category - 1) + '.jpg')
#     else:
#         'huh?'

### Normalize file naming convention ###
# for file in os.listdir(path):
#     if(len(file.split('_').pop(0)) < 4):
#         new_file = '0' + file
#         print(new_file)
#         os.rename(path + file, path + new_file)

### Fix fake train images from counting test images in their name ###
# for file in os.listdir(path):
#     new_file = str(int(file.split('_').pop(0)) - 1000)
#     new_file = new_file + '_' + file.split('_').pop(1)
#     print(new_file)
#     os.rename(path + file, path + new_file)