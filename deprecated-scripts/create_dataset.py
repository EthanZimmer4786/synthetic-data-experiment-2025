import os, shutil, random
from datetime import datetime
from PIL import Image

import qol_print

# [airplane, car, bird, cat, deer, dog, frog, horse, ship, truck]

LOGGER = True

TRAIN_IMAGE_COUNT = 1000 # 50,000 limit
FAKE_IMAGE_RATIO = 0.2 # 0.00 - 1.00

TEST_IMAGE_COUNT = 1000
ADD_FAKE_TEST_IMAGES = False # Includes additional folder with fake test images

TRAIN_IAMGE_COUNT_REAL = int(TRAIN_IMAGE_COUNT * (1 - FAKE_IMAGE_RATIO))
TRAIN_IAMGE_COUNT_FAKE = TRAIN_IMAGE_COUNT - TRAIN_IAMGE_COUNT_REAL

########## CIFAKE ##########

CIFAKE_PATH_TRAIN_REAL = './CIFAKE Dataset/train/REAL/'
CIFAKE_PATH_TRAIN_FAKE  = './CIFAKE Dataset/train/FAKE/'

CIFAKE_PATH_TEST_REAL = './CIFAKE Dataset/test/REAL/'
if(ADD_FAKE_TEST_IMAGES):
    CIFAKE_PATH_TEST_FAKE  = './CIFAKE Dataset/test/FAKE/'

########## Pre Validations ##########

if(TRAIN_IAMGE_COUNT_REAL > len(os.listdir(CIFAKE_PATH_TRAIN_REAL))):
    print(f'Requested {TRAIN_IAMGE_COUNT_REAL} real training images')
    print(f'Cannot exceed TRAIN_IAMGE_COUNT_REAL of: {str(len(os.listdir(CIFAKE_PATH_TRAIN_REAL)))} images')
    quit()
if(TRAIN_IAMGE_COUNT_FAKE > len(os.listdir(CIFAKE_PATH_TRAIN_FAKE))):
    print(f'Requested {TRAIN_IAMGE_COUNT_FAKE} fake training images')
    print(f'Cannot exceed TRAIN_IAMGE_COUNT_FAKE of: {str(len(os.listdir(CIFAKE_PATH_TRAIN_FAKE)))} images')
    quit()

if(TEST_IMAGE_COUNT > len(os.listdir(CIFAKE_PATH_TEST_REAL))):
    print(f'Requested {TEST_IMAGE_COUNT} testing images')
    print(f'Cannot exceed TEST_IMAGE_COUNT of: {str(len(os.listdir(CIFAKE_PATH_TEST_REAL)))} images')
    quit()
if(ADD_FAKE_TEST_IMAGES == True):
    if(TEST_IMAGE_COUNT > len(os.listdir(CIFAKE_PATH_TEST_FAKE))):
        print(f'Requested {TEST_IMAGE_COUNT} fake testing images')
        print(f'Cannot exceed TEST_IMAGE_COUNT of: {str(len(os.listdir(CIFAKE_PATH_TEST_FAKE)))} images')
        quit()

########## Dataset Directory Creation ##########

dataset_name = 'Dataset ' + str(datetime.now())
dataset_name = dataset_name.split('.').pop(0)
dataset_name = dataset_name.replace(':','.')
if(LOGGER):
    print('----------')
    print('Dataset name: ' + dataset_name)
    print('----------')

DATASET_PATH = './Generated Datasets/' + dataset_name + '/'

os.makedirs(DATASET_PATH)
os.makedirs(DATASET_PATH + 'train/')
if(TEST_IMAGE_COUNT > 0):
    os.makedirs(DATASET_PATH + 'test/real/')
    if(ADD_FAKE_TEST_IMAGES):
        os.makedirs(DATASET_PATH + 'test/fake/')

for i in range(10):
    os.makedirs(DATASET_PATH + f'train/{i}')
    if(TEST_IMAGE_COUNT > 0):
        os.makedirs(DATASET_PATH + f'test/real/{i}')
        if(ADD_FAKE_TEST_IMAGES):
            os.makedirs(DATASET_PATH + f'test/fake/{i}')

########## Get Random Files Function ##########

def get_random_files(count, src_dir, dst_dir, suffix):
    if(LOGGER):
        print(f'Getting random files from: {src_dir}')
        
    files = os.listdir(src_dir)
    for i in range(count):
        index = random.randrange(len(files))
        file = files.pop(index) 
        folder = dst_dir + file.split('_').pop(1).replace('.jpg','')
        shutil.copy(src_dir + file, folder + '/' + file.split('.').pop(0) + f'{suffix}.jpg')

        if(LOGGER):
            if((i + 1) % round(count / 4) == 0):
                print(f'{round((i + 1) / count * 100)}%')
            
    if(LOGGER):
        print('----------')

########## Training Directory ##########

if(LOGGER):
    print(f'Training real image count: {TRAIN_IAMGE_COUNT_REAL}')
    print(f'Training fake image count: {TRAIN_IAMGE_COUNT_FAKE}')
    print('----------')

if(TRAIN_IAMGE_COUNT_REAL > 0):
    get_random_files(TRAIN_IAMGE_COUNT_REAL, CIFAKE_PATH_TRAIN_REAL, f'{DATASET_PATH}train/', '_r')

if(TRAIN_IAMGE_COUNT_FAKE > 0):
    get_random_files(TRAIN_IAMGE_COUNT_FAKE, CIFAKE_PATH_TRAIN_FAKE, f'{DATASET_PATH}train/', '_f')

########## Testing Directory ##########

if(LOGGER):
    print(f'Testing image count: {TEST_IMAGE_COUNT}')
    print(f'Add fake testing images? {ADD_FAKE_TEST_IMAGES}')
    print('----------')

if(TEST_IMAGE_COUNT > 0):
    get_random_files(TEST_IMAGE_COUNT, CIFAKE_PATH_TEST_REAL, f'{DATASET_PATH}test/real/', '_r')

    if(ADD_FAKE_TEST_IMAGES):
        get_random_files(TEST_IMAGE_COUNT, CIFAKE_PATH_TEST_FAKE, f'{DATASET_PATH}test/fake/', '_f')

########## Post Validations ##########

paths = [] 
counts = []

if(TRAIN_IMAGE_COUNT > 0):
    paths.append(f'{DATASET_PATH}train/')
    counts.append(TRAIN_IMAGE_COUNT)

if(TEST_IMAGE_COUNT > 0):
    paths.append(f'{DATASET_PATH}test/real/')
    counts.append(TEST_IMAGE_COUNT)

    if(ADD_FAKE_TEST_IMAGES):
        paths.append(f'{DATASET_PATH}test/fake/')
        counts.append(TEST_IMAGE_COUNT)

if(len(paths) != len(counts)):
    print('Discrepency between paths length and counts length in post validations')
    quit()

for i in range(len(paths)):
    sum = 0
    for j in range(10):
        sum += len(os.listdir(paths[i] + str(j)))

    if(LOGGER):
        print(f'{sum}/{counts[i]} files in {paths[i].replace(DATASET_PATH,'')}')
        print('----------')

    if(sum != counts[i]):
        print(f'Discrepency between number of files in {paths[i].replace(DATASET_PATH,'')} and expected number of {counts[i]}')
        quit()

########## Chart ##########

if(LOGGER):
    for i in range(len(paths)):
        frequencies = [0] * 10
        for j in range(10):
            frequencies[j] = len(os.listdir(paths[i] + str(j))) / counts[i]
            
        qol_print.print_chart(frequencies, f'Relative class frequencies in {paths[i].replace(DATASET_PATH,'')}:')