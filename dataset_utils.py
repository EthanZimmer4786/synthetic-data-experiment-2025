import os, shutil, random
from datetime import datetime

import qol_print

########## Count Validation ##########

def validate_counts(*args):
    """
    Ensures there are enough images in a directory to fulfill the desired amount

    Arguments:
    *args (tuple): tuple of image count and dataset path (# images, path)

    Returns:
    True if validation succeeds, False if otherwise
    """
    
    for arg in args:
        if arg[0] > len(os.listdir(arg[1])):
            print(f'Requested {arg[0]} files from directory: {arg[1]}')
            print(f'Cannot exceed file count of: {str(len(os.listdir(arg[1])))}')
            return False
    return True

########## Dataset Directory Creation ##########

def create_directories(DATASET_PATH, TEST_IMAGE_COUNT, ADD_FAKE_TEST_IMAGES):
    # Header Folders
    os.makedirs(DATASET_PATH, 0o777)
    os.makedirs(f'{DATASET_PATH}train/')
    if TEST_IMAGE_COUNT > 0:
        os.makedirs(f'{DATASET_PATH}test/real/')
        if ADD_FAKE_TEST_IMAGES:
            os.makedirs(f'{DATASET_PATH}test/fake/')
            
    # Class Subdirectories
    for i in range(10):
        os.makedirs(f'{DATASET_PATH}train/{i}')
        if TEST_IMAGE_COUNT > 0:
            os.makedirs(f'{DATASET_PATH}test/real/{i}')
            if ADD_FAKE_TEST_IMAGES:
                os.makedirs(f'{DATASET_PATH}test/fake/{i}')

########## Get Random Files ##########

def get_random_files(count, src_dir, dst_dir, suffix, LOGGER):
    if LOGGER: print(f'Getting random files from: {src_dir}')
        
    files = os.listdir(src_dir)
    for i in range(count):
        index = random.randrange(len(files))
        file = files.pop(index) 
        folder = dst_dir + file.split('_').pop(1).replace('.jpg','')
        shutil.copy2(src_dir + file, folder + '/' + file.split('.').pop(0) + f'{suffix}.jpg')

        if LOGGER:
            if (i + 1) % round(count / 4) == 0:
                print(f'{round((i + 1) / count * 100)}%')
            
    if LOGGER: print('----------')

########## Directory Validations ##########

def validate_directory_counts(LOGGER, *args):
    """
    Ensures each dataset has the exact number of images requested upon generation

    Arguments:
    LOGGER (bool): whether or not logging is enabled
    *args (tuple): tuple of image count and dataset path (# images, path)

    Returns:
    True if validation succeeds, False if otherwise
    """
    
    for arg in args:
        sum = 0
        for i in range(10):
            sum += len(os.listdir(arg[1] + str(i)))
        if LOGGER:
            print(f'{sum}/{arg[0]} files in {arg[1]}')
            print('----------')
        if sum != arg[1]:
            print(f'Discrepency between number of files in {arg[1]} and expected number of {arg[0]}')
            return False
    return True

########## Chart ##########

def print_chart(counts, paths, DATASET_PATH):
    for i in range(len(paths)):
        frequencies = [0] * 10
        for j in range(10):
            frequencies[j] = len(os.listdir(paths[i] + str(j))) / counts[i]
            
        qol_print.print_chart(frequencies, f'Relative class frequencies in {paths[i].replace(DATASET_PATH,"")}:')
        print('----------')

########## Calculate Directory Size ##########

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
