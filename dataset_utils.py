import os, shutil, random
from datetime import datetime

import qol_print

########## Count Validation ##########

def validate_counts(counts, paths):
    if len(counts) != len(paths):
        print('Discrepency between counts length and paths length in validate_counts')
        return False
    for i in range(len(counts)):
        if counts[i] > len(os.listdir(paths[i])):
            print(f'Requested {counts[i]} files from directory: {paths[i]}')
            print(f'Cannot exceed file count of: {str(len(os.listdir(paths[i])))}')
            return False
    return True

########## Dataset Directory Creation ##########

def create_directories(DATASET_PATH, TEST_IMAGE_COUNT, ADD_FAKE_TEST_IMAGES):
    # Header Folders
    os.makedirs(DATASET_PATH)
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

def validate_directories(counts, paths, DATASET_PATH, LOGGER):
    if len(counts) != len(paths):
        print('Discrepency between counts length and paths length in validate_directories')
        return False
    for i in range(len(paths)):
        sum = 0
        for j in range(10):
            sum += len(os.listdir(paths[i] + str(j)))
        if LOGGER:
            print(f'{sum}/{counts[i]} files in {paths[i].replace(DATASET_PATH,"")}')
            print('----------')
        if sum != counts[i]:
            print(f'Discrepency between number of files in {paths[i].replace(DATASET_PATH,"")} and expected number of {counts[i]}')
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