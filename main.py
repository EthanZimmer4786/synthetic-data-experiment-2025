import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from datetime import datetime
import random

import tensorflow as tf
import keras
from keras import layers

import sqlite3

import dataset_utils
import cnn_utils
import sqlite_utils

########## ########## ##########

def main(LOGGER = True,
         ### Dataset Values ###
         TRAIN_IMAGE_COUNT = 0, # 50,000 limit
         FAKE_IMAGE_RATIO = 0, # 0.00 - 1.00
     
         TEST_IMAGE_COUNT = 0,
         ADD_FAKE_TEST_IMAGES = False, # Includes additional folder with fake test images
         ### Model Values ###
         BATCH_SIZE = -1,
         
         IMG_HEIGHT = 32,
         IMG_WIDTH  = 32,

         EPOCHS = 30, # 50

         SEED = round(random.random() * 100),
         ### Dataset Paths ###     
         SOURCE = './CIFAKE-dataset/',
         DESTINATION = './generated-datasets/',
         ### Database Paths ###
         DATABASE_PATH = '',
         ):
    
    start_time = datetime.now()
    
    AUTOTUNE = tf.data.AUTOTUNE

    ########## Dataset Constants ##########

    DATASET_NAME = ('Dataset ' + str(datetime.now())).split('.').pop(0).replace(':','.')
    if LOGGER:
        print('----------')
        print('Dataset name: ' + DATASET_NAME)
        print('----------')

    DATASET_PATH = f'{DESTINATION}{DATASET_NAME}/'

    SOURCE_TRAIN_REAL = f'{SOURCE}train/real/'
    SOURCE_TRAIN_FAKE = f'{SOURCE}train/fake/'

    SOURCE_TEST_REAL = f'{SOURCE}test/real/'
    if ADD_FAKE_TEST_IMAGES:
        SOURCE_TEST_FAKE = f'{SOURCE}test/fake/' if ADD_FAKE_TEST_IMAGES else ''

    ########## Model Constants ##########

    if BATCH_SIZE == -1:
        BATCH_SIZE = cnn_utils.calc_batch_size(TRAIN_IMAGE_COUNT)
    if LOGGER: print(f'Batch Size: {BATCH_SIZE}', '\n', '----------')

    TRAIN_IAMGE_COUNT_REAL = int(TRAIN_IMAGE_COUNT * (1 - FAKE_IMAGE_RATIO))
    TRAIN_IAMGE_COUNT_FAKE = TRAIN_IMAGE_COUNT - TRAIN_IAMGE_COUNT_REAL

    ########## Dataset ##########

    # Pre Validations

    if dataset_utils.validate_counts(
        (TRAIN_IAMGE_COUNT_REAL, SOURCE_TRAIN_REAL),
        (TRAIN_IAMGE_COUNT_FAKE, SOURCE_TRAIN_FAKE),
        (TEST_IMAGE_COUNT, SOURCE_TEST_REAL),
        ) == False:
        print('!!! validate_counts error !!!')
        quit()

    if ADD_FAKE_TEST_IMAGES:
         if dataset_utils.validate_counts(
             (TEST_IMAGE_COUNT, SOURCE_TEST_FAKE),
             ) == False:
             print('!!! validate_counts error !!!')
             quit()
         
    dataset_utils.create_directories(DATASET_PATH, TEST_IMAGE_COUNT, ADD_FAKE_TEST_IMAGES)

    # Training Directory

    if LOGGER:
        print(f'Training real image count: {TRAIN_IAMGE_COUNT_REAL}')
        print(f'Training fake image count: {TRAIN_IAMGE_COUNT_FAKE}')
        print('----------')

    if TRAIN_IAMGE_COUNT_REAL > 0:
        dataset_utils.get_random_files(TRAIN_IAMGE_COUNT_REAL, SOURCE_TRAIN_REAL, f'{DATASET_PATH}train/', '_r', LOGGER)

    if TRAIN_IAMGE_COUNT_FAKE > 0:
        dataset_utils.get_random_files(TRAIN_IAMGE_COUNT_FAKE, SOURCE_TRAIN_FAKE, f'{DATASET_PATH}train/', '_f', LOGGER)

    # Testing Directory

    if LOGGER:
        print(f'Testing image count: {TEST_IMAGE_COUNT}')
        print(f'Add fake testing images? {ADD_FAKE_TEST_IMAGES}')
        print('----------')

    if TEST_IMAGE_COUNT > 0:
        dataset_utils.get_random_files(TEST_IMAGE_COUNT, SOURCE_TEST_REAL, f'{DATASET_PATH}test/real/', '_r', LOGGER)

        if ADD_FAKE_TEST_IMAGES:
            dataset_utils.get_random_files(TEST_IMAGE_COUNT, SOURCE_TEST_FAKE, f'{DATASET_PATH}test/fake/', '_f', LOGGER)

    # Post Validations

    counts = [TRAIN_IMAGE_COUNT, TEST_IMAGE_COUNT]
    paths = [f'{DATASET_PATH}train/', f'{DATASET_PATH}test/real/']
    if ADD_FAKE_TEST_IMAGES:
        counts.append(TEST_IMAGE_COUNT)
        paths.append(f'{DATASET_PATH}test/fake/')

    if dataset_utils.validate_directory_counts(LOGGER,
        (TRAIN_IMAGE_COUNT, f'{DATASET_PATH}train/'),
        (TEST_IMAGE_COUNT, f'{DATASET_PATH}test/real/'),
        ) == False:
        print('!!! validate_directories error !!!')
        quit()

    if ADD_FAKE_TEST_IMAGES:
        if dataset_utils.validate_directory_counts(LOGGER,
            (TEST_IMAGE_COUNT, f'{DATASET_PATH}test/fake/'),
            ) == False:
            print('!!! validate_directories error !!!')
            quit()

    if LOGGER: dataset_utils.print_chart(counts, paths, DATASET_PATH)

    ########## Convolutional Neural Network ##########

    # Format Created Dataset to Tensorflow Dataset

    train_ds = cnn_utils.create_dataset(f'{DATASET_PATH}train/', 'training', BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED, LOGGER)
    val_ds = cnn_utils.create_dataset(f'{DATASET_PATH}train/', 'validation', BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED, LOGGER)

    if LOGGER:
        train_batch_count = tf.data.experimental.cardinality(train_ds).numpy()
        print('Training Dataset Contains: ', train_batch_count, ' Batches of ', BATCH_SIZE, ' Images')
        print('Approximately ', train_batch_count * BATCH_SIZE, ' Images')
        print('----------')
        val_batch_count = tf.data.experimental.cardinality(val_ds).numpy()
        print('Validation Dataset Contains: ', val_batch_count, ' Batches of ', BATCH_SIZE, ' Images')
        print('Approximately ', val_batch_count * BATCH_SIZE, ' Images')
        print('----------')

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

    # Model Creation

    input_shape = layers.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))

    model = cnn_utils.get_model(input_shape)
    # if LOGGER: model.summary()

    model.compile(optimizer = keras.optimizers.Adam(),
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    # Model Training

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
        verbose = 1 if LOGGER else 0,
        )

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Model Testing

    test_ds = cnn_utils.create_dataset(f'{DATASET_PATH}test/REAL/', False, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED, LOGGER)
    if LOGGER: print('----------')

    test_results = model.evaluate(test_ds,
                                  verbose=True if LOGGER else False)

    if ADD_FAKE_TEST_IMAGES:
        fake_test_ds = cnn_utils.create_dataset(f'{DATASET_PATH}test/REAL/', False, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED, LOGGER)
        if LOGGER: print('----------')

        fake_test_results = model.evaluate(fake_test_ds,
                                           verbose=True if LOGGER else False)

    print('----------')

    print("Train Accuracy:", train_acc[-1])
    print("Validate Accuracy:", val_acc[-1])
    print("Test Accuracy:", test_results[1])

    print('----------')

    print("Train Loss:", train_loss[-1])
    print("Validate Loss:", val_loss[-1])
    print("Test Loss:", test_results[0])

    print('----------')

    end_time = datetime.now()

    ########## SQLite Database ##########

    # https://sqliteviewer.app/

    data = {'dataset_name':      DATASET_NAME, 
            'test_acc':          test_results[1], 
            'test_loss':         test_results[0], 
            'train_acc_arr':     [round(x, 5) for x in train_acc],
            'train_loss_arr':    [round(x, 5) for x in train_loss],
            'val_acc_arr':       [round(x, 5) for x in val_acc],
            'val_loss_arr':      [round(x, 5) for x in val_loss],

            'train_image_count': TRAIN_IMAGE_COUNT,
            'test_image_count':  TEST_IMAGE_COUNT,
            'fake_image_ratio':  FAKE_IMAGE_RATIO,
            'batch_size':        BATCH_SIZE,
            'epochs':            EPOCHS,

            'dataset_size':      dataset_utils.get_dir_size(DATASET_PATH),
            'execution_time_sec':    (end_time - start_time).total_seconds(),
            }
    
    if(ADD_FAKE_TEST_IMAGES):
        data.update({'fake_test_acc':  fake_test_results[1], 
                     'fake_test_loss': fake_test_results[0], })
        
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    sqlite_utils.create_table(cursor, data)

    sqlite_utils.insert_data(cursor, data)

    conn.commit()

    conn.close()

    ########## Delete Dataset ##########

    # os.remove(DATASET_PATH)

########## ########## ##########

if __name__ == "__main__":
    step = 0

    train_image_count = [200, 1000, 5000, 20000] # [100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000]
    fake_image_ratio = [0.0, 0.25, 0.5, 0.75, 1.0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_trials = 50

    database_path = './data/deviation.db'
    if not os.path.exists(database_path): quit()

    ########## Trials Generator ##########

    trial_permutations = [[x, y] for x in train_image_count for y in fake_image_ratio]

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    tables = cursor.execute(
        """SELECT name FROM sqlite_master WHERE type='table' AND name='data';"""
        ).fetchall()
    
    if (len(tables) > 0):
        trial_history = sqlite_utils.get_trial_history(cursor, 'train_image_count', 'fake_image_ratio')
    else:
        trial_history = []
    
    conn.close()

    trials = []
    for trial in trial_permutations:
        for i in range(max(num_trials - trial_history.count(trial), 0)):
            trials.append(trial)
    
    ########## ########## ##########

    for trial in trials:
        print(f'----------\nTraining Model {step + 1}/{len(trials)}\n----------')

        main(TRAIN_IMAGE_COUNT=trial[0], 
            FAKE_IMAGE_RATIO=trial[1],
            TEST_IMAGE_COUNT=1000,
            EPOCHS=15,

            ADD_FAKE_TEST_IMAGES=True,

            DATABASE_PATH=database_path,
            
            LOGGER=False,
            )
        
        step += 1
