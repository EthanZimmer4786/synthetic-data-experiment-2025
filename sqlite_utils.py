import sqlite3

type_map = {'str': 'text',
            'list': 'text',
            'int': 'integer',
            'float': 'real',
           }

########## Ensure Table Is Created ##########

def create_table(cursor, data):
    """Creates a sqlite table from a dictionary"""
    table_arr = []
    for i in range(len(data.keys())):
        if (type_map.get(type(list(data.values())[i]).__name__), -1) == -1:
            print('Unrecognized value type in create_table')
            quit()

        table_arr.append(f'{list(data.keys())[i]} {type_map.get(type(list(data.values())[i]).__name__)}')

    table_str = str(table_arr).replace('\'','')[1:-1]

    cursor.execute(f'CREATE TABLE IF NOT EXISTS data({table_str})')

    # cursor.execute(f'''CREATE TABLE IF NOT EXISTS data(
    #                dataset_name text, 
    #                test_acc real, test_loss real,
    #                train_acc_arr text, train_loss_arr text, 
    #                val_acc_arr text, val_loss_arr text,

    #                train_image_count integer, test_image_count integer,
    #                fake_image_ratio real,
    #                batch_size integer, epochs integer
    #                )''')

########## Insert Row Of Data ##########

def insert_data(cursor, data):
    value_arr = []
    for key in list(data.keys()):
        if type_map.get(type(data.get(key)).__name__) == 'text':
            value_arr.append(str(data.get(key)))
        else:
            value_arr.append(data.get(key))

    value_arr = str(value_arr)[1:-1]

    cursor.execute(f'INSERT INTO data VALUES ({value_arr})')

########## Get Previous Trials ##########

def get_trial_history(cursor, *args):
    trials = []

    for entry in args:
        cursor.execute(f"SELECT {entry} FROM data")
        data_arr = cursor.fetchall()

        # Run once on first iteration to set size of groups
        while len(trials) != len(data_arr):
            trials.append([])

        for i in range(len(data_arr)):
            trials[i].append(data_arr[i][0])

    return(trials)