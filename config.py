opt = {
    'opt': 'with_accents',
    'ipt': 'without_accents',
    'max_len_load' : 100000,
    'train_size': 99998,
    'val_size': 1,
    'test_size': 1,
    'filename': 'train_tieng_viet.txt',
    'seq_len': 500,
    'batch_size': 8,
    'num_epochs': 10,
    'lr': 10**-4,
    'd_model': 512,
}

'''
max_len_load = train_size + val_size + test_size
'''