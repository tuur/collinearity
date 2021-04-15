#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data reading / handling for dys / xer datasets"""

__author__      = "Tuur Leeuwenberg"
__email__ = "A.M.Leeuwenberg-15@umcutrecht.nl"


from lib.SPSSReader import read_sav
import pickle, os
import base64
import hashlib
from cryptography.fernet import Fernet

# Source https://nitratine.net/blog/post/encryption-and-decryption-in-python/
def save_python_object_encrypted(obj, file_path, password):
    key = base64.b64encode(hashlib.md5(password.encode()).hexdigest().encode())
    byte_obj = pickle.dumps(obj)
    crypt = Fernet(key)
    encoded = crypt.encrypt(byte_obj)
    with open(file_path, 'wb') as f:
        f.write(encoded)
        print('written encrypted file to', file_path)

def load_python_object_encrypted(file_path, password):
    key = base64.b64encode(hashlib.md5(password.encode()).hexdigest().encode())
    crypt = Fernet(key)
    with open(file_path, 'rb') as f:
        encoded = f.read()
    byte_obj = crypt.decrypt(encoded)
    obj = pickle.loads(byte_obj)
    return obj

class CITOR:
    dev_path = "/media/sf_HTx/C_Data/4 Final_data/1 Unimputed/CITOR.development.data.sav"
    val_path = "/media/sf_HTx/C_Data/4 Final_data/1 Unimputed/CITOR.validatie.data.sav"

    def __init__(self):
        self.val_data, _ = read_sav(self.val_path)
        self.dev_data, _ = read_sav(self.dev_path)



#citor_data = CITOR()
#save_python_object_encrypted(citor_data,"/media/sf_HTx/C_Data/4 Final_data/1 Unimputed/CITOR.data.encr","????")

#print('done')