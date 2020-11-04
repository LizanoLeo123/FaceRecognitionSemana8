import numpy as np
import argparse
import cv2
import os
import pandas as pd
import urllib
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def load_local_images():
    img_df = pd.DataFrame()

    for person_dir in os.scandir(args["directory"]):
        if person_dir.is_dir():
            for file in os.scandir(person_dir.path):
                if file.is_file() and (file.path.endswith(".jpg")
                                       or file.path.endswith(".png")
                                       or file.path.endswith(".jpeg")):
                    temp_df = load_local_image_to_df(file.path, person_dir.name)
                    if img_df.empty:
                        img_df = temp_df
                    else:
                        img_df = pd.concat((img_df, temp_df), ignore_index=True)
    print(img_df)
    img_df.to_csv(args["saveFile"], index=False)


def load_cloud_images():
    img_df = pd.DataFrame()

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    dir_list = drive.ListFile({'q': "'1IhufSGU5Llbp_3pO8lw2A4g3cz12IlRB' in parents and trashed=false"}).GetList()

    for person_dir in dir_list:
        file_list = drive.ListFile({'q': "'" + person_dir['id'] + "' in parents and trashed=false"}).GetList()
        for file in file_list:
            temp_df = load_cloud_image_to_df(file['id'], person_dir["title"])
            if img_df.empty:
                img_df = temp_df
            else:
                img_df = pd.concat((img_df, temp_df), ignore_index=True)
    print(img_df.info())
    img_df.to_csv(args["saveFile"], index=False)


def load_local_image_to_df(img_source, target_name):
    img_array = cv2.resize(cv2.imread(img_source, cv2.IMREAD_GRAYSCALE), (100, 100))
    img_array = (img_array.flatten())
    img_array = img_array.reshape(-1, 1).T
    df = pd.DataFrame(img_array)
    df['target'] = pd.Series(target_name, index=df.index)
    return df


def load_cloud_image_to_df(img_id, target_name):
    resp = urllib.request.urlopen("https://drive.google.com/uc?export=view&id=" + img_id)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    img_array = cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE), (100, 100))
    img_array = (img_array.flatten())
    img_array = img_array.reshape(-1, 1).T
    df = pd.DataFrame(img_array)
    df['target'] = pd.Series(target_name, index=df.index)
    return df


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory",
                default=".\Dataset",
                help="path to image directory")
ap.add_argument("-sf", "--saveFile", type=str,
                default="./data.csv",
                help="path to the csv file to store the data")
args = vars(ap.parse_args())

load_local_images()
# load_cloud_images()
