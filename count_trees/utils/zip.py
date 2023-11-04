import os
import zipfile
from tqdm import tqdm
from pathlib import Path
from glob import glob 


def zip_folder(root_path,name_zip):
    base_diname = os.path.basename(root_path)
    dirname = os.path.dirname(root_path)
    files = Path(root_path).glob('**/*')

    files = filter(lambda x: filter_files(str(x)), files)

    with zipfile.ZipFile(name_zip, mode="w") as archive:
        files = tqdm(files)
        for file_path in files: 
            new_name_file = str(file_path).replace(dirname,'')
            files.set_description(f"Adding {new_name_file}")  
            archive.write(file_path, arcname=new_name_file)


def unzip(file_path,destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

