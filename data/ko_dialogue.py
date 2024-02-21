import os
import json
import tarfile
import multiprocessing
import argparse

from tqdm import tqdm
from glob import glob
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from g2pk import G2p
from lhotse.audio.recording import Recording
from lhotse.audio.recording_set import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet


class SexCode(Enum):
    MALE = 'M'
    FEMALE = 'F'

class AgeCode(Enum):
    CHILD = 'C'
    TEENAGER = 'T'
    ADULT = 'A'
    SENIOR = 'S'
    ECT = 'Z'

class DialectCode(Enum):
    SEOUL = 1
    KANGWON = 2
    CHOONGCHUN = 3
    KYEONGSANG = 4
    JEONLA = 5
    JEJU = 6
    ETC = 9
    
class ReferenceCode(Enum):
    BROADCAST = 1
    MADE = 2
    CROWDSOURCE = 3
    ETC = 9
    
class QualityCode(Enum):
    GOOD = 1
    NOISE = 2
    BAD = 3
    REMOTE = 4
    

G2P = G2p()
DIR_PREFIX = os.path.join('data', 'remote', 'PROJECT', 'AI학습데이터', 'KoreanSpeech', 'data')

def set_root_path(root_path):
    global DIR_PREFIX
    DIR_PREFIX = os.path.join(root_path, DIR_PREFIX)


def unzip(targz_file, dest):    
    os.makedirs(dest, exist_ok=True)        
    with tarfile.open(targz_file, 'r:gz') as tar:
        tar.extractall(dest)
    
def unzip_all(root_dir):
    tar_files = glob(os.path.join(root_dir, '**', '*.tar.gz'), recursive=True)
    print(f'{len(tar_files)} tar.gz files found.')
    
    for i in tqdm(range(len(tar_files))):
        unzip(tar_files[i], root_dir)


class MetaData():
    def __init__(self, data, filter=None) -> None:
        self.valid = False
        
        self.file_path = os.path.join(DIR_PREFIX, data[0][1:])
        # self.subject = data[1]
        # self.subject_detail = data[2]
        self.sex = data[3]
        self.age = data[4]
        # self.living = data[5]
        self.dialect = int(data[6])
        self.reference = int(data[7])
        self.quality = int(data[8])


        if not os.path.exists(self.file_path):
            return

        # filtering by attributes
        if not self.__filter__(filter):
            return
        
        file_name = os.path.basename(self.file_path).split(".")[0]
        dir_path = os.path.dirname(self.file_path)
        # read meta json
        meta_json_path = os.path.join(dir_path, file_name + '.json')
        if not os.path.exists(meta_json_path):
            return

        with open(meta_json_path, 'r', encoding='utf-8') as f:
            meta_json = json.load(f)
            self.start = meta_json["start"]
            self.end = meta_json["end"]
            self.length = meta_json["length"]
            self.id = meta_json['metadata']
        # read text
        text_path = os.path.join(dir_path, file_name + '.txt')
        if not os.path.exists(text_path):
            return
        
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = G2P(f.read())
        
        self.valid = True
        
    def __filter__(self, filter):
        if filter is not None:
            for key, item in filter.items():
                value = getattr(self, key)
                if value in item:
                    return False
                
        return True


def read_meta_data(file_path, filter):
    metadata_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            metadata = MetaData(line.strip().split(' | '), filter)
            if metadata.valid:
                metadata_list.append(metadata)

    return metadata_list


def get_metadata_list(root_path, filter=None):
    metadata_files = glob(os.path.join(root_path, '**', '*_metadata.txt'), recursive=True)
    metadata_list = []
    
    # using thread
    task = lambda metadata_file: metadata_list.extend(read_meta_data(metadata_file, filter))
    
    thread_count = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        executor.map(task, metadata_files)
    
    return metadata_list



def create_manifest(metadata_list, dest):
    recordings = RecordingSet.from_recordings(
        Recording.from_file(metadata.file_path, recording_id=metadata.id) for metadata in metadata_list
    )
    supervisions = SupervisionSet.from_segments(
        SupervisionSegment(
            id=metadata.id,
            recording_id=metadata.id,
            start=metadata.start,
            duration=recordings[metadata.id].duration,
            text=metadata.text,
            language='Korean'
        ) for metadata in metadata_list
    )

    # Save manifests
    recordings.to_jsonl(os.path.join(dest, 'ljspeech_recordings_all.jsonl.gz'))
    supervisions.to_jsonl(os.path.join(dest, 'ljspeech_supervisions_all.jsonl.gz'))


def main(args):
    data_path = args.data
    set_root_path(data_path)

    # unzip
    if args.unzip:
        unzip_all(data_path)

    # create filter
    filter = {}
    if args.sex:
        filter['sex'] = [args.sex]
    if args.age:
        filter['age'] = args.age
 
    metadata_list = get_metadata_list(data_path, filter)
    print(f'{len(metadata_list)} data imported.')

    create_manifest(metadata_list, data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='korean dialouge data path')
    parser.add_argument('--unzip', action='store_const', const=True, default=False, help='unzip .tar.gz files')
    parser.add_argument('--sex', help='M or F')
    parser.add_argument('--age', type=list, help='["C"(child), "T"(teenager), "A"(adult), "S"(senior), "F"(etc)]')
    main(parser.parse_args())