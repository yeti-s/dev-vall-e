import os
import json
import tarfile
import multiprocessing
import argparse

from tqdm import tqdm
from glob import glob
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from g2pk import G2p
from lhotse.audio.recording import Recording
from lhotse.audio.recording_set import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet

NUM_THREADS = multiprocessing.cpu_count()
DIR_PREFIX = os.path.join('data', 'remote', 'PROJECT', 'AI학습데이터', 'KoreanSpeech', 'data')

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
    def __init__(self, data, g2p, filter=None) -> None:
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
            self.text = g2p(f.read())
        
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
    print(f'=== find *_metadata.txt in {root_path}')
    metadata_files = glob(os.path.join(root_path, '**', '*_metadata.txt'), recursive=True)
    num_metadata_files = len(metadata_files)
    print(f'-- {num_metadata_files} files found.')

    # define task
    def task(lines):
        g2p = G2p()
        metadata_sub_list = []
        
        for line in lines:
            metadata = MetaData(line.strip().split(' | '), g2p, filter)
            if metadata.valid:
                metadata_sub_list.append(metadata)
                
        return metadata_sub_list
        
    metadata_list = []

    for i in tqdm(range(num_metadata_files)):
        metadata_file = metadata_files[i]
        
        with open(metadata_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            num_lines = len(lines)
            
            # process with multi thread
            if NUM_THREADS > 1:
                unit_tasks = num_lines // NUM_THREADS
                with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                    futures = [executor.submit(task, lines[unit_tasks*t:unit_tasks*(t+1)]) for t in range(NUM_THREADS-1)]
                    futures.append(executor.submit(task, lines[unit_tasks*(NUM_THREADS-1):]))

                    for future in as_completed(futures):
                        metadata_list.extend(future.result())
            else:
                metadata_list = task(lines)
        
    print(f'-- {len(metadata_list)} data created.')
    
    return metadata_list



def create_manifest(metadata_list, dest):
    print(f'=== create manifest from {len(metadata_list)} metadata')
    
    num_data = len(metadata_list)
    # create recordings & supervisions
    recordings = []
    supervisions = []
    print(f'-- create recordings')
    
    def task(metadata_sub_list):
        sub_recordings = [Recording.from_file(
            metadata.file_path, 
            recording_id=metadata.file_path
        ) for metadata in metadata_sub_list]
        
        sub_supervisions = [SupervisionSegment(
            id=metadata.file_path,
            recording_id=metadata.file_path,
            start=metadata.start,
            duration=metadata.end,
            text=metadata.text,
            language='Korean',
            custom={'normalized_text': metadata.text.strip()}
        ) for metadata in metadata_sub_list]
        
        return sub_recordings, sub_supervisions
        
    # process with multi threads
    if NUM_THREADS > 1:
        unit_tasks = num_data // NUM_THREADS
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(task, metadata_list[unit_tasks*t:unit_tasks*(t+1)]) for t in range(NUM_THREADS-1)]
            futures.append(executor.submit(task, metadata_list[unit_tasks*(NUM_THREADS-1):]))

            for future in as_completed(futures):
                sub_recordings, sub_supervisions = future.result()
                recordings.extend(sub_recordings)
                supervisions.extend(sub_supervisions)
    # process with single thread
    else:
        recordings, supervisions = task(metadata_list)        
    
    # create recording set & supervision set
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    # export data
    recording_set.to_jsonl(os.path.join(dest, 'ljspeech_recordings_all.jsonl.gz'))
    supervision_set.to_jsonl(os.path.join(dest, 'ljspeech_supervisions_all.jsonl.gz'))
    print(f'-- recordings & supervisions is saved on {dest}')


def main(args):
    data_path = args.data
    out_path = args.out
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
    create_manifest(metadata_list, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='korean dialouge data path (required!)')
    parser.add_argument('--unzip', action='store_const', const=True, default=False, help='unzip .tar.gz files')
    parser.add_argument('--sex', help='M or F')
    parser.add_argument('--age', type=list, help='["C"(child), "T"(teenager), "A"(adult), "S"(senior), "F"(etc)]')
    parser.add_argument('--out', default='data/manifests', help='output directory. default is "data/manifests"')
    main(parser.parse_args())