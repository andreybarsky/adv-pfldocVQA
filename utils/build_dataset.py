import numpy as np
from collections import defaultdict
import os
from PIL import Image
import pickle
import tqdm
import random

random.seed(0)

def train_val_stat():
    res = []
    #dataset_path = f'/data/shared/PFL-DocVQA/imdb/1.0centralized/imdb_val.npy'
    dataset_path = "/data/shared/PFL-DocVQA/imdb/1.2/red_train_both.npy"
    data = np.load(dataset_path, allow_pickle=True)[1:]
    dataset_path = f'/data/shared/PFL-DocVQA/imdb/1.0centralized/imdb_val.npy'
    data1 = np.load(dataset_path, allow_pickle=True)[1:]

    blue_path = '/data/shared/PFL-DocVQA/images/blue'
    red_path = '/data/shared/PFL-DocVQA/images/red'

    blue_files = set(os.listdir(blue_path))
    red_files = set(os.listdir(red_path))

    blue_data = 0
    red_data = 0
    both = 0
    seen = set()
    import tqdm
    for sample in tqdm.tqdm(data):
        answers = sample["answers"].split("\n")
        question = sample["question"]
        #for answer in answers:
        res.append((question,answers))
        if sample["image_name"] not in seen:
            seen.add(sample["image_name"])
        else:
            continue        
        img_id = sample["image_name"]+".jpg"
        if img_id in blue_files:
            blue_data += 1
        if img_id in red_files:
            red_data += 1
        if img_id in blue_files and img_id in red_files:
            both += 1 
    
    print(blue_data, red_data, both)

    for sample in data1:
        answers = sample["answers"].split("\n")
        question = sample["question"]
        #for answer in answers:
        res.append((question,answers))
        
    print(len(res))

def build_elsa_release():
    all_samples = []

    blue_dataset_train_path = '/data/shared/PFL-DocVQA/imdb/1.0/blue/'
    image_directory = "/data/shared/PFL-DocVQA/images/full"
    # blue + red paths
    directories = ['/data/shared/PFL-DocVQA/imdb/1.0/blue/blue_valid.npy',
                   '/data/shared/PFL-DocVQA/imdb/1.0/blue/blue_test.npy',
                   '/data/shared/PFL-DocVQA/imdb/1.0/red/red_train.npy',
                   '/data/shared/PFL-DocVQA/imdb/1.0/red/red_test.npy',
                   '/data/shared/PFL-DocVQA/imdb/1.0/red/red_test_negative.npy',
                   '/data/shared/PFL-DocVQA/imdb/1.0/red/red_test_positive.npy']
    directories = ['/home/mpintore/blue/test.npy']
    full_images_name = set(os.listdir(image_directory)) # all the images are .jpg

    for i in range(0,10):
        client_path = blue_dataset_train_path + f'blue_train_client_{i}.npy'
        blue_train = np.load(client_path, allow_pickle=True)[1:]
        all_samples.extend(blue_train)
    
    # Build the unsplitted dataset
    for directory in directories:
        data = np.load(directory, allow_pickle=True)[1:]
        all_samples.extend(data)
    
    # Merge them
    res = defaultdict(lambda: {"image": None, "questions": defaultdict(set)})

    for sample in tqdm.tqdm(all_samples):
        # Add pair question, answers
        id_image = sample["image_name"]
        question = sample["question"]
        answers = sample["answers"].split("\n")
        answers.sort()
        res[id_image]["questions"][question].update(answers)

    print('The dataset is ready')

    # filter out keeping samples with n_questions questions
    n_questions = 5
    res = {k: v for k, v in res.items() if len(v["questions"]) == n_questions}
    print('Len filtered dataset = ', len(res))

    n_samples = 1000
    if len(res) >= n_samples:
        sampled_res = dict(random.sample(list(res.items()), n_samples))

    # Finally load the images
    large_images = 0
    for id_image, _ in tqdm.tqdm(sampled_res.items()):
        image_path = os.path.join(image_directory, id_image) + ".jpg"
        
        if id_image+".jpg" in full_images_name:
            try:
                with Image.open(image_path) as image:
                    res[id_image]["image"] = image.convert("RGB")
                if image.size > (1000,1000):
                    large_images += 1
            except Exception as e:
                print(f"Error  {id_image}: {e}")
        else:
            print(f"Missing file: {image_path}")
    print('large images = ', large_images)
    # from itertools import islice

    # n_samples = 500
    # sampled_res = dict(islice(sampled_res.items(), n_samples))

    with open(f'/home/mpintore/github/adv_docVQA/utils/advdoc_data_nsampl{n_samples}_nqst{n_questions}.pkl','wb') as f:
        pickle.dump(sampled_res,f)
        
    return sampled_res

if __name__ == '__main__':
    build_elsa_release()
    # train_val_stat()