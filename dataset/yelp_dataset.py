import jsonlines
import re
import torch
from torch.utils.data import Dataset
from os import path
from torchvision import transforms
from PIL import Image
from util.random_augment import RandomAugment


class YelpDataset(Dataset):
    def _load_transform(self, json_name):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        if 'train' in json_name:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=self.config.image_res, 
                    scale=(0.5, 1.0), 
                    interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity', 'AutoContrast', 
                        'Equalize', 'Brightness', 
                        'Sharpness', 'ShearX', 'ShearY', 
                        'TranslateX', 'TranslateY', 'Rotate'
                    ]
                ),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    size=(self.config.image_res, self.config.image_res),
                    interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ])

    def __init__(self, json_name: str, config) -> None:
        super().__init__()
        self.config = config
        self.dataset_root = '/home/pris/dw/project/datasets/yelp'
        self.json_path = path.join(self.dataset_root, json_name)
        self.img_root = path.join(self.dataset_root, 'image')
        self._load_json()
        self._text_preprocess()
        self._load_transform(json_name)

    def __item__(self, index):
        entry = self._entries[index]
        label = int(entry['label']) - 1

        imgs = torch.zeros(self.config.num_img, 3, 384, 384)
        cnt = 0
        try:
            photos = entry['photos']
            for im in photos:
                im_id = im['_id']
                image_path = path.join(
                    self.img_root,
                    im_id + '.jpg'
                )
                if path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')   
                    # image = Image.new('RGB', (256, 256), (255, 255, 255))
                    image = self.transform(image)
                    imgs[cnt] = image
                    cnt += 1
                    if cnt == self._max_num_img:
                        break
        except KeyError:
            pass
        
        return entry['id'], imgs, entry['text'], label
    
    def __len__(self):
        return len(self._entries)

    def _load_json(self):
        self._entries = []
        with open(self.json_path, "r", encoding="utf8") as f:
            for sample in jsonlines.Reader(f):
                self._entries.append({
                        "label": sample["Rating"],
                        "id": sample["_id"],
                        "text": sample["Text"],
                        "photos": sample["Photos"],
                    }
                )        
    def _text_preprocess(self):
        for i in range(len(self._entries)):
            url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
            self._entries[i]['text'] = re.sub(url_pattern, '', self._entries[i]['text'])
            tag_pattern = '#[a-zA-Z0-9]*'
            self._entries[i]['text'] = re.sub(tag_pattern, '', self._entries[i]['text'])
            at_pattern = '@[a-zA-Z0-9]*'
            self._entries[i]['text'] = re.sub(at_pattern, '', self._entries[i]['text'])
            not_ascii_pattern = '[^a-zA-Z0-9|]'
            self._entries[i]['text'] = re.sub(not_ascii_pattern, ' ', self._entries[i]['text'])
            self._entries[i]['text'] = self._entries[i]['text'].replace('|||', '[SEP]')
            self._entries[i]['text'] = re.sub(' +', ' ', self._entries[i]['text'])
            self._entries[i]['text'] = self._entries[i]['text'].strip()
            if self._entries[i]['text'][-5:] != '[SEP]':
                self._entries[i]['text'] = self._entries[i]['text'] + '[SEP]'
            else:
                self._entries[i]['text'] = self._entries[i]['text']