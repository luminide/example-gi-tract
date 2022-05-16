import pandas as pd

from dataset import coco_prep
from util import get_class_names

def main():
    input_dir = '../input'
    meta_file = f'{input_dir}/train.csv'
    meta_df = pd.read_csv(meta_file)
    category_names = get_class_names(meta_df)
    categories = [{'name': name, 'id': ident} for ident, name in enumerate(category_names, 1)]
    print(f'categories: {categories}')

    img_dir = f'{input_dir}/train'
    for set_name in ['val', 'train']:
        coco_prep(input_dir, set_name, img_dir, meta_file, categories)

if __name__ == '__main__':
    main()
