import json
import numpy as np


class thing:

    def __init__(self):
        self.reuslt_id = list()

    @staticmethod
    def save2json(file, filename):
        with open(filename, 'a') as json_file:
            json.dump(file, json_file)

    @staticmethod
    def loadjson(filename):
        with open(filename) as json_file:
            person = json.load(json_file)
        return person

    def get_person_instance(self, original_instance, instance_save_path):
        original_data = self.loadjson(original_instance)

        categories = original_data['categories']
        info = original_data['info']
        licenses = original_data['licenses']

        annotations = original_data['annotations']
        images = original_data['images']

        print('or_a:', str(len(annotations)), 'or_image:', str(len(images)))
        new_data = dict()

        new_annotations = [ele for ele in annotations if ele['category_id'] == 1]

        imgs_id_list = [ele['image_id'] for ele in new_annotations]

        imgs_id_list = list(set(imgs_id_list))

        new_images = [ele for ele in images if ele['id'] in imgs_id_list]

        print('new_a:', str(len(new_annotations)), 'new_image:', str(len(new_images)))

        new_data['info'] = info
        new_data['licenses'] = licenses
        new_data['images'] = new_images
        new_data['annotations'] = new_annotations
        new_data['categories'] = categories
        # self.save2json(filename=instance_save_path,
        #                file=new_data)
        self.reuslt_id = imgs_id_list

    def get_person_result(self, filename, result_save_path):

        content = self.loadjson(filename)

        result = []
        for ele in content:
            if ele['category_id'] == 1 and ele['image_id'] in self.reuslt_id:
                result.append(ele)
        self.save2json(filename=result_save_path,
                       file=result)

    def get_test_result_and_instances(self, original_instance, instance_save_path):
        original_data = self.loadjson(original_instance)

        categories = original_data['categories']
        info = original_data['info']
        licenses = original_data['licenses']

        annotations = original_data['annotations']
        images = original_data['images']

        new_data = dict()

        new_annotations = [annotations[0], annotations[1]]

        new_images = [ele for ele in images if ele['id'] == annotations[0]['image_id'] or ele['id'] == annotations[1]['image_id']]

        new_data['info'] = info
        new_data['licenses'] = licenses
        new_data['images'] = new_images
        new_data['annotations'] = new_annotations
        new_data['categories'] = categories
        self.save2json(filename=instance_save_path,
                       file=new_data)

    def get_test_result(self, save_path):
        res = [{"image_id": 289343, "category_id": 18, "bbox": [250.0, 287.0, 52.0, 13.0], "score": 0.6855026483535767}]
        self.save2json(filename=save_path,
                       file=res)


if __name__ == '__main__':
    t = thing()
    t.get_person_instance('./coco_data/instances_val2017.json', './person_instances_val2017.json')
    t.get_person_result('./trained16_w_person_2017_result.json', './trained16_w_filtered_person_2017_result.json')
    # t.get_test_result_and_instances('./coco_data/instances_val2017.json', './test_instances.json')
    # t.get_test_result('./test_result.json')