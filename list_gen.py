import os


def preprocess_faces(path: str = "inference/12_people_min_samples"):

    peoples_dir = path  # os.path.join(os.getcwd(), path)
    peoples = os.listdir(path)

    for people in peoples:
        person_dir = os.path.join(peoples_dir, people)
        # print(person_dir)
        if not os.path.isdir(person_dir):
            continue
        fio = open(f"{person_dir}/img.list", 'w')
        person_dir_list = os.listdir(person_dir)
        for image_path in person_dir_list:  # os.listdir(person_dir):
            image = os.path.join(person_dir, image_path)
            if image.endswith(".jpg"):
                fio.write('{} \n'.format(image))
        # close
        fio.close()
