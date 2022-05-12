
from core.recognition.openvinoreid import OpenVinoReidentifier
from core.commons.pose_estimator_openvino import HeadPosEstimator
import core.pipeline.tools as tools

import cv2

reid = OpenVinoReidentifier(
    "/home/chaki/Projects/gods_eye/core/weights/intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095")
headpose = HeadPosEstimator(
    "/home/chaki/Projects/gods_eye/core/weights/intel/whenet/FP32/whenet_224x224.xml")


def extract_features(path):
    features = []
    paths = []
    ids = []
    alignment_error = []
    for directory in path:
        if not os.path.isdir(directory):
            continue
        id = directory.split("/")[-1]
        for image in os.listdir(directory):
            image_path = os.path.join(directory, image)
            if image_path.endswith(".jpg"):
                image = cv2.imread(image_path)
                headpose.process(image)
                features.append(reid.encode(image))
                if headpose.job_finished():
                    # ? if condition checks if the job is finished - might be replaced by 'while'
                    head_pose_output = headpose.get_output()
                    # ? A function that returns the error based on headpose mainly, powered by blur and size errors
                    error = tools.compute_errors(
                        image, head_pose_output, None, None
                    )

                    # ? A function that returns the error based on head_pose mainly, powered by blur and size errors
                paths.append(image_path)
                ids.append(id)
                alignment_error.append(error)

    return ids, features, paths, alignment_error


id, features, paths, alignment_error = extract_features(directories)

features = [[*feature] for feature in features]
data = {'id': id,
        'path': paths,
        'features_openvino': features,
        'alignment_error': alignment_error
        }

# Create DataFrame
openvino = pd.DataFrame(data)
openvino.to_csv(
    "/home/chaki/Projects/gods_eye/output/12_04-21_11-25-09/in/openvino_data.csv")
