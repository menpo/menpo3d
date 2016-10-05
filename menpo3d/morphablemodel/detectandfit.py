from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import load_balanced_frontal_face_fitter

detect = load_dlib_frontal_face_detector()
fitter = load_balanced_frontal_face_fitter()


def detect_and_fit(image):
    bboxes = detect(image)
    if len(bboxes) == 0:
        raise ValueError('no detections!')
    fr = fitter.fit_from_bb(image, bboxes[0])
    image.landmarks['ibug68'] = fr.final_shape
