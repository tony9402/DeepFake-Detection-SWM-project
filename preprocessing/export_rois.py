import os
import pickle
from glob import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN

from utils.export_utils import load_video

# Data
OUT_DIR = "data/rois"
MP4_ROOT = "/workspace/dataset/train/dfdc_train_part_*"

# Face detection
MAX_FRAMES_TO_LOAD = 300
MAX_FACES = 2
FACE_FRAMES = 10
FACEDETECTION_DOWNSAMPLE = 0.5
MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)  # Default [0.6, 0.7, 0.7]
MMTNN_FACTOR = 0.709  # Default 0.709 p

os.makedirs(OUT_DIR, exist_ok=True)

mp4_dirs = sorted(glob(MP4_ROOT))
mp4_dirs_exported = [os.path.basename(d).split('_',1)[1].rsplit('_',1)[0] for d in glob(os.path.join(OUT_DIR, "*"))]
mp4_dirs_unexported = [d for d in mp4_dirs if os.path.basename(d) not in mp4_dirs_exported]

print(f"Found {len(mp4_dirs)} MP4 dirs; {len(mp4_dirs_unexported)} needing export")
keep_all = True if MAX_FACES > 1 else False

if FACEDETECTION_DOWNSAMPLE:
    facedetection_upsample = 1/FACEDETECTION_DOWNSAMPLE
else:
    facedetection_upsample = 1

mtcnn = MTCNN(margin=0, keep_all=keep_all, post_process=False, device='cuda',
              thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)


for i_mp4_dir, mp4_dir in enumerate(mp4_dirs_unexported):
    boxes_by_frame_by_videopath = {}
    mp4_dir_name = os.path.basename(mp4_dir)

    print(f"Dir {mp4_dir} ({i_mp4_dir+1} of {len(mp4_dirs_unexported)})")
    mp4_paths = glob(os.path.join(mp4_dir, "*.mp4"))
    for mp4_path in tqdm(mp4_paths):
        try:
            video, pil_frames = load_video(mp4_path,
                                           every_n_frames=FACE_FRAMES,
                                           to_rgb=True,
                                           rescale=FACEDETECTION_DOWNSAMPLE,
                                           inc_pil=True,
                                           max_frames=MAX_FRAMES_TO_LOAD)

            if len(pil_frames):  # Randomly in 1 committ this was an empty list?!
                boxes, _probs = mtcnn.detect(pil_frames, landmarks=False)  # MAX_FRAMES / FACE_FRAMES boxes (e.g. 150 / 10 -> 15)
                boxes_by_frame_by_videopath[mp4_path] = boxes
        except Exception as e:
            print(f"Error with file {mp4_path}: {e}")

    with open(f"{OUT_DIR}/faces_{mp4_dir_name}_{FACE_FRAMES}.pickle", 'wb') as f:
        pickle.dump(boxes_by_frame_by_videopath, f)
