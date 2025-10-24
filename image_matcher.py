import os
from kornia.feature import LoFTR
import kornia as K
import torch
import cv2
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loftr = LoFTR(pretrained='outdoor').to(device)

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    img = K.image_to_tensor(img, False).float() / 255.0
    return img.to(device)

def match_images(img1_path, img2_path):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    input_dict = {"image0": img1, "image1": img2}
    with torch.no_grad():
        correspondences = loftr(input_dict)

    if len(correspondences['keypoints0']) < 8:
        return 0.0

    F_matrix, mask = cv2.findFundamentalMat(
        correspondences['keypoints0'].cpu().numpy(),
        correspondences['keypoints1'].cpu().numpy(),
        cv2.FM_RANSAC
    )

    if F_matrix is None:
        return 0.0

    inlier_count = mask.sum()
    return inlier_count

# ====================== RUNNING ======================

test_image_name = input("Enter the test image filename (example: ajanta.jpg): ").strip()
test_image_path = test_image_name  # Test image is in same folder

ajanta_folder = 'test/Ajanta Caves'  # SIMPLE PATH, not overcomplicated

max_score = 0
for ajanta_img in sorted(os.listdir(ajanta_folder)):
    ajanta_img_path = os.path.join(ajanta_folder, ajanta_img)
    score = match_images(test_image_path, ajanta_img_path)
    print(f"Compared with {ajanta_img}, Score: {score}")
    if score > max_score:
        max_score = score

if max_score > 40:  # You can adjust threshold
    print(f"\n✅ This is likely Ajanta Caves! (Best match score: {max_score})")
else:
    print(f"\n❌ This is likely NOT Ajanta Caves. (Best match score: {max_score})")
