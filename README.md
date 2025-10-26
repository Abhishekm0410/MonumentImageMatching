# MonumentImageMatching
Monument Image Matcher using LoFTR

This project uses Kornia’s LoFTR (Local Feature Transformer) — a deep learning-based feature matching network — to identify whether a given test image belongs to the Ajanta Caves dataset.

It compares the test image with all reference images in a folder and determines the best match based on inlier correspondences using a Fundamental Matrix estimated by RANSAC.

🚀 Features

Uses LoFTR (outdoor pretrained model) for feature matching

Automatically detects keypoints and matches correspondences

Estimates Fundamental Matrix via RANSAC for robust matching

Computes an inlier count score for similarity measurement

Identifies if the image likely belongs to Ajanta Caves dataset

🧠 How It Works

Loads the test image and dataset images.

Extracts feature correspondences using LoFTR.

Applies RANSAC to find geometric inliers.

Computes an inlier score to measure match strength.

Outputs whether the test image matches the Ajanta Caves dataset.

🧩 Requirements

Make sure you have Python ≥ 3.8 and install dependencies:

pip install torch torchvision torchaudio
pip install kornia opencv-python tqdm

📁 Project Structure
.
├── test/
│   └── Ajanta Caves/
│       ├── ajanta1.jpg
│       ├── ajanta2.jpg
│       └── ...
├── loftr_matcher.py
└── README.md

⚙️ Usage

Place your test image (e.g., ajanta.jpg) in the same directory as the script.

Place reference images in test/Ajanta Caves/.

Run the script:

python loftr_matcher.py


Enter the image name when prompted:

Enter the test image filename (example: ajanta.jpg): ajanta.jpg

🧾 Example Output
Compared with ajanta1.jpg, Score: 68
Compared with ajanta2.jpg, Score: 73
Compared with ajanta3.jpg, Score: 15

✅ This is likely Ajanta Caves! (Best match score: 73)


If the best match score ≤ 40:

❌ This is likely NOT Ajanta Caves. (Best match score: 28)

🔧 Adjusting Threshold

You can tweak the threshold in the final section of the script:

if max_score > 40:
    print("✅ This is likely Ajanta Caves!")


Increase the threshold for stricter matching, or lower it for more lenient detection.

🧑‍💻 Author

Abhishek Maheshwari
📍 VIT Vellore


🧠 References

LoFTR – “Detector-Free Local Feature Matching with Transformers” (CVPR 2021) (PDF): https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html
 

LoFTR arXiv preprint: https://arxiv.org/abs/2104.00680
 

Kornia documentation page – image matching / LoFTR: https://kornia.readthedocs.io/en/latest/models/loftr.html
 


Kornia GitHub repository: https://github.com/kornia/kornia
 
