# Fashion Detection and Vibe Analysis System

This system analyzes video content to detect clothing items and determine the aesthetic vibe of the content using AI models. It combines computer vision for clothing detection with natural language processing for vibe analysis.

## Features

- Real-time clothing detection in videos
- Similar product matching using FAISS vector search
- Aesthetic vibe analysis using Gemini AI
- Frame extraction and processing
- JSON output with detection results and vibe analysis

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
ROBOFLOW_API_KEY=your_roboflow_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Required Files

1. FAISS Index Files:
   - `fashio_index.faiss`: FAISS index for similar product matching
   - `image_ids.npy`: Corresponding image IDs for the FAISS index

2. Video and Caption Files:
   - Place your videos in the `videos/` directory
   - Place corresponding caption files in the `videos/` directory
   - Format: `reel_X.mp4` and `reel_X_caption.txt`

## Usage

1. Prepare your video and caption files:
   - Place your video file in the `videos/` directory
   - Create a corresponding caption file with the same name (e.g., `reel_3.mp4` and `reel_3_caption.txt`)

2. Run the detection script:
```bash
python models/detection.py
```

## Output

The system generates the following outputs:

1. Processed Video:
   - Location: `output_<video_name>.mp4`
   - Contains bounding boxes around detected clothing items
   - Shows similar product matches

2. Extracted Frames:
   - Location: `frames/<video_name>_frame_<number>.jpg`
   - Contains frames where clothing was detected with confidence > 0.5

3. JSON Results:
   - Location: `output/detection_results_<video_name>.json`
   - Contains:
     - Video ID
     - Detected vibes (aesthetic categories)
     - Detected products with confidence scores
     - Matched product IDs

## System Workflow

1. Video Processing:
   - Reads video frames at 4 FPS
   - Detects clothing items using YOLOv8 model
   - Extracts frames with high-confidence detections

2. Similar Product Matching:
   - Uses CLIP model to generate image embeddings
   - Matches detected items with similar products using FAISS index
   - Returns top 3 similar products

3. Vibe Analysis:
   - Analyzes video caption using Gemini AI
   - Classifies content into aesthetic categories
   - Returns top 2-3 matching vibes

## Supported Aesthetic Vibes

1. Coquette – soft, flirty, feminine, romantic
2. Clean Girl – minimalist, neutral, glowy, polished
3. Cottagecore – nature-inspired, vintage, cozy, rural
4. Streetcore – urban, bold, oversized, edgy
5. Y2K – nostalgic 2000s, shiny, techy, playful
6. Boho – earthy, free-spirited, eclectic, festival
7. Party Glam – bold, sparkly, luxurious, nightlife

## Directory Structure

```
.
├── models/
│   └── detection.py
├── videos/
│   ├── reel_*.mp4
│   └── reel_*_caption.txt
├── frames/
├── output/
├── .env
└── README.md
```

## Troubleshooting

1. If you encounter encoding errors with caption files:
   - Ensure caption files are saved in UTF-8 encoding
   - The system will automatically handle encoding issues with the `errors='replace'` parameter

2. If FAISS index files are missing:
   - Ensure `fashio_index.faiss` and `image_ids.npy` are present in the root directory
   - These files are required for similar product matching

3. If GPU memory issues occur:
   - Reduce batch size in the detection model
   - Process videos at a lower resolution
