from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import cv2
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from inference import get_model
import os 
from dotenv import load_dotenv
import numpy as np
import faiss
import pickle
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from datetime import datetime
load_dotenv()
import base64
import os
from google import genai
from google.genai import types

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"Al files {os.getcwd()}")
print(os.listdir())

# Initialize models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = get_model(model_id="my-first-project-zmpms/2",api_key=ROBOFLOW_API_KEY) # use the fine tune model of YOLOV8.  # using now my-first-project-zmpms/2 # before use clothing-detection-s4ioc/6

try:
    path = r"fashio_index.faiss"
    index = faiss.read_index(path)# Load the FAISS index and image IDs
    print(f"Index {index}")
except Exception as e:
    print(f"Not found due to :{e}")


def load_index():
    try:
        path = r"fashio_index.faiss"
        index = faiss.read_index(path)
        print(f"here is index")
        image_ids = np.load(r"image_ids.npy")
        return index, image_ids
    except Exception as e:
        print("Error: Could not load index files. Make sure fashion_index.faiss and image_ids.npy exist.",e)
        return None, None



def get_image_embedding(frame):
    # Convert BGR to RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Preprocess using CLIP processor
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(clip_model.device)

    # Get image features
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    
    # Convert to numpy
    image_features_np = image_features.cpu().numpy().astype(np.float32)

    # Normalize for cosine similarity (very important for IndexFlatIP)
    image_features_np /= np.linalg.norm(image_features_np)

    return image_features_np


def search_similar_images(frame, index, image_ids, k=3):
    # Get normalized embedding
    query_embedding = get_image_embedding(frame)
    query_embedding = query_embedding.reshape(1, -1)  # FAISS expects (1, dim)

    # Search top-k similar
    distances, indices = index.search(query_embedding, k)

    similar_images = []
    for i, idx in enumerate(indices[0]):
        similar_images.append({
            "image_id": image_ids[idx],
            "distance": float(distances[0][i])  # Higher = more similar
        })

    return similar_images

def process_video_frame(frame, index, image_ids):
    # Get detections
    detections = detect_clothing_from_frame(frame)
    
    # Get similar images - after getting visual frames from detecting_clothing... passes to the search_similiar_imaegs 
    similar_images = search_similar_images(frame, index, image_ids)
    
    # Find max confidence detection
    max_confidence = 0
    max_detection = None
    for det in detections:
        if det["confidence"] > max_confidence:
            max_confidence = det["confidence"]
            max_detection = det
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bounding_box"])
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add similar images info
    y_offset = 30
    for i, sim_img in enumerate(similar_images):
        text = f"Similar {i+1}: {sim_img['image_id']} ({sim_img['distance']:.2f})"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += 25
    
    return frame, max_detection, similar_images

def detection_video(video_path):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Load FAISS index
    index, image_ids = load_index()
    if index is None or image_ids is None:
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame interval for 2 frames per second
    frame_interval = fps // 4
    
    # Create output video writer with video name
    output_path = f"output_{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    results = []
    saved_frames = []
    
    # Dictionary to track clothing types and their occurrences
    clothing_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process only every nth frame (2 frames per second)
        if frame_count % frame_interval != 0:
            continue
            
        processed_frame, max_detection, similar_images = process_video_frame(frame, index, image_ids)
        
        # Only save frames where clothing is detected
        if max_detection is not None and max_detection["confidence"] > 0.5:  # Confidence threshold
            # Track clothing types
            clothing_type = max_detection["class"]
            if clothing_type not in clothing_counts:
                clothing_counts[clothing_type] = {
                    "count": 0,
                    "max_confidence": 0,
                    "similar_products": []
                }
            
            clothing_counts[clothing_type]["count"] += 1
            if max_detection["confidence"] > clothing_counts[clothing_type]["max_confidence"]:
                clothing_counts[clothing_type]["max_confidence"] = max_detection["confidence"]
                clothing_counts[clothing_type]["similar_products"] = similar_images
            
            # Store results
            frame_result = {
                "frame_number": frame_count,
                "timestamp": datetime.now().isoformat(),
                "max_detection": max_detection,
                "similar_images": similar_images
            }
            results.append(frame_result)
            
            # Save the frame with video name
            frame_path = f"frames/{video_name}_frame_{frame_count}.jpg"
            os.makedirs("frames", exist_ok=True)
            cv2.imwrite(frame_path, processed_frame)
            saved_frames.append(frame_path)
            
            # Write the processed frame to output video
            out.write(processed_frame)
        
        # Display the processed frame
        cv2.imshow('Detection', processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Get top 2 most frequent clothing items
    top_clothing = sorted(
        clothing_counts.items(),
        key=lambda x: (x[1]["count"], x[1]["max_confidence"]),
        reverse=True
    )[:2]
    
    # Get vibes from vibe_detection
    vibes = vibe_detection(f"captions/{video_name}.txt")
    if vibes:
        try:
            # Convert string representation of list to actual list
            vibes = eval(vibes)
        except:
            vibes = ["Unknown"]
    else:
        vibes = ["Unknown"]
    
    # Format products for JSON
    products = []
    for clothing_type, data in top_clothing:
        print(f"All similar products {data['similar_products']}")
        if data["similar_products"]:
            best_match = data["similar_products"][0]  # Get the best matching product
            products.append({
                "type": clothing_type,
                "matched_product_id": best_match["image_id"],
                "match_type": "exact" if best_match["distance"] < 0.65 else "similar",
                "confidence": float(data["max_confidence"])
            })
    
    # Create final JSON output
    output_json = {
        "video_id": video_name,
        "vibes": vibes,
        "products": products
    }
    
    # Save results to JSON with video name

   
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {len(results)}")
    print(f"Saved frames: {saved_frames}")
    
    return output_json

def detect_clothing_from_frame(frame):
    results = model.infer(frame)[0]
    detections = []

    # Roboflow returns predictions in a different format
    for prediction in results.predictions:
        if prediction.class_name =='Female':
            continue
        detections.append({
            "class": prediction.class_name,
            "confidence": prediction.confidence,
            "bounding_box": [
                prediction.x - prediction.width/2,  # x1
                prediction.y - prediction.height/2, # y1
                prediction.x + prediction.width/2,  # x2
                prediction.y + prediction.height/2  # y2
            ]
        })
    if 'Female' in detections:
        detections.remove()
    print(f"Here is the detections {detections}")        
   
    return detections


def vibe_detection(caption):
    try:
        with open(caption, 'r', encoding='utf-8', errors='replace') as file:
            caption = file.read()
        client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

        model = "gemini-2.0-flash-lite"
        contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=caption),
            ],
        ),
    ]
        generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are a professional content stylist and aesthetics expert trained to classify social media content based on fashion and lifestyle trends.

Task:
Given an Instagram Reel caption, identify which aesthetic vibe it most closely matches from the following list:

1. Coquette – soft, flirty, feminine, romantic
2. Clean Girl – minimalist, neutral, glowy, polished
3. Cottagecore – nature-inspired, vintage, cozy, rural
4. Streetcore – urban, bold, oversized, edgy
5. Y2K – nostalgic 2000s, shiny, techy, playful
6. Boho – earthy, free-spirited, eclectic, festival
7. Party Glam – bold, sparkly, luxurious, nightlife

Instructions:
- Choose only **one** vibe that fits best.
- Do **not explain** your choice unless specifically asked.
- Return output in the format:  
`Vibe: <Aesthetic Vibe>`

Output:-
- Return only vibe at least 2 , maximum 3 in the form of list
example:-
['Streetcore ','Cottagecore ']

"""),
        ],
    )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        print(f"Here is the vibe bro {response.text}")
        return response.text

    except FileNotFoundError:
        print(f"Error: File {caption} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

# Remove test code and add proper function
video_caption = "videos/reel_3_caption.txt"
video_path = "videos/reel_3.mp4"


result = detection_video(video_path=video_path)
vibe =  vibe_detection(video_caption)
result['vibes'] = vibe
print(f"Here is the vibes := {vibe}")
video_name = os.path.splitext(os.path.basename(video_path))[0]
json_path = f"output/detection_results_{video_name}.json"
os.makedirs("output", exist_ok=True)
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"here is the results :- {result}")



# Data Flow 
# detection_video(video) -> process_video_frame(video) ->  detect_clothing_from_frame(frame) -> return to process_video_frame() -> search_similiar_images(frame,faiss,index) -> get_embedding(frame) -> return to search_similiar_images() -> search with similiar search -> return to precess_video_frame() -> return to detectin_video() -> save the frames and also output in json with vibe
#                                                                                       
# vibe_detection(video_caption) -> returnt the vibe