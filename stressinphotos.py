import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

def analyze_emotion(img):
    """
    Analyze emotions in an image using DeepFace.

    Args:
        img: Image array in BGR format

    Returns:
        tuple: (dominant_emotion, emotions_dict) or (None, None) if analysis fails
    """
    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        return analysis['dominant_emotion'], analysis['emotion']
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return None, None

def analyze_stress(emotions_dict):
    """
    Analyze stress levels based on detected emotions.

    Args:
        emotions_dict: Dictionary of emotions and their scores

    Returns:
        dict: Analysis results including stress level and indicators
    """
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    top_3_emotions = sorted_emotions[:3]

    emotion_names = [e[0] for e in top_3_emotions]
    stress_level = 0
    stress_indicators = []

    if 'angry' in emotion_names[:2] and 'fear' in emotion_names[:2]:
        if emotion_names[2] in ['neutral', 'disgust']:
            stress_level = 3
            stress_indicators.append("Primary stress pattern detected")

    elif ('angry' in emotion_names[:2] or 'fear' in emotion_names[:2]):
        if 'disgust' in emotion_names[:2]:
            stress_level = 2
            stress_indicators.append("Secondary stress pattern detected")

    total_negative = sum(emotions_dict[emotion] for emotion in ['angry', 'fear', 'disgust'])
    if total_negative > 70:
        stress_level = max(stress_level, 2)
        stress_indicators.append("High negative emotion intensity")

    if emotions_dict['happy'] < 10:
        stress_level = max(stress_level, 1)
        stress_indicators.append("Low happiness detected")

    return {
        'stress_level': stress_level,
        'indicators': stress_indicators,
        'top_emotions': top_3_emotions
    }

def draw_results(img, face_detections, stress_analysis, emotions_dict):
    """
    Draw analysis results on the image.

    Args:
        img: Original image array
        face_detections: Detected faces information
        stress_analysis: Stress analysis results
        emotions_dict: Dictionary of detected emotions

    Returns:
        array: Image with visualization overlays
    """
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    if face_detections:
        for face in face_detections:
            facial_area = face['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']

            stress_colors = {
                0: (0, 255, 0),    # Green for no stress
                1: (255, 255, 0),  # Yellow for mild stress
                2: (255, 165, 0),  # Orange for moderate stress
                3: (255, 0, 0)     # Red for high stress
            }

            color = stress_colors[stress_analysis['stress_level']]
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)

            stress_levels = ['No', 'Mild', 'Moderate', 'High']
            stress_text = f"Stress Level: {stress_levels[stress_analysis['stress_level']]}"
            y_offset = 30
            cv2.putText(img_rgb, stress_text, (x, y - y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_rgb

def process_image(image_path, output_dir):
    """
    Process a single image and save results.

    Args:
        image_path: Path to the image file
        output_dir: Directory to save results

    Returns:
        dict: Analysis results or None if processing failed
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # Detect faces and analyze emotions
        face_detections = DeepFace.extract_faces(img, enforce_detection=False)
        dominant_emotion, emotions_dict = analyze_emotion(img)

        if dominant_emotion and emotions_dict:
            stress_analysis = analyze_stress(emotions_dict)
            result_image = draw_results(img, face_detections, stress_analysis, emotions_dict)

            # Save the processed image
            output_image_path = output_dir / f"processed_{image_path.name}"
            plt.imsave(str(output_image_path), result_image)

            # Prepare results dictionary
            results = {
                'image_name': image_path.name,
                'stress_level': stress_analysis['stress_level'],
                'indicators': '; '.join(stress_analysis['indicators']),
                'top_emotions': '; '.join([f"{e}: {s:.1f}%" for e, s in stress_analysis['top_emotions']])
            }

            return results

        return None

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_image_directory(input_dir, output_base_dir=None):
    """
    Process all images in a directory and generate analysis report.

    Args:
        input_dir: Directory containing input images
        output_base_dir: Base directory for outputs (optional)
    """
    input_dir = Path(input_dir)
    if output_base_dir is None:
        output_base_dir = input_dir / 'stress_analysis_results'
    else:
        output_base_dir = Path(output_base_dir)

    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image and collect results
    results = []
    total_images = len(image_files)

    print(f"Found {total_images} images to process")

    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{total_images}: {image_path.name}")
        result = process_image(image_path, output_dir)
        if result:
            results.append(result)

    # Generate and save summary report
    if results:
        df = pd.DataFrame(results)
        report_path = output_dir / 'analysis_report.csv'
        df.to_csv(report_path, index=False)

        print(f"\nAnalysis complete. Results saved to {output_dir}")
        print(f"Processed {len(results)} out of {total_images} images successfully")

        # Print stress level summary
        stress_summary = df['stress_level'].value_counts().sort_index()
        stress_levels = ['No Stress', 'Mild Stress', 'Moderate Stress', 'High Stress']
        print("\nStress Level Summary:")
        for level, count in stress_summary.items():
            print(f"{stress_levels[level]}: {count} images")
    else:
        print("No successful analyses to report")

if __name__ == "__main__":
    # Replace with your input directory path
    input_directory = "testt"
    process_image_directory(input_directory)
