from flask import Flask, request, jsonify 
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import io

app = Flask(__name__)
CORS(app)

def compare_images(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(img1_gray, img2_gray, full=True)
    diff = (diff * 255).astype("uint8")

    return score, diff

def explain_differences(diff_img):
    # Ensure diff image is grayscale
    gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    explanations = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Ignore tiny noise
            explanations.append(f"Change detected at (x:{x}, y:{y}) with size {w}x{h}")
    
    if not explanations:
        explanations.append("No major visual shifts detected.")
    
    return explanations

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'designA' not in request.files or 'designB' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400

    img_a = Image.open(request.files['designA']).convert('RGB')
    img_b = Image.open(request.files['designB']).convert('RGB')

    img_a_np = np.array(img_a)
    img_b_np = np.array(img_b)

    # Resize to match dimensions
    if img_a_np.shape != img_b_np.shape:
        img_b_np = cv2.resize(img_b_np, (img_a_np.shape[1], img_a_np.shape[0]))

    score, diff = compare_images(img_a_np, img_b_np)
    explanations = explain_differences(diff)


    # Generate basic defect report
    report = {
        "visual_consistency_score": round(score, 4),
        "visual_similarity_score": round(score * 100, 2),  # Converts to percentage like 87.56%
        "Functionality Testing": "All buttons and forms respond as expected (simulated).",
        "Visual Consistency": "High deviation detected" if score < 0.95 else "Visually consistent.",
        "Usability Insights": "Navigation is clear. Content hierarchy well maintained (simulated).",
        "Accessibility Checks": "Supports text resizing and screen reader basics (simulated).",
        "Cross-Browser & Device Testing": "Compatible across Chrome, Firefox, Safari, Edge (simulated).",
        "AI Explanation": explanations[:5]

    }
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True) 