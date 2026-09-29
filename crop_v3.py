import cv2
import numpy as np
import glob
import os

def crop_profile(img_path):
    print(f"Processing {img_path}")
    img = cv2.imread(img_path)
    if img is None: return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We know the circle has a white border. Let's threshold near-white pixels.
    # #FFFFFF is 255,255,255.
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_circle = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000: # We expect a large circle
            # Get bounding box
            x,y,w,h = cv2.boundingRect(cnt)
            # A circle's bounding box is square
            aspect_ratio = float(w)/h
            if 0.9 < aspect_ratio < 1.1:
                # Calculate extent
                rect_area = w*h
                extent = float(area)/rect_area
                # A perfect circle has extent pi/4 ~ 0.785
                if 0.65 < extent < 0.9:
                    if area > best_score:
                        best_score = area
                        # We found the white circle border!
                        # The inner profile pic is slightly smaller.
                        r = w // 2
                        cx = x + r
                        cy = y + r
                        best_circle = (cx, cy, r)
                        
    if best_circle:
        cx, cy, r = best_circle
        # shrink radius by 5% to drop the white border entirely
        r_inner = int(r * 0.95)
        x1 = max(0, cx - r_inner)
        y1 = max(0, cy - r_inner)
        x2 = min(img.shape[1], cx + r_inner)
        y2 = min(img.shape[0], cy + r_inner)
        
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(img_path, cropped)
        print(f"-> Successfully cropped {img_path}: center ({cx},{cy}), radius {r_inner}")
    else:
        print(f"-> Failed to find circle in {img_path}")

files = glob.glob("web/*.png")
# Restore originals first
os.system("cp /home/pseudo/.gemini/antigravity-ide/brain/ee56ce8c-1d74-4f56-878f-7706fc5917aa/.user_uploaded/media_1790643126283.png web/khagesh.png")
os.system("cp /home/pseudo/.gemini/antigravity-ide/brain/ee56ce8c-1d74-4f56-878f-7706fc5917aa/.user_uploaded/media_1790643144576.png web/sudipto.png")
os.system("cp /home/pseudo/.gemini/antigravity-ide/brain/ee56ce8c-1d74-4f56-878f-7706fc5917aa/.user_uploaded/media_1790643156262.png web/ayush.png")
os.system("cp /home/pseudo/.gemini/antigravity-ide/brain/ee56ce8c-1d74-4f56-878f-7706fc5917aa/.user_uploaded/media_1790643168179.png web/shivanshu.png")
os.system("cp /home/pseudo/.gemini/antigravity-ide/brain/ee56ce8c-1d74-4f56-878f-7706fc5917aa/.user_uploaded/media_1790643206666.png web/satyam.png")

for f in files:
    if "oceanembed-logo" in f: continue
    crop_profile(f)
