from PIL import Image, ImageDraw, ImageFilter
import os
import math

out_dir = r"C:\Users\deonb\OneDrive\Desktop\ESE\frontend\public\sequence"
os.makedirs(out_dir, exist_ok=True)

width, height = 1920, 1080
for i in range(100):
   img = Image.new('RGB', (width, height), color=(15, 23, 42))
   d = ImageDraw.Draw(img)
   
   x = width/2 + math.sin(i / 10.0) * 300
   y = height/2 + math.cos(i / 10.0) * 300
   
   d.ellipse([x-250, y-250, x+250, y+250], fill=(16, 185, 129))
   
   x2 = width/2 + math.cos(i / 15.0) * 400
   y2 = height/2 + math.sin(i / 15.0) * 400
   d.ellipse([x2-300, y2-300, x2+300, y2+300], fill=(6, 182, 212))
   
   # Apply blur to make it look premium/abstract
   img = img.filter(ImageFilter.GaussianBlur(100))
   
   img.save(os.path.join(out_dir, f"{i:04d}.webp"))
print("Done generating 100 sequence frames.")
