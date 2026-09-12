from ultralytics import YOLO

# Load the new best weights once training completes
model = YOLO("runs/train/exp/weights/best.pt")

# Print saved class names to double-check
print("Model classes:", model.names)


-> check weights of the trained model


from ultralytics import YOLO

# Load your completed weights
model = YOLO("runs/train/exp/weights/best.pt")

# Rename class 0 from 'item' to 'dick'
model.model.names[0] = "dick"

# Save as a fixed checkpoint
model.save("models/best_dick.pt")

print("Updated model saved! Verified class names:", model.names)


=> script to change the class if it's wrong