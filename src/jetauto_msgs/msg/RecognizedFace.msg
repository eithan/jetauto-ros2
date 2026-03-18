# A recognized (or unknown) face from the face recognition pipeline.
string name                         # Person name ("unknown" if no match)
float32 confidence                  # Recognition confidence [0.0, 1.0]
float32 x1                          # Face bounding box top-left x (pixels)
float32 y1                          # Face bounding box top-left y (pixels)
float32 x2                          # Face bounding box bottom-right x (pixels)
float32 y2                          # Face bounding box bottom-right y (pixels)
float32 person_x1                   # Original person bounding box (from YOLO)
float32 person_y1
float32 person_x2
float32 person_y2
