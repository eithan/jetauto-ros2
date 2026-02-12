# A single detected object from the vision pipeline.
string label                    # Class name (e.g. "cup", "person")
float32 confidence              # Detection confidence [0.0, 1.0]
float32 x1                      # Bounding box top-left x (pixels)
float32 y1                      # Bounding box top-left y (pixels)
float32 x2                      # Bounding box bottom-right x (pixels)
float32 y2                      # Bounding box bottom-right y (pixels)
