import cv2
import numpy as np

# ============================
# PART 2: CONNECTED COMPONENTS LABELING (Using Union-Find)
# ============================
def connected_components_sparse(binary):
    # Get coordinates of non-zero pixels
    coords = np.argwhere(binary > 0)
    coords = coords[np.lexsort((coords[:,1], coords[:,0]))]  # Sort by rows then columns

    labels = {}      # Dictionary to store labels for each pixel
    parent = {}      # Union-Find parent dictionary
    current_label = 1

    # Find function with path compression
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Union function to join two sets
    def union(x, y):
        rootx = find(x)
        rooty = find(y)
        if rootx != rooty:
            parent[rooty] = rootx

    # First pass: assign labels and perform unions
    for (i, j) in coords:
        neighbor_labels = []
        if (i, j-1) in labels:  # Check left neighbor
            neighbor_labels.append(labels[(i, j-1)])
        if (i-1, j) in labels:  # Check top neighbor
            neighbor_labels.append(labels[(i-1, j)])
        if not neighbor_labels:
            # New component
            labels[(i, j)] = current_label
            parent[current_label] = current_label
            current_label += 1
        else:
            # Assign smallest label and union others
            min_label = min(neighbor_labels)
            labels[(i, j)] = min_label
            for label in neighbor_labels:
                union(min_label, label)

    # Second pass: flatten the union-find structure
    for key in labels.keys():
        labels[key] = find(labels[key])

    # Create label image
    label_image = np.zeros_like(binary, dtype=np.int32)
    for (i, j), lab in labels.items():
        label_image[i, j] = lab

    return label_image

# ============================
# PART 3: EXTRACT BOUNDING BOXES FOR EACH COMPONENT
# ============================
def get_bounding_boxes(labels):
    boxes = []
    for i in np.unique(labels):
        if i == 0:
            continue  # Skip background
        ys, xs = np.where(labels == i)
        if ys.size == 0 or xs.size == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        boxes.append((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))
    return boxes

# ============================
# COMPUTE AVERAGE BACKGROUND FRAME
# ============================
video_path = 'videos/VIDEO_3.mp4'
cap = cv2.VideoCapture(video_path)

sum_frame = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame.astype(np.float32)
    if sum_frame is None:
        sum_frame = np.zeros_like(frame)
    sum_frame += frame
    frame_count += 1

if frame_count == 0:
    print("Video contains no frames!")
    cap.release()
    exit()

# Compute average background
background = (sum_frame / frame_count).astype(np.uint8)
cap.release()

# ============================
# PROCESS VIDEO AND SAVE DETECTION OUTPUT
# ============================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_detected = cv2.VideoWriter('videos/detected_object_noCV_2.mp4', fourcc, fps, frame_size)
out_binary = cv2.VideoWriter('videos/binary_noCV_2.mp4', fourcc, fps, frame_size)

threshold_value = 50
min_area = 5000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Subtract background
    diff = np.abs(frame.astype(np.int16) - background.astype(np.int16)).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_blurred = cv2.GaussianBlur(diff_gray, (5, 5), 1.2)

    # Threshold to binary
    binary = np.where(diff_blurred > threshold_value, 255, 0).astype(np.uint8)
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for saving

    # Label connected components and extract bounding boxes
    labels = connected_components_sparse(binary)
    boxes = get_bounding_boxes(labels)

    # Draw bounding boxes on detected objects
    for (x, y, w, h) in boxes:
        if w * h < min_area:
            continue  # Skip small regions
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out_detected.write(frame)
    out_binary.write(binary_colored)

cap.release()
out_detected.release()
out_binary.release()

# ============================
# MERGE MULTIPLE VIDEOS INTO ONE COMPOSITE VIDEO
# ============================
video_path1 = 'videos/VIDEO_3.mp4'
video_path2 = 'videos/binary_noCV_2.mp4'
video_path3 = 'videos/detected_object_noCV_2.mp4'

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)
cap3 = cv2.VideoCapture(video_path3)

ret, frame_sample = cap1.read()
height, width, _ = frame_sample.shape
fps = cap1.get(cv2.CAP_PROP_FPS)
composite_frame_size = (2 * width, 2 * height)

out_writer = cv2.VideoWriter('videos/composite_video.mp4', fourcc, fps, composite_frame_size)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if not ret1:
        break

    # Resize frames or fill with blank if missing
    frame1 = cv2.resize(frame1, (width, height)) if ret1 else np.zeros((height, width, 3), np.uint8)
    frame2 = cv2.resize(frame2, (width, height)) if ret2 else np.zeros((height, width, 3), np.uint8)
    frame3 = cv2.resize(frame3, (width, height)) if ret3 else np.zeros((height, width, 3), np.uint8)

    blank = np.zeros((height, width, 3), dtype=np.uint8)

    # Combine frames into a 2x2 grid
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, blank))
    composite = np.vstack((top_row, bottom_row))

    out_writer.write(composite)
    cv2.imshow("Composite Video", composite)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cap1.release()
cap2.release()
cap3.release()
out_writer.release()
cv2.destroyAllWindows()
