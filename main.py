import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np
from norfair import Detection, Tracker
from tqdm import tqdm
from ultralytics import YOLO


def setup_logging(args=None):  # Modified to accept args
    """Configures the logging module."""
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file_path = None

    if args and args.log_to_file and args.output_dir:
        # Ensure output_dir is created before attempting to log there.
        # This is usually handled in main(), but good to be safe or if setup_logging is called elsewhere.
        os.makedirs(args.output_dir, exist_ok=True)
        log_file_path = os.path.join(args.output_dir, "processing.log")
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    if log_file_path:
        logging.info(f"Logging to file: {log_file_path}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect, track objects in videos and save slices."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the directory containing input videos.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the directory where sliced videos will be saved.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        required=True,
        help="List of object class names to track (e.g., person bicycle).",
    )
    parser.add_argument(
        "--model-name",
        default="yolo11x.pt",
        help="YOLO model name/path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.30,
        help="YOLO confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="YOLO IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.4,
        help="Norfair distance threshold (default: 0.7).",
    )
    parser.add_argument(
        "--hit-counter-max",
        type=int,
        default=30,
        help="Norfair hit counter max (default: 15).",
    )
    parser.add_argument(
        "--initialization-delay",
        type=int,
        default=10,
        help="Norfair initialization delay (default: 10).",
    )
    parser.add_argument(
        "--min-track-seconds",
        type=float,
        default=1.0,
        help="Minimum track duration in seconds to save a slice (default: 1.0).",
    )
    parser.add_argument(
        "--draw-bounding-boxes",
        action="store_true",
        default=False,
        help="Draw bounding boxes on sliced videos (default: False).",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        default=False,
        help="Enable logging to a file (processing.log) in the output directory.",
    )
    return parser.parse_args()


# Convert seconds to Thhmmss format
def format_time_iso(total_seconds):
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    seconds = int(total_seconds % 60)
    return f"T{hours:02}{minutes:02}{seconds:02}"


def yolo_detections_to_norfair(yolo_results, target_classes_set, model_names):
    """Converts YOLO detection results to Norfair Detection objects."""
    norfair_detections = []
    try:
        boxes = yolo_results.boxes.cpu().numpy()  # xyxy format
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model_names[class_id].lower()
            if class_name in target_classes_set:
                score = box.conf[0]
                coords = box.xyxy[0]
                detection = Detection(
                    points=np.array([[coords[0], coords[1]], [coords[2], coords[3]]]),
                    scores=np.array([score, score]),
                    label=class_name,  # Store class name in label
                    data={"class_name": class_name, "score": score},
                )
                norfair_detections.append(detection)
    except Exception as e:
        logging.error(f"Error converting YOLO detections: {e}")
    return norfair_detections


def process_video(video_path, args, model, target_classes_set):
    """Processes a single video file for tracking and slicing."""
    logging.info(f"Starting processing for: {video_path}")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            logging.error(f"Invalid video properties for {video_path}. Skipping.")
            cap.release()
            return

        # Initialize Norfair Tracker
        # Using IoU distance for bounding boxes
        tracker = Tracker(
            distance_function="iou",
            distance_threshold=args.distance_threshold,
            hit_counter_max=args.hit_counter_max,
            initialization_delay=args.initialization_delay,
        )

        # Store track info including bounding boxes per frame
        # track_id -> {class_name, start_frame, last_seen_frame, boxes: {frame_idx: box_coords}}
        track_data_store = {}
        frame_idx = 0

        # Frame processing loop
        with tqdm(
            total=total_frames, desc=f"Processing {os.path.basename(video_path)}"
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Perform detection
                try:
                    results = model(
                        frame,
                        conf=args.conf_threshold,
                        iou=args.iou_threshold,
                        verbose=False,
                    )[0]
                except Exception as e:
                    logging.error(f"Error during YOLO detection: {e}")
                    continue  # Skip frame on detection error

                # Convert detections
                detections = yolo_detections_to_norfair(
                    results, target_classes_set, model.names
                )

                # Update tracker
                try:
                    current_tracked_objects = tracker.update(detections=detections)
                except Exception as e:
                    logging.error(f"Error during Norfair update: {e}")
                    continue  # Skip frame on tracking error

                # Aggregate Track Lifespan Info and store boxes
                for obj in current_tracked_objects:
                    track_id = obj.id
                    class_name = (
                        obj.label
                        if obj.label
                        else obj.last_detection.data.get("class_name", "unknown")
                    )
                    box_coords = obj.estimate  # Get the bounding box estimate

                    if track_id not in track_data_store:
                        track_data_store[track_id] = {
                            "class_name": class_name,
                            "start_frame": frame_idx,
                            "last_seen_frame": frame_idx,
                            "boxes": {frame_idx: box_coords},  # Store first box
                        }
                    else:
                        track_data_store[track_id]["last_seen_frame"] = frame_idx
                        track_data_store[track_id]["boxes"][frame_idx] = (
                            box_coords  # Store subsequent boxes
                        )

                frame_idx += 1
                pbar.update(1)

        cap.release()
        logging.info(f"Finished frame processing for {video_path}")

        # Log detected objects summary
        if track_data_store:
            detected_objects_summary = [
                f"{data['class_name']}_{track_id}"
                for track_id, data in track_data_store.items()
            ]
            logging.info(
                f"Detected objects in {os.path.basename(video_path)}: {', '.join(detected_objects_summary)}"
            )
        else:
            logging.info(
                f"No target objects detected in {os.path.basename(video_path)}"
            )

        # Video Slicing Loop
        logging.info(f"Starting slicing phase for {video_path}")
        if not track_data_store:
            logging.info(f"No tracks found or persisted in {video_path}")
            return

        slice_count = 0
        for track_id, data in tqdm(
            track_data_store.items(), desc=f"Slicing {os.path.basename(video_path)}"
        ):
            duration_frames = data["last_seen_frame"] - data["start_frame"] + 1
            duration_seconds = duration_frames / fps

            # Filter by Duration
            if duration_seconds < args.min_track_seconds:
                continue

            start_sec = data["start_frame"] / fps
            end_sec = data["last_seen_frame"] / fps
            object_id = f"{data['class_name']}_{track_id}"
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            # Basic sanitization for filename
            safe_object_id = "".join(
                c for c in object_id if c.isalnum() or c in ("_", "-")
            ).rstrip()

            start_time_iso = format_time_iso(start_sec)
            end_time_iso = format_time_iso(end_sec)

            slice_filename = (
                f"{base_name}_{safe_object_id}_{start_time_iso}_{end_time_iso}.mp4"
            )
            output_path = os.path.join(args.output_dir, slice_filename)

            # Create Slice with Overlays
            try:
                cap_slice = cv2.VideoCapture(video_path)
                if not cap_slice.isOpened():
                    logging.error(f"Failed to reopen video for slicing: {video_path}")
                    continue

                # Check if start frame is valid
                if not cap_slice.set(cv2.CAP_PROP_POS_FRAMES, data["start_frame"]):
                    logging.error(
                        f"Failed to set start frame {data['start_frame']} for slicing {video_path}"
                    )
                    cap_slice.release()
                    continue

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # fourcc = cv2.VideoWriter_fourcc(*"av01")
                out_slice = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out_slice.isOpened():
                    logging.error(f"Failed to open VideoWriter for: {output_path}")
                    cap_slice.release()
                    continue

                # Write frames for the slice with overlays
                for current_frame_idx in range(
                    data["start_frame"], data["last_seen_frame"] + 1
                ):
                    ret_slice, frame_slice = cap_slice.read()
                    if not ret_slice:
                        logging.warning(
                            f"Could not read frame {current_frame_idx} while slicing {slice_filename}. Slice might be shorter."
                        )
                        break

                    # Draw bounding box if it exists for this frame and track and if drawing is enabled
                    if args.draw_bounding_boxes and current_frame_idx in data["boxes"]:
                        box = data["boxes"][current_frame_idx]
                        # Ensure box has 2 points (top-left, bottom-right)
                        if box is not None and len(box) == 2:
                            pt1 = tuple(map(int, box[0]))
                            pt2 = tuple(map(int, box[1]))
                            cv2.rectangle(
                                frame_slice, pt1, pt2, (0, 255, 0), 2
                            )  # Green box
                            label = f"{data['class_name']} ID:{track_id}"
                            # Put text above the box
                            (w, h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            text_pt = (
                                pt1[0],
                                pt1[1] - 5 if pt1[1] > 20 else pt1[1] + h + 5,
                            )
                            cv2.putText(
                                frame_slice,
                                label,
                                text_pt,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                    out_slice.write(frame_slice)

                out_slice.release()
                cap_slice.release()
                logging.debug(f"Successfully created slice: {output_path}")
                slice_count += 1

            except Exception as e:
                logging.error(f"Error creating slice {output_path}: {e}")
                # Ensure resources are released even if error occurs mid-slicing
                if "cap_slice" in locals() and cap_slice.isOpened():
                    cap_slice.release()
                if "out_slice" in locals() and out_slice.isOpened():
                    out_slice.release()

        logging.info(
            f"Finished slicing for {video_path}. Created {slice_count} slices."
        )

    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")
        # Ensure capture is released if an error occurred before release
        if "cap" in locals() and cap.isOpened():
            cap.release()


def main():
    """Main function to orchestrate video processing."""
    args = parse_arguments()  # Parse arguments first
    setup_logging(args)  # Setup logging with args

    # Convert target classes to lowercase set for efficient lookup
    target_classes_set = {cls.lower() for cls in args.target_classes}
    logging.info(f"Target classes: {target_classes_set}")

    # Validate input/output directories
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory: {args.output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)

    # Find video files (adjust patterns as needed)
    video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

    if not video_files:
        logging.warning(f"No video files found in {args.input_dir}")
        sys.exit(0)

    logging.info(f"Found {len(video_files)} video files to process.")

    # Load YOLO model once
    try:
        model = YOLO(args.model_name)
        logging.info(f"Loaded YOLO model: {args.model_name}")
        # Verify target classes against model classes
        model_classes_lower = {name.lower() for name in model.names.values()}
        invalid_classes = target_classes_set - model_classes_lower
        if invalid_classes:
            logging.warning(
                f"Target classes {invalid_classes} not found in model {args.model_name}. They will be ignored."
            )
            target_classes_set -= invalid_classes  # Remove invalid classes
        if not target_classes_set:
            logging.error(
                "No valid target classes remain after checking against the model. Exiting."
            )
            sys.exit(1)

    except Exception as e:
        logging.error(f"Failed to load YOLO model {args.model_name}: {e}")
        sys.exit(1)

    # Process each video
    for video_path in video_files:
        process_video(video_path, args, model, target_classes_set)

    logging.info("All videos processed.")


if __name__ == "__main__":
    main()
