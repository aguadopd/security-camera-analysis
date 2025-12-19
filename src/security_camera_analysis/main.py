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
import ffmpeg
import time


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
        level=logging.DEBUG if args and args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    if log_file_path:
        logging.info(f"Logging to file: {log_file_path}")

    if args and args.verbose:
        logging.debug("Verbose logging enabled")


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
        default="yolo11l.pt",
        help="YOLO model name/path (default: yolo11l.pt).",
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
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG level) logging (default: False).",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        default=False,
        help="Enable logging to a file (processing.log) in the output directory.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("x1", "y1", "x2", "y2"),
        help="Region of interest as four integers: x1 y1 x2 y2 (pixel coordinates).",
        default=None,
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
        # Open original video (used later for slicing). We'll optionally create
        # a temporary cropped video for detection to speed up processing.
        orig_cap = cv2.VideoCapture(video_path)
        if not orig_cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return

        # Original video properties
        orig_fps = orig_cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_total_frames = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if orig_fps <= 0 or orig_total_frames <= 0:
            logging.error(
                f"Invalid video properties for {video_path} (fps: {orig_fps}, frames: {orig_total_frames}). Skipping."
            )
            orig_cap.release()
            return

        # Determine detection source: either the original file (no ROI) or a
        # temporary cropped file (ROI provided).
        temp_cropped_path = None
        detect_cap = None
        detect_fps = orig_fps
        if args.roi_tuple:
            x1, y1, x2, y2 = args.roi_tuple
            # Clamp ROI to video bounds
            x1 = max(0, min(x1, orig_width - 1))
            y1 = max(0, min(y1, orig_height - 1))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            roi_w = x2 - x1
            roi_h = y2 - y1
            if roi_w <= 0 or roi_h <= 0:
                logging.error(
                    f"ROI after clamping has non-positive size: {(x1, y1, x2, y2)}"
                )
                orig_cap.release()
                return

            if roi_w == orig_width and roi_h == orig_height and x1 == 0 and y1 == 0:
                logging.info(
                    "ROI equals full frame; skipping cropping and using original video for detection."
                )
                detect_cap = orig_cap
                detect_fps = orig_fps
            else:
                # Create unique temp path inside output dir
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                timestamp = int(time.time())
                pid = os.getpid()
                temp_cropped_path = os.path.join(
                    args.output_dir, f".tmp_{base_name}_roi_{timestamp}_{pid}.mp4"
                )
                logging.info(
                    f"Creating temporary cropped video: {temp_cropped_path} (crop {roi_w}x{roi_h} at {x1},{y1})"
                )
                try:
                    (
                        ffmpeg.input(video_path)
                        .filter("crop", roi_w, roi_h, x1, y1)
                        .output(
                            temp_cropped_path,
                            vcodec="libx264",
                            preset="ultrafast",
                            crf=23,
                            r=orig_fps,
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logging.error(
                        f"ffmpeg crop failed for {video_path}: {e.stderr.decode('utf8') if e.stderr else str(e)}"
                    )
                    orig_cap.release()
                    return

                detect_cap = cv2.VideoCapture(temp_cropped_path)
                if not detect_cap.isOpened():
                    logging.error(
                        f"Could not open temporary cropped video: {temp_cropped_path}"
                    )
                    orig_cap.release()
                    # Attempt cleanup
                    try:
                        if os.path.exists(temp_cropped_path):
                            os.remove(temp_cropped_path)
                    except Exception:
                        pass
                    return

                detect_fps = detect_cap.get(cv2.CAP_PROP_FPS)
                logging.debug(
                    f"Reopened cropped video {temp_cropped_path} with fps={detect_fps}"
                )
        else:
            detect_cap = orig_cap
            detect_fps = orig_fps

        # Initialize Norfair Tracker
        tracker = Tracker(
            distance_function="iou",
            distance_threshold=args.distance_threshold,
            hit_counter_max=args.hit_counter_max,
            initialization_delay=args.initialization_delay,
        )

        track_data_store = {}
        frame_idx = 0

        # Use detect_cap for detection loop (could be original or temp cropped)
        detect_total_frames = int(detect_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(
            total=detect_total_frames, desc=f"Processing {os.path.basename(video_path)}"
        ) as pbar:
            while True:
                ret, frame = detect_cap.read()
                if not ret:
                    break

                try:
                    results = model(
                        frame,
                        conf=args.conf_threshold,
                        iou=args.iou_threshold,
                        verbose=False,
                    )[0]
                except Exception as e:
                    logging.error(f"Error during YOLO detection: {e}")
                    continue

                detections = yolo_detections_to_norfair(
                    results, target_classes_set, model.names
                )

                try:
                    current_tracked_objects = tracker.update(detections=detections)
                except Exception as e:
                    logging.error(f"Error during Norfair update: {e}")
                    continue

                for obj in current_tracked_objects:
                    track_id = obj.id
                    class_name = (
                        obj.label
                        if obj.label
                        else obj.last_detection.data.get("class_name", "unknown")
                    )
                    box_coords = obj.estimate

                    # Map detection frame index (in detect_cap space) to original frame index
                    # Use timestamps to be robust to fps differences
                    time_sec = (
                        frame_idx / detect_fps
                        if detect_fps and detect_fps > 0
                        else frame_idx / orig_fps
                    )
                    orig_frame_idx = int(round(time_sec * orig_fps))

                    # Map box coordinates from cropped->original if ROI was used
                    if args.roi_tuple:
                        x_off, y_off = args.roi_tuple[0], args.roi_tuple[1]
                        try:
                            mapped_box = np.array(box_coords) + np.array([x_off, y_off])
                        except Exception:
                            mapped_box = box_coords
                    else:
                        mapped_box = box_coords

                    if track_id not in track_data_store:
                        track_data_store[track_id] = {
                            "class_name": class_name,
                            "start_frame": orig_frame_idx,
                            "last_seen_frame": orig_frame_idx,
                            "boxes": {orig_frame_idx: mapped_box},
                        }
                    else:
                        track_data_store[track_id]["last_seen_frame"] = orig_frame_idx
                        track_data_store[track_id]["boxes"][orig_frame_idx] = mapped_box

                frame_idx += 1
                pbar.update(1)

        # Release detection cap if it was a separate temp file
        try:
            if detect_cap is not None and detect_cap is not orig_cap:
                detect_cap.release()
        except Exception:
            pass

        # orig_cap remains open for slicing below
        logging.info(f"Finished frame processing for {video_path}")

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

        logging.info(f"Starting slicing phase for {video_path}")
        if not track_data_store:
            logging.info(f"No tracks found or persisted in {video_path}")
            return

        slice_count = 0
        for track_id, data in tqdm(
            track_data_store.items(), desc=f"Slicing {os.path.basename(video_path)}"
        ):
            duration_frames = data["last_seen_frame"] - data["start_frame"] + 1
            duration_seconds = duration_frames / orig_fps

            if duration_seconds < args.min_track_seconds:
                continue

            start_sec = data["start_frame"] / orig_fps
            slice_duration_sec = (data["last_seen_frame"] + 1) / orig_fps - start_sec

            object_id = f"{data['class_name']}_{track_id}"
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            safe_object_id = "".join(
                c for c in object_id if c.isalnum() or c in ("_", "-")
            ).rstrip()

            start_time_iso = format_time_iso(start_sec)
            end_sec_for_filename = (data["last_seen_frame"]) / orig_fps
            end_time_iso = format_time_iso(end_sec_for_filename)

            slice_filename = (
                f"{base_name}_{safe_object_id}_{start_time_iso}_{end_time_iso}.mp4"
            )
            output_path = os.path.join(args.output_dir, slice_filename)

            try:
                if not args.draw_bounding_boxes:
                    # Slice from original video (preserve context)
                    (
                        ffmpeg.input(video_path, ss=start_sec, t=slice_duration_sec)
                        .output(output_path, c="copy")
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True, quiet=True)
                    )
                    logging.debug(
                        f"Successfully created slice (direct copy): {output_path}"
                    )
                    slice_count += 1
                else:
                    # For drawing, open original video and seek to original start frame
                    cap_slice = cv2.VideoCapture(video_path)
                    if not cap_slice.isOpened():
                        logging.error(
                            f"Failed to reopen video for slicing with bboxes: {video_path}"
                        )
                        continue

                    if not cap_slice.set(cv2.CAP_PROP_POS_FRAMES, data["start_frame"]):
                        logging.error(
                            f"Failed to set start frame {data['start_frame']} for slicing {slice_filename}"
                        )
                        cap_slice.release()
                        continue

                    slice_fps = cap_slice.get(cv2.CAP_PROP_FPS)
                    if slice_fps <= 0:
                        slice_fps = orig_fps

                    process = (
                        ffmpeg.input(
                            "pipe:",
                            format="rawvideo",
                            pix_fmt="bgr24",
                            s=f"{orig_width}x{orig_height}",
                            r=slice_fps,
                        )
                        .output(
                            output_path,
                            vcodec="libx264",
                            video_bitrate="5M",
                            preset="medium",
                        )
                        .overwrite_output()
                        .run_async(pipe_stdin=True, quiet=True)
                    )

                    frames_to_write = data["last_seen_frame"] - data["start_frame"] + 1
                    frames_written_successfully = 0
                    for current_frame_num in range(frames_to_write):
                        actual_frame_idx = data["start_frame"] + current_frame_num
                        ret_slice, frame_slice = cap_slice.read()
                        if not ret_slice:
                            logging.warning(
                                f"Could not read frame {actual_frame_idx} (attempt {current_frame_num + 1}/{frames_to_write}) while slicing with bboxes {slice_filename}. Slice might be shorter."
                            )
                            break

                        if actual_frame_idx in data["boxes"]:
                            box = data["boxes"][actual_frame_idx]
                            if box is not None and len(box) == 2:
                                pt1 = tuple(map(int, box[0]))
                                pt2 = tuple(map(int, box[1]))
                                cv2.rectangle(frame_slice, pt1, pt2, (0, 255, 0), 2)
                                label = f"{data['class_name']} ID:{track_id}"
                                (w_text, h_text), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                text_pt = (
                                    pt1[0],
                                    pt1[1] - 5 if pt1[1] > 20 else pt1[1] + h_text + 5,
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

                        try:
                            process.stdin.write(frame_slice.tobytes())
                            frames_written_successfully += 1
                        except BrokenPipeError:
                            logging.warning(
                                f"Broken pipe while writing frame {actual_frame_idx} to ffmpeg for {slice_filename}. FFmpeg might have exited. Slice might be incomplete."
                            )
                            break
                        except Exception as e_pipe:
                            logging.error(
                                f"Error writing frame {actual_frame_idx} to ffmpeg pipe for {slice_filename}: {e_pipe}"
                            )
                            break

                    process.stdin.close()
                    process.wait()
                    cap_slice.release()

                    if (
                        process.returncode == 0
                        and frames_written_successfully == frames_to_write
                    ):
                        logging.debug(
                            f"Successfully created slice with bboxes: {output_path}"
                        )
                    elif process.returncode == 0:
                        logging.warning(
                            f"Slice with bboxes {output_path} created, but frame count mismatch. Expected {frames_to_write}, wrote {frames_written_successfully}. FFmpeg exit code 0."
                        )
                    else:
                        logging.error(
                            f"Failed to create slice with bboxes: {output_path}. FFmpeg exit code: {process.returncode}. Wrote {frames_written_successfully}/{frames_to_write} frames."
                        )
                    slice_count += 1

            except ffmpeg.Error as e:
                error_message = e.stderr.decode("utf8") if e.stderr else str(e)
                logging.error(
                    f"ffmpeg error creating slice {output_path}: {error_message}"
                )
                if (
                    args.draw_bounding_boxes
                    and "cap_slice" in locals()
                    and cap_slice.isOpened()
                ):
                    cap_slice.release()
                if (
                    args.draw_bounding_boxes
                    and "process" in locals()
                    and process.poll() is None
                ):
                    process.kill()
            except Exception as e:
                logging.error(f"Generic error creating slice {output_path}: {e}")
                if (
                    args.draw_bounding_boxes
                    and "cap_slice" in locals()
                    and cap_slice.isOpened()
                ):
                    cap_slice.release()
                if (
                    args.draw_bounding_boxes
                    and "process" in locals()
                    and "process.stdin" in locals()
                    and not process.stdin.closed
                ):
                    process.stdin.close()
                if (
                    args.draw_bounding_boxes
                    and "process" in locals()
                    and process.poll() is None
                ):
                    process.kill()

            # Cleanup: release original capture and delete temporary cropped file if created
            try:
                if "orig_cap" in locals() and orig_cap.isOpened():
                    orig_cap.release()
            except Exception:
                pass

            if temp_cropped_path:
                try:
                    if os.path.exists(temp_cropped_path):
                        os.remove(temp_cropped_path)
                        logging.info(
                            f"Deleted temporary cropped video: {temp_cropped_path}"
                        )
                except Exception as e:
                    logging.warning(
                        f"Failed to delete temporary cropped file {temp_cropped_path}: {e}"
                    )

    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")
        try:
            if "orig_cap" in locals() and orig_cap.isOpened():
                orig_cap.release()
        except Exception:
            pass


def main():
    """Main function to orchestrate video processing."""
    args = parse_arguments()  # Parse arguments first
    setup_logging(args)  # Setup logging with args
    logging.info(f"Script arguments: {vars(args)}")

    # Parse and validate ROI argument (if provided). Argparse now provides four ints when given.
    if args.roi is not None:
        try:
            x1, y1, x2, y2 = args.roi
            if x2 <= x1 or y2 <= y1:
                logging.error(
                    "ROI coordinates invalid: x2 must be > x1 and y2 must be > y1"
                )
                sys.exit(1)
            args.roi_tuple = (x1, y1, x2, y2)
            logging.info(f"ROI provided: {args.roi_tuple}")
        except Exception as e:
            logging.error(f"Failed to parse --roi: {e}")
            sys.exit(1)
    else:
        args.roi_tuple = None

    target_classes_set = {cls.lower() for cls in args.target_classes}
    logging.info(f"Target classes: {target_classes_set}")

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory: {args.output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)

    video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

    if not video_files:
        logging.warning(f"No video files found in {args.input_dir}")
        sys.exit(0)

    logging.info(f"Found {len(video_files)} video files to process.")

    try:
        model = YOLO(args.model_name)
        logging.info(f"Loaded YOLO model: {args.model_name}")
        model_classes_lower = {name.lower() for name in model.names.values()}
        invalid_classes = target_classes_set - model_classes_lower
        if invalid_classes:
            logging.warning(
                f"Target classes {invalid_classes} not found in model {args.model_name}. They will be ignored."
            )
            target_classes_set -= invalid_classes
        if not target_classes_set:
            logging.error(
                "No valid target classes remain after checking against the model. Exiting."
            )
            sys.exit(1)

    except Exception as e:
        logging.error(f"Failed to load YOLO model {args.model_name}: {e}")
        sys.exit(1)

    for video_path in video_files:
        process_video(video_path, args, model, target_classes_set)

    logging.info("All videos processed.")


if __name__ == "__main__":
    main()
