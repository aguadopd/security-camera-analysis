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
import ffmpeg  # Add this import


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
            logging.error(
                f"Invalid video properties for {video_path} (fps: {fps}, frames: {total_frames}). Skipping."
            )
            cap.release()
            return

        # Initialize Norfair Tracker
        tracker = Tracker(
            distance_function="iou",
            distance_threshold=args.distance_threshold,
            hit_counter_max=args.hit_counter_max,
            initialization_delay=args.initialization_delay,
        )

        track_data_store = {}
        frame_idx = 0

        with tqdm(
            total=total_frames, desc=f"Processing {os.path.basename(video_path)}"
        ) as pbar:
            while True:
                ret, frame = cap.read()
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

                    if track_id not in track_data_store:
                        track_data_store[track_id] = {
                            "class_name": class_name,
                            "start_frame": frame_idx,
                            "last_seen_frame": frame_idx,
                            "boxes": {frame_idx: box_coords},
                        }
                    else:
                        track_data_store[track_id]["last_seen_frame"] = frame_idx
                        track_data_store[track_id]["boxes"][frame_idx] = box_coords

                frame_idx += 1
                pbar.update(1)

        cap.release()
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
            duration_seconds = duration_frames / fps

            if duration_seconds < args.min_track_seconds:
                continue

            start_sec = data["start_frame"] / fps
            slice_duration_sec = (data["last_seen_frame"] + 1) / fps - start_sec

            object_id = f"{data['class_name']}_{track_id}"
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            safe_object_id = "".join(
                c for c in object_id if c.isalnum() or c in ("_", "-")
            ).rstrip()

            start_time_iso = format_time_iso(start_sec)
            end_sec_for_filename = (data["last_seen_frame"]) / fps
            end_time_iso = format_time_iso(end_sec_for_filename)

            slice_filename = (
                f"{base_name}_{safe_object_id}_{start_time_iso}_{end_time_iso}.mp4"
            )
            output_path = os.path.join(args.output_dir, slice_filename)

            try:
                if not args.draw_bounding_boxes:
                    (
                        ffmpeg.input(video_path, ss=start_sec, t=slice_duration_sec)
                        .output(
                            output_path,
                            c="copy",
                            video_bitrate=None,
                            audio_bitrate=None,
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True, quiet=True)
                    )
                    logging.debug(
                        f"Successfully created slice (direct copy): {output_path}"
                    )
                    slice_count += 1
                else:
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
                        slice_fps = fps

                    process = (
                        ffmpeg.input(
                            "pipe:",
                            format="rawvideo",
                            pix_fmt="bgr24",
                            s=f"{width}x{height}",
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

        logging.info(
            f"Finished slicing for {video_path}. Created {slice_count} slices."
        )

    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")
        if "cap" in locals() and cap.isOpened():
            cap.release()


def main():
    """Main function to orchestrate video processing."""
    args = parse_arguments()  # Parse arguments first
    setup_logging(args)  # Setup logging with args
    logging.info(f"Script arguments: {vars(args)}")

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
