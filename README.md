# Security Camera Analysis

A short Python script that detects and tracks objects from video —using
[`norfair`](https://tryolabs.github.io/norfair/2.2/) and some detector (here a
[`yolo11` by
Ultralytics](https://docs.ultralytics.com/models/yolo11/#overview), [trained on
the COCO dataset](https://cocodataset.org/#home))—, and then outputs video
slices with the detected objects (optionally drawing bounding boxes).


## Installation

Should be installable as a Python package. I've only tested with GPU, used by
`ultralytics`-provided YOLO detector. Used version requires `torch`, and you
may need to manually install CUDA stuff.

Project was developed with [`uv`](https://docs.astral.sh/uv/) as a Python
package manager. You can use the package with

```
# cd into project dir
uv run security-camera-analysis
```

and `uv` will take care of creating a virtual environment (at the project dir),
install dependencies and run the script.

No need to use `uv`, though. Any modern Python packaging tool should work, as
far as I understand.


## Usage


```
> security-camera-analysis --help

Detect, track objects in videos and save slices.

options:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        Path to the directory containing input videos.
  --output-dir OUTPUT_DIR
                        Path to the directory where sliced videos will be saved.
  --target-classes TARGET_CLASSES [TARGET_CLASSES ...]
                        List of object class names to track (e.g., person bicycle).
  --model-name MODEL_NAME
                        YOLO model name/path (default: yolo11l.pt).
  --conf-threshold CONF_THRESHOLD
                        YOLO confidence threshold (default: 0.25).
  --iou-threshold IOU_THRESHOLD
                        YOLO IoU threshold for NMS (default: 0.45).
  --distance-threshold DISTANCE_THRESHOLD
                        Norfair distance threshold (default: 0.7).
  --hit-counter-max HIT_COUNTER_MAX
                        Norfair hit counter max (default: 15).
  --initialization-delay INITIALIZATION_DELAY
                        Norfair initialization delay (default: 10).
  --min-track-seconds MIN_TRACK_SECONDS
                        Minimum track duration in seconds to save a slice (default: 1.0).
  --draw-bounding-boxes
                        Draw bounding boxes on sliced videos (default: False).
  --verbose             Enable verbose (DEBUG level) logging (default: False).
  --log-to-file         Enable logging to a file (processing.log) in the output directory.
```

Example:
```sh
uv run security-camera-analysis --input-dir ./test/input \
    --output-dir ./test/output \
    --target-classes cat dog person \
    --model yolo11s
```

See https://tryolabs.github.io/norfair/2.2/reference/tracker/ for some notes on
parameters.

If using the provided `yolo11` models (automatically downloaded by
`ultralytics`), those are trained using the COCO dataset. [Per the Ultralytics
docs](https://docs.ultralytics.com/datasets/detect/coco/#applications), the
detected classes are:

```yaml
# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

But you may use any other `ultralytics`-compatible YOLO model.


## Known issues

- Video input is done with `opencv`, which uses its own pre-compiled version of
  `ffmpeg`, which may fail to load some codecs.


## Development

.


## Meta

A bike was stolen to some neighbor. Security cameras at home don't have any
detection system enabled, and scrubbing 16h of video was not fun. Also part of
an ongoing experiment with AI agents. Done in a day with the help of AI agents
(Gemini 2.5 Pro and Claude 3.7 Sonnet) via GitHub Copilot.

None of the detected bicycles was the stolen one. A lot of false positives,
too, but still did review the output videos in ~20 min.

## Changelog

- 2025-05-08: first version.


## To do

- [X] Add some usage notes.
- [ ] Add some example images.
- [X] Link to `norfair` parameters.
- [ ] Cite alternatives, and explain why the ones I found weren't good for my
  use case.
- [ ] Propose enhancements: roi, list of detector classes, GUI?, replace opencv
  with torchvision/torchcodec, ...
- [ ] Add some type-hinting.
- [ ] Add some tests.
- [ ] Optimise after reading
  https://docs.ultralytics.com/quickstart/#modifying-settings 
- [ ] Check how ultralytics works with other pre-trained models. List
  interesting models or point to a model zoo.
  
