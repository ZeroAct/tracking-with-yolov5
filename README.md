# Tracking with yolov5
This implementation is for who need to tracking multi-object only with detector. 
You can easily track mult-object with your well trained [yolov5](https://github.com/ultralytics/yolov5) model.
I used [SORT algorithm implementation](https://github.com/abewley/sort) to track each bounding boxes.
<br><br>
And I added my nobel(maybe) smoothing method. This method reduces the shaking of bounding boxes. You can easily deactivate smoothing method by specifying `--no_smoothing` option.<br><br>
I hope this repository can help someone :)

## Preparation
This implementation use yolov5 detection results. If you have another trained detector just follow this format. <br>
```
# file name
[video_name]_[frame_idx].txt

# center_x, center_y, width, height should be normalized with Video Width Height
class_id center_x center_y width height confidence 
...
```

or just run (if you have trained yolov5 model)

```
python detect.py  --weights [your model weight]
                  --source  [video path]
                  --save-txt --save-conf --nosave --iou-thres 1 --img 640  # keep this line same
```

## Run
I uploaded a [sample dataset](https://motchallenge.net/data/MOT17Det/) in `sample_data` directory. If you want to use this data, just run
```
python main.py --show
```
If you have your own dataset, run
```
python main.py  --video_path [video path]
                --det_dir    [your yolov5 det files directory]
                --show       [if you want to pre-visualize the results]
```

## Result
You can find your result video and text file under `results` directory.
