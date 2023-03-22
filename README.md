[简体中文](README_ch.md) | English

# Easier, faster, and more accurate to deploy in trajectory tracking with Re-identification using openvino

<img src=".\common\show_pic\image.png" alt="image" style="zoom:200%;" />

# How To Use:  **Quick start**

## 1、Preparation

### 1.0、Installing requirements

This project is based on openvino, so in order to run this project, make sure you have prepared the  openvino environment. Download Link: [here](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html)

You can also take a reference from requirements.txt .

```python
pip install -r requirements.txt
```

### 1.1、Add your video information in configs/configs.json

e.g. Supposed you have a supermarket surveillance video 0.mp4, if the time when the video starts shooting is 8:00, add "supermarket" in place, where "0" is the name of the video and 28800 is the number of seconds when the video starts shooting, that is, 8x60x60 = 28800.

~~~json
{
    "place":{
        "supermarket":{
            "0":28800
        }
    }
}
~~~

### 1.2、Put your video into ./video_input 

### 1.3、Put your target image into ./target_input

We provide our models and samples. you can also download our sample image and video which had been zipped. Baidu Drive:[here](https://pan.baidu.com/s/1nkM6b2jDnjJHJTUZV-qGDw?pwd=oj8k) (password:oj8k)

You can just copy all files into main project path without any other process after downloading.

## 2、Run 

Input the codes as followed in terminal console:

~~~python
python ReID_main.py -v ./video_input -t ./target_input --data_output ./data_output --top_k 50 --skip 10
~~~

A few arguments should be noticed:

top-k：Top k indexes in gallery features. Both too large and too small will affect the results.

skip：Skip n frames when analyzing. If you don't want to skip any frames, input 1.

## 3、Results

After running, you can check the results in ./data_output/message .

If you have already had the detections.npy and rm.npy results, you can set the argument --is_only_txt True to rerun it without deploying.

~~~python
python ReID_main.py -v ./video_input -t ./target_input --is_only_txt True --npy_path <your-rm-npy-path> --top_k 50 --skip 10
~~~

