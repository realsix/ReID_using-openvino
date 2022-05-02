简体中文 | [English](README_en.md)

# How To Use

**第一次使用**

## 1、前期准备

### 0、安装环境

本项目基于openvino推理，因此你需要安装openvino。Windows与Linux的安装方式有所不同，详见官网。

另外，你可以参考我们的requirements.txt进行安装相关依赖。

```python
pip install -r requirements.txt
```

### 1、在configs文件夹中的configs.json中加入你想要加入的地点及初始时间

如 目前我有一个位于超市门口的监控视频0.mp4，视频开始拍摄的时间是8：00，则在place中加入supermarket ，其中，"0"是该视频名称，28800是开始拍摄时间的秒数，即8x60x60=28800

~~~json
{
    "place":{
        "supermarket":{
            "0":28800
        }
    }
}
~~~

### 2、将对应地点的视频文件复制到vido_input中

### 3、将目标人员的图片复制到target_input中

你可以下载我们的示例图片与视频，我们都已打包好。百度网盘链接：（待定）（密码：）

下载完之后，直接复制到项目主文件夹即可。

## 2、运行

装配好环境后，在命令行输入

~~~python
python ReID_main.py -v ./video_input -t ./target_input --data_output ./data_output --top_k 50 --skip 10
~~~

输入完成后点击回车键，等待程序运行完毕即可。

有几个参数是你需要注意的：

top-k：概率最大的前k个人。过大或过小都将影响结果。

skip：处理视频时，选择跳过n帧处理一次。若你希望不进行跳帧，则输入1.

## 3、结果

运行结果，可以在文件夹 data_output中的message中查看。

在拥有运行结果后

如 rm.npy

可以选择参数 --is_only_txt True 复现检查结果。

~~~python
python ReID_main.py -v ./video_input -t ./target_input --is_only_txt True --npy_path <your-rm-npy-path> --top_k 50 --skip 10
~~~

