# py-face-rekognition
Facial Recognition System using Python (>=3.6)
- Face Detection is based on the Caffe Single Shot Detector ([SSD](https://medium.com/acm-juit/ssd-object-detection-in-real-time-deep-learning-and-caffe-f41e40eea968)) framework with a ResNet10 base network
- Face Recognition is based on OpenCV's Local Binary Pattern Histogram ([LBPH](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b)) Algorithm

### Setup

```bash
git clone https://github.com/heisenbug237/py-face-rekognition.git
cd py-face-rekognition
pip install -r requirements.txt
```
### Test Image
```bash
python main.py sample.jpg
```
### Test Webcam
```bash
python main.py
```
### Sample Output
![Output](https://github.com/heisenbug237/py-face-rekognition/blob/main/faces.jpg?raw=true)
