# 🧠 Object Detection Using TensorFlow & OpenCV

This project performs real-time object detection using a pre-trained SSD MobileNet V2 model with TensorFlow and OpenCV.

## 📦 Features
- Real-time webcam object detection
- SSD MobileNet V2 trained on COCO dataset
- Displays bounding boxes and labels
- Simple and lightweight setup

## 🗂️ Project Structure
```
Object-Detection-Using-tensorflow/
├── model/
│   ├── frozen_inference_graph.pb
│   └── ssd_mobilenet_v2.pbtxt
├── utils/
│   └── label_map.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions
```bash
git clone https://github.com/Vishal-Palve/Object-Detection.git
cd Object-Detection

# (Optional) Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Run the Project
```bash
python main.py
```

## 📄 Example Labels (COCO)
```python
{
  1: 'person',
  2: 'bicycle',
  3: 'car',
  ...
}
```

## 📸 Sample Output
> *(Optional: Add screenshot here)*  
> `assets/sample_output.png`

## 📌 To-Do
- [ ] Web-based interface using Flask
- [ ] Export detection logs
- [ ] Train on custom objects

## 📜 License
Licensed under the [MIT License](LICENSE)

## 👨‍💻 Author
Made with ❤️ by [Vishal Palve](https://github.com/Vishal-Palve)
