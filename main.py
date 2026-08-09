from maix import image, display, app, camera, time, nn, uart, pinmap
import cv2
import numpy as np
import os

# ========== 串口配置 ==========
try:
    pinmap.set_pin_function("A18", "UART1_RX")
    pinmap.set_pin_function("A19", "UART1_TX")
    serial_dev = uart.UART("/dev/ttyS1", 115200)
    print("UART1 initialized on A18(RX), A19(TX)")
except Exception as e:
    print("UART init failed:", e)
    class DummyUART:
        def write_str(self, s): pass
        def read(self, size=-1): return None
    serial_dev = DummyUART()

# ========== 加载分类器模型 ==========
model_path = "model_275071.mud"
if not os.path.exists(model_path):
    model_path = "/root/models/maixhub/275071/model_275071.mud"
classifier = nn.Classifier(model=model_path)

# ========== 置信度阈值 ==========
CONF_THRESHOLD = 0.7   # 低于此值认为不可靠，返回1（tri作为默认）

# ========== 摄像头和显示 ==========
cam = camera.Camera(320, 240, image.Format.FMT_BGR888)
disp = display.Display()

# ========== 辅助函数 ==========
def send_result(code_int):
    """发送整数结果(0-4)到单片机，加换行"""
    try:
        serial_dev.write_str(str(code_int) + "\n")
        print("[UART] send:", code_int)
    except Exception as e:
        print("[UART] send error:", e)

def classifier_detection(img_maix):
    """
    返回:
        1: tri
        2: cir
        3: cube
        1: 置信度低于阈值或未识别（默认返回1）
    """
    w = classifier.input_width()
    h = classifier.input_height()
    fmt = classifier.input_format()
    img_resized = img_maix.resize(w, h)
    img_converted = img_resized.to_format(fmt)
    res = classifier.classify(img_converted)
    max_idx, max_prob = res[0]
    
    if max_prob < CONF_THRESHOLD:
        return 2  # 默认返回cir（与单片机协议一致）
    
    label = classifier.labels[max_idx]   # "cube", "tri", "cir"
    if label == "cir":
        return 2
    elif label == "cube":
        return 3
    elif label == "tri":
        return 1
    else:
        return 1

# ========== 主循环 ==========
print("系统启动，仅使用分类器检测...")
while not app.need_exit():
    img_maix = cam.read()
    # 转换为OpenCV格式用于显示文字（也可用maix.image绘制）
    frame = image.image2cv(img_maix, ensure_bgr=False, copy=False)

    # 分类器检测并发送结果
    code = classifier_detection(img_maix)
    detect_str = f"Classifier: {code} (1=tri, 2=cir, 3=cube)"
    send_result(code)

    # 屏幕显示
    cv2.putText(frame, detect_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Press button to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    img_show = image.cv2image(frame, bgr=True, copy=False)
    disp.show(img_show)

    time.sleep_ms(10)