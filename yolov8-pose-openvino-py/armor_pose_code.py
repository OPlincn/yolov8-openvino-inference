import cv2
import numpy as np
from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat

# 定义类别名称和对应的颜色
coconame = [
    "BG", "B1", "B2", "B3", "B4", "B5", "BO", "BBs", "BBb",
    "RG", "R1", "R2", "R3", "R4", "R5", "RO", "RBs", "RBb",
    "NG", "N1", "N2", "N3", "N4", "N5", "NO", "NBs", "NBb",
    "PG", "P1", "P2", "P3", "P4", "P5", "PO", "PBs", "PBb"
]

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128),
    (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 100, 0),
    (85, 107, 47), (139, 69, 19), (160, 82, 45),
    (255, 140, 0), (255, 165, 0), (255, 215, 0),
    (184, 134, 11), (218, 165, 32), (238, 232, 170),
    (189, 183, 107), (0, 128, 128), (0, 139, 139),
    (25, 25, 112), (70, 130, 180), (100, 149, 237),
    (123, 104, 238), (106, 90, 205), (176, 196, 222)
]

class YOLOV8:
    def __init__(self, config):
        # 初始化模型配置参数
        self.conf_threshold = config['confThreshold']
        self.nms_threshold = config['nmsThreshold']
        self.score_threshold = config['scoreThreshold']
        self.inp_width = config['inpWidth']
        self.inp_height = config['inpHeight']
        self.onnx_path = config['onnx_path']
        self.initial_model()

    def initial_model(self):
        # 加载OpenVINO模型
        core = Core()
        model = core.read_model(self.onnx_path)
        ppp = PrePostProcessor(model)

        # 设置模型输入配置
        ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.BGR).scale([255, 255, 255])
        ppp.input().model().set_layout(Layout("NCHW"))

        # 设置模型输出配置
        ppp.output().tensor().set_element_type(Type.f32)

        # 构建并编译模型
        model = ppp.build()
        self.compiled_model = core.compile_model(model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()

    def preprocess_img_letterbox(self, frame):
        # 计算缩放比例和添加黑边的偏移量
        orig_height, orig_width, _ = frame.shape
        width_ratio = self.inp_width / orig_width
        height_ratio = self.inp_height / orig_height
        scale_ratio = min(width_ratio, height_ratio)
        scaled_width, scaled_height = int(orig_width * scale_ratio), int(orig_height * scale_ratio)

        # 缩放图像并添加黑边
        scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        top, bottom = (self.inp_height - scaled_height) // 2, (self.inp_height - scaled_height + 1) // 2
        left, right = (self.inp_width - scaled_width) // 2, (self.inp_width - scaled_width + 1) // 2
        letterbox_frame = cv2.copyMakeBorder(scaled_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 记录缩放和偏移量信息
        self.rx, self.ry = orig_width / scaled_width, orig_height / scaled_height
        self.dx, self.dy = left, top
        return letterbox_frame

    def detect(self, frame):
        # 对输入图像进行预处理
        letterbox_frame = self.preprocess_img_letterbox(frame)
        self.infer_request.infer({self.compiled_model.input().get_any_name(): np.expand_dims(letterbox_frame, 0)})

        # 获取模型输出
        output_tensor = self.infer_request.get_output_tensor()
        detections = output_tensor.data
        output_shape = output_tensor.shape

        # 后处理并在原始图像上绘制检测结果
        self.postprocess_img(frame, letterbox_frame, detections, output_shape)

    def postprocess_img(self, frame, letterbox_frame, detections, output_shape):
        boxes, class_ids, confidences, keypoints = [], [], [], []

        for i in range(output_shape[2]):
            detection = detections[0, :, i]
            scores = detection[4:40]
            class_id = np.argmax(scores)
            score = scores[class_id]

            if score > self.conf_threshold:
                cx, cy, w, h = detection[:4]
                box = [cx - 0.5 * w, cy - 0.5 * h, w, h]
                boxes.append(box)
                class_ids.append(class_id)
                confidences.append(score)
                kpts = [[detection[j], detection[j + 1]] for j in range(40, 50, 2)]
                keypoints.append(kpts)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.nms_threshold)

        for idx in indices:
            box = boxes[idx]
            class_id = class_ids[idx]
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

            # 转换坐标回原始图像尺寸
            box[0] = int((box[0] - self.dx) * self.rx)
            box[1] = int((box[1] - self.dy) * self.ry)
            box[2] = int(box[2] * self.rx)
            box[3] = int(box[3] * self.ry)

            xmax, ymax = box[0] + box[2], box[1] + box[3]

            # 在原始图像上绘制边界框和类别标签
            cv2.rectangle(frame, (box[0], box[1]), (xmax, ymax), color, 3)
            class_str = f"{coconame[class_id]} {confidences[idx]:.2f}"
            cv2.putText(frame, class_str, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制关键点
            for kpt in keypoints[idx]:
                kpt_x = int((kpt[0] - self.dx) * self.rx)
                kpt_y = int((kpt[1] - self.dy) * self.ry)
                cv2.circle(frame, (kpt_x, kpt_y), 3, (0, 0, 255), -1)

if __name__ == "__main__":
    config = {
        'confThreshold': 0.75,
        'nmsThreshold': 0.4,
        'scoreThreshold': 0.75,
        'inpWidth': 640,
        'inpHeight': 640,
        'onnx_path': 'module/epz165_fp32_SOTA/best.xml'  # 替换为你的模型路径
    }

    detector = YOLOV8(config)
    input_type = 'video'  # 可以改为 'image' 来选择图片输入

    if input_type == 'image':
        frame = cv2.imread('path_to_your_image.jpg')  # 替换为你的图片路径
        detector.detect(frame)
        cv2.imshow('Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_type == 'video':
        cap = cv2.VideoCapture('/Users/oplin/OpDocuments/VscodeProjects/yoloV8/2.mp4')  # 替换为你的视频路径

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detector.detect(frame)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
