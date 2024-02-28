#include"yolov8.h"
#include<iostream>
#include<string>
#include<time.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace dnn;

const vector<string> coconame = {
        "BG", "B1", "B2", "B3", "B4", "B5", "BO", "BBs", "BBb",
        "RG", "R1", "R2", "R3", "R4", "R5", "RO", "RBs", "RBb",
        "NG", "N1", "N2", "N3", "N4", "N5", "NO", "NBs", "NBb",
        "PG", "P1", "P2", "P3", "P4", "P5", "PO", "PBs", "PBb"
};

const std::vector<cv::Scalar> CLASS_COLORS = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(192, 192, 192), cv::Scalar(128, 128, 128), cv::Scalar(128, 0, 0),
        cv::Scalar(128, 128, 0), cv::Scalar(0, 128, 0), cv::Scalar(128, 0, 128),
        cv::Scalar(0, 128, 128), cv::Scalar(0, 0, 128), cv::Scalar(72, 61, 139),
        cv::Scalar(47, 79, 79), cv::Scalar(47, 79, 47), cv::Scalar(0, 100, 0),
        cv::Scalar(85, 107, 47), cv::Scalar(139, 69, 19), cv::Scalar(160, 82, 45),
        cv::Scalar(255, 140, 0), cv::Scalar(255, 165, 0), cv::Scalar(255, 215, 0),
        cv::Scalar(184, 134, 11), cv::Scalar(218, 165, 32), cv::Scalar(238, 232, 170),
        cv::Scalar(189, 183, 107), cv::Scalar(0, 128, 128), cv::Scalar(0, 139, 139),
        cv::Scalar(25, 25, 112), cv::Scalar(70, 130, 180), cv::Scalar(100, 149, 237),
        cv::Scalar(123, 104, 238), cv::Scalar(106, 90, 205), cv::Scalar(176, 196, 222)
};


YOLOV8::YOLOV8(Config config) {
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
    this->scoreThreshold = config.scoreThreshold;
	this->inpWidth = config.inpWidth;
	this->inpHeight = config.inpHeight;
	this->onnx_path = config.onnx_path;
    this->initialmodel();

}
YOLOV8::~YOLOV8(){}
void YOLOV8::detect(Mat & frame) {
//    clock_t start_pre = clock();

    preprocess_img_letterBox(frame);
//
//    clock_t end_pre = clock();
//
//    std::cout << "preProcess time = " << double(end_pre - start_pre) / CLOCKS_PER_SEC*1000 << "ms" << std::endl;

    //TODO count infer time

//    auto start = std::chrono::high_resolution_clock::now();
//
    infer_request.infer();
//    //TODO infer time
//
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    //异步推理
//    infer_request.start_async();
//    infer_request.wait();
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float* detections = output_tensor.data<float>();
    this->postprocess_img(frame, detections, output_shape);
}

void YOLOV8::initialmodel() {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(this->onnx_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
    //TODO f32 to f16， import colorFormat is BGR no RGB
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::BGR).scale({ 255, 255, 255 });// .scale({ 112, 112, 112 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32); //TODO change to 16
    model = ppp.build();
    this->compiled_model = core.compile_model(model, "AUTO",ov::hint::performance_mode (ov::hint::PerformanceMode::LATENCY), ov::hint::inference_precision(ov::element::f16)); //, ov::hint::num_requests(8))
    this->infer_request = compiled_model.create_infer_request();

}

//void YOLOV8::initialmodel() {
//    ov::Core core;
//    std::shared_ptr<ov::Model> model = core.read_model(this->onnx_path);
//
//    // 对于int8量化模型，通常不需要对输入进行复杂的预处理
//    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
//    ppp.input().tensor().set_element_type(ov::element::u8)
//            .set_layout("NHWC") // 将布局改为NCHW
//            .set_color_format(ov::preprocess::ColorFormat::RGB); // 明确指定颜色格式的布局
//    ppp.input().preprocess().convert_element_type(ov::element::f32); // 如果需要，继续进行数据类型转换
//    ppp.input().model().set_layout("NCHW");
//// 不需要进行颜色转换或缩放，因为模型已经量化
//    ppp.output().tensor().set_element_type(ov::element::f32); // 设置输出数据类型
//    model = ppp.build();
//
//
//    // 编译模型时指定使用INT8精度，如果硬件支持
//    this->compiled_model = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
//    // 如果你的硬件支持INT8优化，可以通过配置来启用它，例如使用ov::hint::inference_precision指定精度
//
//    this->infer_request = compiled_model.create_infer_request();
//}


void YOLOV8::preprocess_img_resize(Mat& frame) {
    try {
        // 获取原始图像的宽和高
        float width = frame.cols;
        float height = frame.rows;

        // 设置新的图像尺寸
        cv::Size new_shape = cv::Size(inpWidth, inpHeight);

        // 计算缩放比例
        float r = float(new_shape.width / max(width, height));
        int new_unpadW = int(round(width * r));
        int new_unpadH = int(round(height * r));

        // 使用INTER_LINEAR方法对图像进行缩放
        cv::resize(frame, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);

        // 计算新图像与目标尺寸的差异
        resize.dw = new_shape.width - new_unpadW;
        resize.dh = new_shape.height - new_unpadH;

        // 如果需要，添加边框以匹配目标尺寸
        if (resize.dw > 0 || resize.dh > 0) {
            cv::Scalar color = cv::Scalar(100, 100, 100); // 设置边框颜色
            cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);
        }

        // 计算原始图像与缩放后图像的比例
        this->rx = (float)frame.cols / (float)(resize.resized_image.cols - resize.dw);
        this->ry = (float)frame.rows / (float)(resize.resized_image.rows - resize.dh);

        // 准备模型输入数据
        float* input_data = (float*)resize.resized_image.data;
        input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}


void YOLOV8::preprocess_img_letterBox(cv::Mat &frame) {
    try {
        // 原始图像尺寸
        float orig_width = frame.cols;
        float orig_height = frame.rows;

        // 目标尺寸
        cv::Size target_size = cv::Size(inpWidth, inpHeight);

        // 计算缩放比例
        float width_ratio = target_size.width / orig_width;
        float height_ratio = target_size.height / orig_height;
        float scale_ratio = min(width_ratio, height_ratio);

        // 计算缩放后的尺寸
        int scaled_width = int(orig_width * scale_ratio);
        int scaled_height = int(orig_height * scale_ratio);

        // 缩放图像
        cv::Mat scaled_frame;
        cv::resize(frame, scaled_frame, cv::Size(scaled_width, scaled_height), 0, 0, cv::INTER_LINEAR);

        // 计算边框大小
        int top_border = (target_size.height - scaled_height) / 2;
        int bottom_border = target_size.height - scaled_height - top_border;
        int left_border = (target_size.width - scaled_width) / 2;
        int right_border = target_size.width - scaled_width - left_border;

        // 添加边框
        cv::Mat letterbox_frame;
        cv::copyMakeBorder(scaled_frame, letterbox_frame, top_border, bottom_border, left_border, right_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // 更新缩放比例和偏移量
        this->rx = orig_width / (float)scaled_width;
        this->ry = orig_height / (float)scaled_height;
        this->dx = left_border; // 水平偏移量
        this->dy = top_border;  // 垂直偏移量

//         准备模型输入数据 fp
        float* input_data = (float*)letterbox_frame.data;
        input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);

        // 将图像数据转换为INT8
//        cv::Mat input_data_int8;
//        letterbox_frame.convertTo(input_data_int8, CV_8S);
//
//        // 准备模型输入数据
//        int8_t* input_data = (int8_t*)input_data_int8.data;
//        input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
//        infer_request.set_input_tensor(input_tensor);

    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}




void YOLOV8::postprocess_img(Mat& frame, float* detections, ov::Shape & output_shape) {
    // 定义用于存储检测结果的容器
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    std::vector<std::vector<cv::Point2f>> keyPointS;

    int out_rows = output_shape[1];
    int out_cols = output_shape[2];
    // 从模型输出中创建一个Mat对象
    const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)detections);

    for (int i = 0; i < det_output.cols; ++i) {
        // 提取每个检测的类别分数
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 40);

        cv::Point class_id_point;
        double score;
        // 找到最高分数及其对应的类别
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // 如果分数高于置信度阈值，则处理这个检测
        if (score > confThreshold) {
            // 提取边界框坐标
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            // 将边界框信息添加到相应的列表中
            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);

            // 提取关键点
            vector<cv::Point2f> kpts;
            for (int j = 0; j < 5; ++j) { // 5个关键点
                float kpt_x = det_output.at<float>(40 + j * 2, i); // 关键点x坐标
                float kpt_y = det_output.at<float>(41 + j * 2, i); // 关键点y坐标
                kpts.push_back(cv::Point2f(kpt_x, kpt_y));
            }
            keyPointS.push_back(kpts);
        }
    }

    // 非极大值抑制（NMS）来过滤重叠的边界框
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, nms_result);

    // 存储最终的检测结果
    std::vector<Detection> output;
    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.keyPoint = keyPointS[idx];
        output.push_back(result);
    }

    // 在图像上绘制检测结果
    for (const auto& detection : output) {
        auto box = detection.box;
        int class_id = detection.class_id;
        cv::Scalar color = CLASS_COLORS[class_id % CLASS_COLORS.size()];
        //TODO resize
//        box.x = static_cast<int>(this->rx * box.x);
//        box.y = static_cast<int>(this->ry * box.y);
//        box.width = static_cast<int>(this->rx * box.width);
//        box.height = static_cast<int>(this->ry * box.height);

        //TODO letterBox
        box.x = static_cast<int>((box.x - this->dx) * this->rx);
        box.y = static_cast<int>((box.y - this->dy) * this->ry);
        box.width = static_cast<int>(box.width * this->rx);
        box.height = static_cast<int>(box.height * this->ry);

        float xmax = box.x + box.width;
        float ymax = box.y + box.height;

        // 绘制边界框-
        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), color, 3);

        // 绘制边界框上的文本
        std::string classString = coconame[detection.class_id] + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2);

        //TODO 绘制关键点 resize
//        for (const auto& kpt : detection.keyPoint) {
//            cv::circle(frame, cv::Point(static_cast<int>(this->rx * kpt.x), static_cast<int>(this->ry * kpt.y)), 3, cv::Scalar(0, 0, 255), -1);
//        }

        //TODO 绘制关键点 letterBox
        for (const auto& kpt : detection.keyPoint) {
            // 调整坐标以映射回原始图像
            int kpt_x = static_cast<int>((kpt.x - this->dx) * this->rx);
            int kpt_y = static_cast<int>((kpt.y - this->dy) * this->ry);

            cv::circle(frame, cv::Point(kpt_x, kpt_y), 3, cv::Scalar(0, 0, 255), -1);
        }

    }
}


