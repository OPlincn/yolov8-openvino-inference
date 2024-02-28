#include <Neural_Armor_Detection.h>
#include <iostream>
#include <string>
#include <ctime>
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


NeuralArmorDetector::NeuralArmorDetector(Config config) {
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->scoreThreshold = config.scoreThreshold;
    this->inpWidth = config.inpWidth;
    this->inpHeight = config.inpHeight;
    this->onnx_path = config.onnx_path;
    this->initial_model();

}


NeuralArmorDetector::~NeuralArmorDetector(){}

/**
 * 对传入的画面进行装甲板检测.
 * 现在的返回类型是void,
 * 再将postprocess_img()函数修改为返回装甲板后,detect的数据1返回类型也应该修改
 *
 * @param frame 输入的原始图像。
 */
void NeuralArmorDetector::detect(Mat & frame) {
    preprocess_img_letterBox(frame);

    infer_request.infer();

    //异步推理
//    infer_request.start_async();
//    infer_request.wait();
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float* detections = output_tensor.data<float>();
    this->postprocess_img(frame, detections, output_shape);
}

/**
 * 初始化模型以供后续使用。
 *
 * 本函数执行以下主要任务：
 * 1. 创建OpenVINO核心对象并设置模型缓存目录。
 * 2. 读取ONNX模型文件。
 * 3. 设置模型的输入和输出预处理步骤，包括数据类型、布局和颜色格式转换等。
 * 4. 编译模型，优化性能，并创建推理请求对象。
 */
void NeuralArmorDetector::initial_model() {
    ov::Core core; // 创建OpenVINO核心对象。
    core.set_property(ov::cache_dir("../model/cache")); // 启用模型缓存功能，并指定缓存目录。

    // 读取模型文件，`this->onnx_path`是模型文件的路径。
    std::shared_ptr<ov::Model> model = core.read_model(this->onnx_path);

    // 创建预处理和后处理配置对象。
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    // 配置模型输入的预处理步骤。
    // 设置输入数据的元素类型为无符号8位整数，布局为NHWC，颜色格式为RGB。
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);

    // 配置模型输出的后处理步骤。
    // 设置输出数据的元素类型为32位浮点数。
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 配置进一步的输入预处理操作。
    // 转换元素类型为32位浮点数，颜色格式转换为BGR，并进行缩放。
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::BGR).scale({ 255, 255, 255 });

    // 设置模型的输入布局。
    ppp.input().model().set_layout("NCHW");

    // 应用预处理和后处理配置。
    model = ppp.build();

    // 编译模型，设置设备为自动选择（"AUTO"），优化性能模式为低延迟，推理精度为16位浮点数。
    this->compiled_model = core.compile_model(model, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), ov::hint::inference_precision(ov::element::f16));

    // 创建推理请求对象，用于执行推理。
    this->infer_request = compiled_model.create_infer_request();
}


/**
 * 对输入图像进行预处理，包括缩放和添加边框，以适应模型的输入尺寸。
 *
 * 此函数首先将图像缩放到接近模型输入尺寸的大小，然后在必要的边缘添加黑色边框，
 * 以确保图像完整地填充模型的输入尺寸，同时保持原始图像的宽高比不变。
 *
 * @param frame 输入的原始图像。
 */
void NeuralArmorDetector::preprocess_img_letterBox(cv::Mat &frame) {
    try {
        // 计算图像缩放比例，以确保缩放后的图像能够保持原始宽高比并适应目标尺寸。
        // 使用 static_cast<float>() 来确保 inpWidth/frame.cols 和 inpHeight/frame.rows 的计算结果为浮点数，避免整数除法导致的精度丢失。
        float scale_ratio = std::min(static_cast<float>(inpWidth) / frame.cols, static_cast<float>(inpHeight) / frame.rows);

        // 根据计算出的缩放比例，确定缩放后图像的新尺寸。
        // 这里将缩放比例应用于原图像的宽度和高度，并将结果转换为整数，因为像素的数量不能是小数。
        cv::Size scaled_size(int(frame.cols * scale_ratio), int(frame.rows * scale_ratio));

        // 使用 cv::resize 函数对原始图像进行缩放操作，得到缩放后的图像 scaled_frame。参数说明：
        // 0, 0：缩放时的x轴和y轴的比例，这里不使用这两个参数，因为缩放的大小已经通过 scaled_size 指定。
        // cv::INTER_LINEAR：缩放时使用的插值方法，INTER_LINEAR 表示双线性插值，适用于缩放操作，可以在保证速度的同时获得较好的视觉效果。
        cv::Mat scaled_frame;
        cv::resize(frame, scaled_frame, scaled_size, 0, 0, cv::INTER_LINEAR);

        // 计算边框的大小
        int top_border = (inpHeight - scaled_size.height) / 2;
        int bottom_border = inpHeight - scaled_size.height - top_border;
        int left_border = (inpWidth - scaled_size.width) / 2;
        int right_border = inpWidth - scaled_size.width - left_border;

        // 添加黑色边框以生成最终的输入图像
        cv::Mat letterbox_frame;
        cv::copyMakeBorder(scaled_frame, letterbox_frame, top_border, bottom_border, left_border, right_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // 更新缩放比例和偏移量，供后续处理模型推理结果时,将结果映射到原画面上时使用
        this->rx = static_cast<float>(frame.cols) / scaled_size.width;
        this->ry = static_cast<float>(frame.rows) / scaled_size.height;
        this->dx = left_border; // 水平偏移量
        this->dy = top_border;  // 垂直偏移量

        // 准备模型输入数据
        float* input_data = reinterpret_cast<float*>(letterbox_frame.data);
        ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "未知异常" << std::endl;
    }
}

/**
 * 对模型的检测结果进行后处理，包括解析检测结果、应用非极大值抑制（NMS）和在图像上绘制检测框和关键点。
 * 这是后处理函数的第二版, 第一版将模型输出转化为Mat类型的数据, 通过使用Mat的相关方法进行索引
 * 而此第二版使用的是指针在数据上直接进行操作, 减少了Mat对象的创建和引用, 延迟降低了 1ms~2ms.
 *
 * 对于模型输出的解释: 模型的输出是连续的8400x50个数, 可以相当于一个行(row)数位50,列(col)数为8400的矩阵,每一列都是模型的一个检测结果的完整检测数据,
 * 也就是说每一行都是每一个检测结果的相同属性,比如这个矩阵的第一行,是每个检测结果的bbox的cx,二第二行是每个检测结果的cy.
 * 又因为模型的输出结果是连续的,相当于
 * 矩阵[1, 2
 *      3, 4]表达为连续的 1, 2, 3, 4.
 * 所以由此可推从第0位到第8399位是8400个检测结果的bbox的cx值,接下来从8400到16799是8400个检测结果的bbox的cy值
 *
 * @param frame 输入的图像，将在其上绘制检测结果。
 * @param detections 模型的原始输出，包含检测到的对象的信息。
 * @param output_shape 模型输出的维度信息。
 */

void NeuralArmorDetector::postprocess_img(cv::Mat &frame, float *detections, ov::Shape &output_shape) {
    // 模型输出的维度，其中行数为属性数量，列数为检测结果数量
    //int num_attributes = output_shape[1]; // 属性数量，这个模型是50个, 为什么用不上注释了呢?因为下面所有属性的提取我们都是写好的,用不上这个值来循环
    int num_detections = output_shape[2]; // 检测结果数量，这个模型是8400个检测结果

    // 定义用于存储检测结果的容器
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    std::vector<std::vector<cv::Point2f>> keyPointS;

    //TODO 下面这段注释,
    // 采用分块求和的方式，即对每一个检测结果的置信度进行累加
    // 累加一定数量的类别置信度后就检查当前的和是否已经超过阈值，如果超过则提前终止求和，进一步处理这个检测结果。
    // 如果没有超过阈值就不寻找其中的的最大值, 这是否是一种优化的方式? 因为求和比用条件判断更快!使用以下筛选需要在 postprocess的最后面补上1个 }
//    float sum_threshold = 0.7f; // 设置一个求和阈值
//
//    for (int i = 0; i < num_detections; ++i) {
//        float sum_score = 0.0;
//
//        // 对每个检测结果的类别置信度求和
//        for (int j = 0; j < 36; ++j) { // 假设有36个类别
//            sum_score += detections[i + (4 + j) * num_detections];
//            if (sum_score > sum_threshold) {
//                break; // 如果求和结果已超过阈值，则提前终止求和
//            }
//        }
//        float max_score = 0.0;
//        if (sum_score <= sum_threshold) {
//            continue; // 如果求和结果未超过阈值，则跳过这个检测结果
//        } else {
//            int class_id = -1;
//            for (int j = 0; j < 36; ++j) { // 假设有36个类别
//                float score = detections[i + (4 + j) * num_detections];
//                if (score > max_score) {
////                if (score > 0.7) cout << "score" << j << " : " << score << endl;
//                    max_score = score;
//                    class_id = j;
//                }
//            }

    //TODO 这个是常规的筛选方式, 对每个检测结果,寻找其最大的score, 然后用max_Score去和confThreshold进行比较
    // 相比上面要少个}

    for (int i = 0; i < num_detections; ++i) {
        // 遍历每一个检测结果
        // 从检测结果中提取最大的类别分数和对应的类别ID
        float max_score = 0.0;
        int class_id = -1;
        for (int j = 0; j < 36; ++j) { // 这里36指遍历36个类别的置信度,如果只要BLUE和RED的置信度,可以将其改为18, 直接拦腰截断了一半
            float score = detections[i + (4 + j) * num_detections];
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }

        // 如果最大分数大于置信度阈值，则记录该检测结果
        if (max_score > confThreshold) {
            float cx = detections[i]; // 第0行到第8399行是cx
            float cy = detections[i + num_detections]; // 第8400行到第16799行是cy
            float ow = detections[i + 2 * num_detections]; // 宽 16800 ~ 25199
            float oh = detections[i + 3 * num_detections]; // 高
            // 计算边界框的左上角点和宽高
            cv::Rect box(static_cast<int>(cx - 0.5 * ow), static_cast<int>(cy - 0.5 * oh), static_cast<int>(ow), static_cast<int>(oh));
            boxes.push_back(box);
            class_ids.push_back(class_id);
            confidences.push_back(max_score);

            // 提取关键点, 10位数 也就是五组xy.
            // 关键点的顺序为: 中心, 左上, 左下, 右下, 右上
            vector<cv::Point2f> kpts;
            for (int k = 0; k < 5; ++k) {
                float kpt_x = ((detections[i + (40 + k * 2) * num_detections] - this->dx) * this->rx);
                float kpt_y = ((detections[i + (41 + k * 2) * num_detections] - this->dy) * this->ry);
                kpts.push_back(cv::Point2f(kpt_x, kpt_y));
            }
            keyPointS.push_back(kpts);
        }
    }

    // 非极大值抑制（NMS）
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nms_result);


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
    // 使用Armor结构体
//        std::vector<Armor> ArmorList;
//    for (int idx : nms_result) {
//        Armor result;
//        result.class_id = class_ids[idx];
//        result.confidence = confidences[idx];
//        result.box = boxes[idx];
//        result.keyPoint = keyPointS[idx];
//        output.push_back(result);
//    }
    drawDetections(frame, output);
}



/**
 * 将模型的检测结果在画面中可视化出来,包括文本,bbox,关键点
 *
 * @param frame 输入的图像，将在其上绘制检测结果。
 * @param detections 经过postprocess处理后的输出，是可用的装甲板的信息。
 */
void NeuralArmorDetector::drawDetections(cv::Mat &frame, const std::vector<Detection> &detections) {

    for (const auto &detection: detections) {
        auto box = detection.box;
        int class_id = detection.class_id;
        cv::Scalar color = CLASS_COLORS[class_id % CLASS_COLORS.size()];

        //因为使用了 letterBox 来处理图像,所以bbox的正确属参数需要映射回去!
        box.x = static_cast<int>((box.x - this->dx) * this->rx);
        box.y = static_cast<int>((box.y - this->dy) * this->ry);
        box.width = static_cast<int>(box.width * this->rx);
        box.height = static_cast<int>(box.height * this->ry);

        float xmax = box.x + box.width;
        float ymax = box.y + box.height;

        // 绘制边界框-
        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), color, 3);

        // 绘制边界框上的文本
        std::string classString =
                coconame[detection.class_id] + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 0, 0), 2);


        // 绘制关键点 因为使用letterBox来处理了图像,所以要把推理得到的点映射回原来的图像上
        for (const auto &kpt: detection.keyPoint) {
            // 调整坐标以映射回原始图像, 提取坐标的时候已经映射了
//            int kpt_x = static_cast<int>((kpt.x - this->dx) * this->rx);
//            int kpt_y = static_cast<int>((kpt.y - this->dy) * this->ry);

            cv::circle(frame, cv::Point(int(kpt.x), int(kpt.y)), 3, cv::Scalar(0, 0, 255), -1);
        }

    }
}

// 使用Mat索引的postprocess
//void NeuralArmorDetector::postprocess_img(Mat& frame, float* detections, ov::Shape & output_shape) {
//    // 定义用于存储检测结果的容器
//    std::vector<cv::Rect> boxes;
//    vector<int> class_ids;
//    vector<float> confidences;
//    std::vector<std::vector<cv::Point2f>> keyPointS;
//
//    int out_rows = output_shape[1];
//    int out_cols = output_shape[2];
//    // 从模型输出中创建一个Mat对象
//    const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)detections);
//
//    for (int i = 0; i < det_output.cols; ++i) {
//        // 提取每个检测的类别分数
//        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 40);
//
//        cv::Point class_id_point;
//        double score;
//        // 找到最高分数及其对应的类别
//        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
//
//        // 如果分数高于置信度阈值，则处理这个检测
//        if (score > confThreshold) {
//            // 提取边界框坐标
//            const float cx = det_output.at<float>(0, i);
//            const float cy = det_output.at<float>(1, i);
//            const float ow = det_output.at<float>(2, i);
//            const float oh = det_output.at<float>(3, i);
//            cv::Rect box;
//            // 因为在nms以后,不需要用到bbox的数据了,所以就不把bbox映射回去了
////            box.x = static_cast<int>(((cx - 0.5 * ow) - this->dx) * this->rx);
////            box.y = static_cast<int>(((cy - 0.5 * oh) - this->dy) * this->ry);
////            box.width = static_cast<int>(ow * this->rx);
////            box.height = static_cast<int>(oh * this->ry);
//
//            // unmap box, 以下是未映射的bbox属性
//            box.x = static_cast<int>((cx - 0.5 * ow));
//            box.y = static_cast<int>((cy - 0.5 * oh));
//            box.width = static_cast<int>(ow);
//            box.height = static_cast<int>(oh);
//
//            // 将边界框信息添加到相应的列表中
//            boxes.push_back(box);
//            class_ids.push_back(class_id_point.y);
//            confidences.push_back(score);
//
//            // 提取关键点, 因为关键点的信息后面要用上, 所以需要将其映射回原来的图像上!
//            vector<cv::Point2f> kpts;
//            for (int j = 0; j < 5; ++j) { // 5个关键点
//                /*
//                 * 五个关键点的储存顺序为: 中心点, 左上, 左下, 右下, 右上.
//                 *
//                 * */
//                float kpt_x = ((det_output.at<float>(40 + j * 2, i) - this->dx) * this->rx); // 关键点x坐标
//                float kpt_y = ((det_output.at<float>(41 + j * 2, i) - this->dy) * this->ry); // 关键点y坐标
//                kpts.push_back(cv::Point2f(kpt_x, kpt_y));
//            }
//            keyPointS.push_back(kpts);
//        }
//    }
//
//    // 非极大值抑制（NMS）来过滤重叠的边界框
//    std::vector<int> nms_result;
//    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, nms_result);
//    std::vector<Detection> output;
//    for (int idx : nms_result) {
//        Detection result;
//        result.class_id = class_ids[idx];
//        result.confidence = confidences[idx];
//        result.box = boxes[idx];
//        result.keyPoint = keyPointS[idx];
//        output.push_back(result);
//    }
//
//    // 在图像上绘制检测结果
//    drawDetections(frame, output);
//}


