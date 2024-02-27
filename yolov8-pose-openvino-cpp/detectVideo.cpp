#include "yolov8.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    try {

        Config config = { 0.75,0.30,0.60,640,640, "/home/oplin/CProjects/yolov8-openvino/model/ep165-bz32-SOAT/weights/best_openvino_model_int8/best.xml"};
        YOLOV8 yolomodel(config);

        cv::VideoCapture cap("/home/oplin/CProjects/yolov8-openvino/testSource/2.mp4");


        if (!cap.isOpened()) {
            std::cerr << "Error opening video file" << std::endl;
            return EXIT_FAILURE;
        }

        // 在视频处理开始前初始化变量
        auto total_duration = std::chrono::milliseconds(0);
        int frame_count = 0;

        cv::Mat frame;
        while (cap.read(frame)) {
//            clock_t start = clock();

            auto start = std::chrono::high_resolution_clock::now();
            yolomodel.detect(frame);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            total_duration += duration; // 累加每一帧的处理时间
            ++frame_count; // 增加帧计数
            std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

//            clock_t end = clock();
//
//            std::cout << "Infer time = " << double(end - start) / CLOCKS_PER_SEC*1000 << "ms" << std::endl;
            // 在视频处理结束后计算并打印平均推理时间
            if (frame_count > 0) {
                auto avg_duration = total_duration.count() / frame_count;
                std::cout << "Average inference time: " << avg_duration << " ms" << std::endl;
            }
            // 展示处理后的帧
            cv::imshow("YOLOv8 Detection", frame);
            if (cv::waitKey(1) == 27) break; // 按 'ESC' 退出
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
