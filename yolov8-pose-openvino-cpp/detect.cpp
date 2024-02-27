# include"yolov8.h"

int main(int argc, char* argv[]) {
    
    try{
//        if(argc!=3){
//            std::cout<<"Usage:"<<argv[0]<<" <path_to_model> <path_to_image>"<<std::endl;
//            return EXIT_FAILURE;
//        }
//        const std::string input_model_path {argv[1]};
//        const std::string input_image_path {argv[2]};
//        Config config = { 0.2,0.4,0.4,640,640, input_model_path};
//        clock_t start, end;
//        cv::Mat img = cv::imread(input_image_path);
        Config config = { 0.2,0.4,0.4,640,640, "/home/adminpc/文档/PythonProjects/yolov8-keypoints/ep200-bz64-noMosaic/train/weights/best_openvino_model_fp32/best.xml"};
        clock_t start, end;
//        cv::Mat img = cv::imread(input_image_path);
//        cv::Mat img = cv::imread("/home/adminpc/文档/PythonProjects/TUP-NN-Train-2/testData/image/498.jpg");
        cv::Mat img = cv::imread("/home/adminpc/文档/PythonProjects/yolov8-keypoints/openvino_demo_cpp/yolov8-openvino/result/result.jpg");

        YOLOV8 yolomodel(config);
        start = clock();
        yolomodel.detect(img);
        end = clock();
        std::cout << "infer time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
        cv::imwrite("result.jpg", img);
    }catch (const std::exception& ex){
        std::cerr << ex.what()<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}

