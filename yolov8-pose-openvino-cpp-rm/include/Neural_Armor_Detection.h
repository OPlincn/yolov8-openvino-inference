#pragma once
#ifndef NEURALARMORDETECTOR_NEURAL_ARMOR_DETECTION_H
#define NEURALARMORDETECTOR_NEURAL_ARMOR_DETECTION_H
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>


//TODO 装甲板结构体, 引入项目以后可以直接使用 Armor_detection.h
//struct Armor : public cv::RotatedRect    //装甲板结构体
//    {
//        Armor() = default;
//        explicit Armor(cv::RotatedRect &box) : cv::RotatedRect(box)
//        {
//            confidence = 0;
//            id = 0;
//            type = SMALL;
//            grade = 0;
//        }
//        cv::Point2f armor_pt4[4];           // 左下角开始逆时针
//        float confidence;                   // 装甲板置信度
//        int id;                             // 装甲板类别
//        int grade;                          // 装甲板分数
//        int type;                           // 装甲板类型
//        Eigen::Vector3d world_position;     // 当前的真实坐标
//        Eigen::Vector3d camera_position;    // 当前的相机坐标
//        Eigen::Matrix<double, 3, 1> R;      // 旋转向量
//    };

struct Config {
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	std::string onnx_path;
};

struct Resize
{
	cv::Mat resized_image;
	int dw;
	int dh;
};

struct Detection {
	int class_id;
	float confidence;
	cv::Rect box;
    std::vector<cv::Point2f> keyPoint;
};

class NeuralArmorDetector {
public:
	NeuralArmorDetector(Config config);
	~NeuralArmorDetector();
	void detect(cv::Mat& frame);

private:
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	float rx;   // the width ratio of original image and resized image
	float ry;   // the height ratio of original image and resized image
    //
    int dx;
    int dy;
    //
	std::string onnx_path;
	Resize resize;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initial_model();
    void preprocess_img_letterBox(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float * detections, ov::Shape & output_shape);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
};



#endif //NEURALARMORDETECTOR_NEURAL_ARMOR_DETECTION_H
