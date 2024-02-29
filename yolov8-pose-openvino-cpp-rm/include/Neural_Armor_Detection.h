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
#define armor_big_max_wh_ratio 5.0
#define armor_big_min_wh_ratio 3.0
#define armor_small_max_wh_ratio 3.0
#define armor_small_min_wh_ratio 0.8
#define near_standard 500
#define height_standard 25
#define grade_standard 60
//装甲板打分系数比例
#define id_grade_ratio 0.6
#define near_grade_ratio 0.2
#define height_grade_ratio 0.2


enum EnemyTypeEnemyType  { SMALL = 1, BIG = 2, BUFF_NO = 3, BUFF_YES = 4};
//TODO 装甲板结构体, 引入项目以后可以直接使用 Armor_detection.h
struct Armor : public cv::RotatedRect    //装甲板结构体
    {
        Armor() = default;
        explicit Armor(cv::RotatedRect &box) : cv::RotatedRect(box)
        {
            confidence = 0;
            id = 0;
            type = SMALL;
            grade = 0;
        }
        cv::Point2f armor_pt4[4];           // 左下角开始逆时针
        float confidence;                   // 装甲板置信度
        int id;                             // 装甲板类别
        int grade;                          // 装甲板分数
        int type;                           // 装甲板类型
        Eigen::Vector3d world_position;     // 当前的真实坐标
        Eigen::Vector3d camera_position;    // 当前的相机坐标
        Eigen::Matrix<double, 3, 1> R;      // 旋转向量
    };

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
    // frame size
    int frame_w;
    int frame_h;
	std::string onnx_path;
	Resize resize;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initial_model();
    void preprocess_img_letterBox(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float * detections, ov::Shape & output_shape);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
    std::vector<Armor> neuralArmorGrade(const std::vector<Detection>& candidateArmors);
    void drawArmors(cv::Mat &frame, const std::vector<Armor> &armors);
};



#endif //NEURALARMORDETECTOR_NEURAL_ARMOR_DETECTION_H
