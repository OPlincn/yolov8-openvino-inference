#pragma once
#include<string>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>


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

class YOLOV8 {
public:
	YOLOV8(Config config);
	~YOLOV8();
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
	void initialmodel();
	void preprocess_img_resize(cv::Mat& frame);
    void preprocess_img_letterBox(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float * detections, ov::Shape & output_shape);
};