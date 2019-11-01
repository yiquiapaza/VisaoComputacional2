// Trabalho02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
	std::string imgName = "exemplo3";
	cv::Mat img = cv::imread("D:/" + imgName + ".jpg");

	if (img.empty())
	{
		std::cout << "Error, cant load ur image" << std::endl;
	}
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	cv::Mat imgThreshold;
	cv::threshold(gray, imgThreshold, 127, 255, cv::THRESH_BINARY_INV);

	std::vector<cv::Vec2f> lines;

	cv::HoughLines(imgThreshold, lines, 1, CV_PI / 180, 585, 0, 0);
	cv::Point2f pt11, pt12;

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt11.x = cvRound(x0 + 1000 * (-b));
		pt11.y = cvRound(y0 + 1000 * (a));
		pt12.x = cvRound(x0 - 1000 * (-b));
		pt12.y = cvRound(y0 - 1000 * (a));
	}
	//cv::line(img, pt11, pt12, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	cv::threshold(gray, imgThreshold, 160, 255, cv::THRESH_BINARY_INV);
	cv::line(imgThreshold, pt11, pt12, cv::Scalar(0, 0, 0), 16, cv::LINE_AA);

	cv::Point2f pt21, pt22;
	cv::HoughLines(imgThreshold, lines, 1, CV_PI / 180, 470, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt21.x = cvRound(x0 + 1000 * (-b));
		pt21.y = cvRound(y0 + 1000 * (a));
		pt22.x = cvRound(x0 - 1000 * (-b));
		pt22.y = cvRound(y0 - 1000 * (a));
	}
	//cv::line(img, pt21, pt22, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

	float gradient1 = ((pt12.y - pt11.y) / (pt12.x - pt11.x));
	float gradient2 = ((pt22.y - pt21.y) / (pt22.x - pt21.x));

	float x = ((pt21.y - gradient2 * pt21.x) - (pt11.y - gradient1 * pt11.x)) / (gradient1 - gradient2);
	float y = gradient1 * x + (pt11.y - gradient1 * pt11.x);

	cv::Point point1, point2;

	if (imgName == "exemplo1")
	{
		float c = y - (-1 / gradient2) * x;
		point1.x = -(10 - c) * gradient2;
		point1.y = 10;

		point2.x = -(img.rows - 50 - c) * gradient2;
		point2.y = img.rows - 50;
		cv::line(img, point1, point2, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
		cv::line(img, pt21, pt22, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
	}
	else
	{
		float c = y - (-1 / gradient1) * x;
		point1.x = -(10 - c) * gradient1;
		point1.y = 10;

		point2.x = -(img.rows - 50 - c) * gradient1;
		point2.y = img.rows - 50;
		cv::line(img, point1, point2, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
		cv::line(img, pt11, pt12, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
	}
	cv::circle(img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1, 32);

	cv::threshold(gray, imgThreshold, 127, 255, cv::THRESH_BINARY_INV);
	cv::line(imgThreshold, pt11, pt12, cv::Scalar(0, 0, 0), 64, cv::LINE_AA);
	cv::line(imgThreshold, pt21, pt22, cv::Scalar(0, 0, 0), 64, cv::LINE_AA);
	cv::Mat rotateMatrix;
	cv::Mat imgRotate;
	if (imgName == "exemplo2")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -30, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
	}
	if (imgName == "exemplo3")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -20, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
	}
	if (imgName == "exemplo1")
	{
		imgRotate = imgThreshold;
	}
	std::vector<cv::Point> _points;
	for (int j = 0; j < imgRotate.rows; j++)
	{
		for (int i = 0; i < imgRotate.cols; i++)
		{
			//std::cout << (int)imgRotate.at<uchar>(i, j) << std::endl;
			if ((int)imgRotate.at<uchar>(j, i) == 255)
			{
				//std::cout << i << " " << j << std::endl;
				_points.push_back(cv::Point(i, j));
			}
		}
	}
	
	for (auto item : _points) {
		cv::circle(img, item, 1, cv::Scalar(255, 0, 255), -1, 1);
	}

	cv::Mat_<float> M(3,3), N(3,1);
	for (size_t i = 0; i < 3; i++)
	{
		int num = std::rand() % _points.size() + 1;
		for (size_t j = 0; j < 3; j++)
		{
			if (j == 0)
				M(i, j) = cv::pow(_points[num].x,2);
			if (j == 1)
				M(i, j) = _points[num].x;
			if (j == 2)
				M(i, j) = 1;
		}
		N(i, 0) = _points[num].y;
	}
	cv::Mat_<float> out;
	std::cout << M << std::endl;
	std::cout << N << std::endl;

	if (cv::solve(M, N, out, cv::DECOMP_LU))
		std::cout << out << std::endl;
	else
		std::cout << "Cant solve this system" << std::endl;


	for (size_t i = 0; i < 900; i++)
	{
		cv::circle(img, cv::Point(i, out.at<float>(0, 0)* std::pow(i, 2) + out.at<float>(1, 0) * i + out.at<float>(2, 0)), 1, cv::Scalar(255, 255, 0), -1, 2);		
	}
	//cv::circle(img, cv::Point(-(10 - c) * gradient1, 10), 32, cv::Scalar(255, 255, 0), -1, 32);
	//cv::circle(img, cv::Point(-(img.rows - 50 - c) * gradient1, img.rows - 50), 32, cv::Scalar(255, 255, 0), -1, 32);


	//std::cout << -(10 - c) * gradient1 << " " << -(img.rows - 50 - c) * gradient1 << std::endl;
	 
	cv::imshow("Color Image", img);
	cv::imshow("Gray Image", gray);
	cv::imshow("Threshold Image", imgThreshold);
	cv::imshow("Rotate Image", imgRotate);

	cv::waitKey(0);
}
