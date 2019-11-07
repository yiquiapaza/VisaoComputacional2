// Trabalho02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
	std::string imgName = "exemplo2";
	cv::Mat img = cv::imread("D:/" + imgName + ".jpg");

	if (img.empty())
	{
		std::cout << "Error, cant load ur image" << std::endl;
	}
	cv::Mat clone = img.clone();
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	cv::Mat imgThreshold;
	cv::threshold(gray, imgThreshold, 127, 255, cv::THRESH_BINARY_INV);

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/gray_1.png", gray);
	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/threshold_1.png", imgThreshold);

	double const  PI = 3.1415926536;

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

	double _theta = 0;
	if (imgName == "exemplo1")
	{
		_theta = std::atanf((gradient1 - 1) / (1 + gradient1 * 1)) ;
		std::cout << "Example1: " << _theta << std::endl;
	}
	else
	{
		_theta = std::atanf((gradient2 - 1) / (1 + gradient2 * 1)) * (180 / PI);
		std::cout << "ExampleX: " << _theta << std::endl;
	}
	
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

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/threshold_1_without_axes.png", imgThreshold);

	cv::Mat rotateMatrix;
	cv::Mat imgRotate;
	cv::Mat imgRotateColor;
	if (imgName == "exemplo2")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		imgRotate = imgThreshold;

	}
	if (imgName == "exemplo3")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		imgRotate = imgThreshold;

	}
	if (imgName == "exemplo1")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
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

	cv::Mat_<float> M(3, 3), N(3, 1);
	cv::Mat_<float> out;
	bool endIteration = true;
	int index_ransanc[3];
	//0.001 exemplo3 and exemplo2
	//0.1 examplo1
	float threshld = 0.001;
	int num = 0, colector = 0;
	bool a = true;
	float dist, dist1 = 0, dist2 = 0, general = 0;
	std::vector<cv::Point> _tmpPoint;
	for (int ij = 0; ij < 1000; ij++)
	{
		for (size_t i = 0; i < 3; i++)
		{
			num = std::rand() % _points.size();
			index_ransanc[i] = num;
			for (size_t j = 0; j < 3; j++)
			{
				if (j == 0)
					M(i, j) = cv::pow(_points[num].x, 2);
				if (j == 1)
					M(i, j) = _points[num].x;
				if (j == 2)
					M(i, j) = 1;
			}
			N(i, 0) = _points[num].y;
		}

		if (cv::solve(M, N, out, cv::DECOMP_LU))
		{
			//std::cout << out << std::endl;
			for (auto _point : _points)
			{
				dist = std::sqrtf(std::pow(out.at<float>(0, 0) * _point.x + out.at<float>(1, 0) * _point.x + out.at<float>(2, 0) - _point.y, 2));
				/*
				general = std::sqrtf(out.at<float>(1, 0) - 4 * out.at<float>(0, 0) * (out.at<float>(2, 0) - _point.y));
				dist1 = std::sqrtf(std::pow(_point.x - (-out.at<float>(1, 0) - general) / (2 * out.at<float>(0, 0)), 2));
				dist2 = std::sqrtf(std::pow(_point.x - (-out.at<float>(1, 0) + general) / (2 * out.at<float>(0, 0)), 2));
				if (dist1 > dist2)
				{
					dist = dist2;
				}
				else
				{
					dist = dist1;
				}
				*/
				if (threshld >= dist)
				{
					_tmpPoint.push_back(_point);
				}
			}
			if (ij == 0)
			{
				colector = _tmpPoint.size();
				_tmpPoint.clear();
			}
			else
			{
				if (colector < _tmpPoint.size())
				{
					colector = _tmpPoint.size();				
				}
			}
		}
		else
		{
			std::cout << "Cant solve this system" << std::endl;
		}
	}
	
	for (auto item : _points)
	{
		cv::circle(img, item, 1, cv::Scalar(0, 0, 255), -1, 1);
	}
	
	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/just_points_1.png", img);
	
	for (auto item : _tmpPoint) {
		cv::circle(clone, item, 1, cv::Scalar(255, 0, 0), -1, 1);
	}

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/using_ransac_1.png", clone);

	cv::Mat_<float> M1(3, 3);
	M1 = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat_<float> N1(3, 1);
	N1 = cv::Mat::zeros(3, 1, CV_32F);

	M1.at<float>(0, 0) = _tmpPoint.size();
	for (int i = 0; i < _tmpPoint.size(); i++)
	{
		M1.at<float>(1, 0) += _tmpPoint[i].x;
		M1.at<float>(0, 1) += _tmpPoint[i].x;

		M1.at<float>(2, 0) += std::pow(_tmpPoint[i].x, 2);
		M1.at<float>(1, 1) += std::pow(_tmpPoint[i].x, 2);
		M1.at<float>(0, 2) += std::pow(_tmpPoint[i].x, 2);

		M1.at<float>(1, 2) += std::pow(_tmpPoint[i].x, 3);
		M1.at<float>(2, 1) += std::pow(_tmpPoint[i].x, 3);

		M1.at<float>(2, 2) += std::pow(_tmpPoint[i].x, 4);

		N1.at<float>(0, 0) += _tmpPoint[i].y;
		N1.at<float>(1, 0) += _tmpPoint[i].x * _tmpPoint[i].y;
		N1.at<float>(2, 0) += std::pow(_tmpPoint[i].x, 2) * _tmpPoint[i].y;
	}


	std::cout << M1 << std::endl;
	std::cout << N1 << std::endl;
	cv::Mat_<float> out1;
	if (cv::solve(M1, N1, out1, cv::DECOMP_LU))
	{
		std::cout << out1 << std::endl;
		std::cout << out << std::endl;
	}
	else
	{
		std::cout << "Cant solve this system" << std::endl;
	}
	/*
	for (size_t i = 0; i < 900; i++)
	{
		cv::circle(img, cv::Point(i, out.at<float>(0, 0)* std::pow(i, 2) + out.at<float>(1, 0) * i + out.at<float>(2, 0)), 1, cv::Scalar(255, 255, 0), -1, 2);
	}
	*/
	float rotation_edge_x = 0;
	float rotation_edge_y = 0;
	
	float edge = -(_theta)* (PI / 180);
	float _x1 = 0, _y1 = 0;
	for (int i = -900; i < 900; i++)
	{
		_x1 = i - img.cols / 2 ;
		_y1 = (out1.at<float>(2, 0) * std::pow(i, 2) + (double)out1.at<float>(1, 0) * i + out1.at<float>(0, 0)) - img.rows / 2 ;
		rotation_edge_x = (_x1)*std::cos(edge) - (_y1)*std::sin(edge) + img.cols / 2 ;
		rotation_edge_y = (_x1)*std::sin(edge) + (_y1)*std::cos(edge) + img.rows / 2 ;
		cv::circle(img, cv::Point(rotation_edge_x, rotation_edge_y), 1, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);		
	}

	cv::imshow("Color Image", img);
	cv::imshow("Gray Image", gray);
	cv::imshow("Threshold Image", imgThreshold);

	//cv::imshow("Rotate Image", imgRotate);
	//cv::imshow("Rotate Image Color", imgRotateColor);

	cv::waitKey(0);
}// Trabalho02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
	std::string imgName = "exemplo2";
	cv::Mat img = cv::imread("D:/" + imgName + ".jpg");

	if (img.empty())
	{
		std::cout << "Error, cant load ur image" << std::endl;
	}
	cv::Mat clone = img.clone();
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	cv::Mat imgThreshold;
	cv::threshold(gray, imgThreshold, 127, 255, cv::THRESH_BINARY_INV);

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/gray_1.png", gray);
	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/threshold_1.png", imgThreshold);

	double const  PI = 3.1415926536;

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

	double _theta = 0;
	if (imgName == "exemplo1")
	{
		_theta = std::atanf((gradient1 - 1) / (1 + gradient1 * 1)) ;
		std::cout << "Example1: " << _theta << std::endl;
	}
	else
	{
		_theta = std::atanf((gradient2 - 1) / (1 + gradient2 * 1)) * (180 / PI);
		std::cout << "ExampleX: " << _theta << std::endl;
	}
	
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

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/threshold_1_without_axes.png", imgThreshold);

	cv::Mat rotateMatrix;
	cv::Mat imgRotate;
	cv::Mat imgRotateColor;
	if (imgName == "exemplo2")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		imgRotate = imgThreshold;

	}
	if (imgName == "exemplo3")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		imgRotate = imgThreshold;

	}
	if (imgName == "exemplo1")
	{
		rotateMatrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), -_theta, 1);
		cv::warpAffine(imgThreshold, imgRotate, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
		cv::warpAffine(img, imgRotateColor, rotateMatrix, cv::Size(imgThreshold.cols, imgThreshold.rows));
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

	cv::Mat_<float> M(3, 3), N(3, 1);
	cv::Mat_<float> out;
	bool endIteration = true;
	int index_ransanc[3];
	//0.001 exemplo3 and exemplo2
	//0.1 examplo1
	float threshld = 0.001;
	int num = 0, colector = 0;
	bool a = true;
	float dist, dist1 = 0, dist2 = 0, general = 0;
	std::vector<cv::Point> _tmpPoint;
	for (int ij = 0; ij < 1000; ij++)
	{
		for (size_t i = 0; i < 3; i++)
		{
			num = std::rand() % _points.size();
			index_ransanc[i] = num;
			for (size_t j = 0; j < 3; j++)
			{
				if (j == 0)
					M(i, j) = cv::pow(_points[num].x, 2);
				if (j == 1)
					M(i, j) = _points[num].x;
				if (j == 2)
					M(i, j) = 1;
			}
			N(i, 0) = _points[num].y;
		}

		if (cv::solve(M, N, out, cv::DECOMP_LU))
		{
			//std::cout << out << std::endl;
			for (auto _point : _points)
			{
				dist = std::sqrtf(std::pow(out.at<float>(0, 0) * _point.x + out.at<float>(1, 0) * _point.x + out.at<float>(2, 0) - _point.y, 2));
				/*
				general = std::sqrtf(out.at<float>(1, 0) - 4 * out.at<float>(0, 0) * (out.at<float>(2, 0) - _point.y));
				dist1 = std::sqrtf(std::pow(_point.x - (-out.at<float>(1, 0) - general) / (2 * out.at<float>(0, 0)), 2));
				dist2 = std::sqrtf(std::pow(_point.x - (-out.at<float>(1, 0) + general) / (2 * out.at<float>(0, 0)), 2));
				if (dist1 > dist2)
				{
					dist = dist2;
				}
				else
				{
					dist = dist1;
				}
				*/
				if (threshld >= dist)
				{
					_tmpPoint.push_back(_point);
				}
			}
			if (ij == 0)
			{
				colector = _tmpPoint.size();
				_tmpPoint.clear();
			}
			else
			{
				if (colector < _tmpPoint.size())
				{
					colector = _tmpPoint.size();				
				}
			}
		}
		else
		{
			std::cout << "Cant solve this system" << std::endl;
		}
	}
	
	for (auto item : _points)
	{
		cv::circle(img, item, 1, cv::Scalar(0, 0, 255), -1, 1);
	}
	
	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/just_points_1.png", img);
	
	for (auto item : _tmpPoint) {
		cv::circle(clone, item, 1, cv::Scalar(255, 0, 0), -1, 1);
	}

	cv::imwrite("D:/VisaoComputacional/Activities/5dc1b836654be800018c3482/images/using_ransac_1.png", clone);

	cv::Mat_<float> M1(3, 3);
	M1 = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat_<float> N1(3, 1);
	N1 = cv::Mat::zeros(3, 1, CV_32F);

	M1.at<float>(0, 0) = _tmpPoint.size();
	for (int i = 0; i < _tmpPoint.size(); i++)
	{
		M1.at<float>(1, 0) += _tmpPoint[i].x;
		M1.at<float>(0, 1) += _tmpPoint[i].x;

		M1.at<float>(2, 0) += std::pow(_tmpPoint[i].x, 2);
		M1.at<float>(1, 1) += std::pow(_tmpPoint[i].x, 2);
		M1.at<float>(0, 2) += std::pow(_tmpPoint[i].x, 2);

		M1.at<float>(1, 2) += std::pow(_tmpPoint[i].x, 3);
		M1.at<float>(2, 1) += std::pow(_tmpPoint[i].x, 3);

		M1.at<float>(2, 2) += std::pow(_tmpPoint[i].x, 4);

		N1.at<float>(0, 0) += _tmpPoint[i].y;
		N1.at<float>(1, 0) += _tmpPoint[i].x * _tmpPoint[i].y;
		N1.at<float>(2, 0) += std::pow(_tmpPoint[i].x, 2) * _tmpPoint[i].y;
	}


	std::cout << M1 << std::endl;
	std::cout << N1 << std::endl;
	cv::Mat_<float> out1;
	if (cv::solve(M1, N1, out1, cv::DECOMP_LU))
	{
		std::cout << out1 << std::endl;
		std::cout << out << std::endl;
	}
	else
	{
		std::cout << "Cant solve this system" << std::endl;
	}
	/*
	for (size_t i = 0; i < 900; i++)
	{
		cv::circle(img, cv::Point(i, out.at<float>(0, 0)* std::pow(i, 2) + out.at<float>(1, 0) * i + out.at<float>(2, 0)), 1, cv::Scalar(255, 255, 0), -1, 2);
	}
	*/
	float rotation_edge_x = 0;
	float rotation_edge_y = 0;
	
	float edge = -(_theta)* (PI / 180);
	float _x1 = 0, _y1 = 0;
	for (int i = -900; i < 900; i++)
	{
		_x1 = i - img.cols / 2 ;
		_y1 = (out1.at<float>(2, 0) * std::pow(i, 2) + (double)out1.at<float>(1, 0) * i + out1.at<float>(0, 0)) - img.rows / 2 ;
		rotation_edge_x = (_x1)*std::cos(edge) - (_y1)*std::sin(edge) + img.cols / 2 ;
		rotation_edge_y = (_x1)*std::sin(edge) + (_y1)*std::cos(edge) + img.rows / 2 ;
		cv::circle(img, cv::Point(rotation_edge_x, rotation_edge_y), 1, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);		
	}

	cv::imshow("Color Image", img);
	cv::imshow("Gray Image", gray);
	cv::imshow("Threshold Image", imgThreshold);

	//cv::imshow("Rotate Image", imgRotate);
	//cv::imshow("Rotate Image Color", imgRotateColor);

	cv::waitKey(0);
}
