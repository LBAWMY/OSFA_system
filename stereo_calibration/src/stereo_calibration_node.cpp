 //input output stream library
#include <iostream>
// string stream library
#include <sstream>
//ros library
#include <ros/ros.h>
// standard message
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
// point cloud
#include <sensor_msgs/PointCloud2.h>
//#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types.h>
//opencv library
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
// image transport
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
//others
#include <vector>
#include <string>
#include <algorithm>
#include "KJCalibration.h"
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;
//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

void ResizeShow(Mat img1, string WindowName) // Showimage in suitble size
{
	Mat img2;
	int w1, w2, h1, h2;
	h1 = img1.rows;
	w1 = img1.cols;
//	h2 = 640;
    h2 = 672;
	w2 = (w1*h2) / h1;
	resize(img1, img2, Size(w2, h2), 0, 0, CV_INTER_LINEAR);
	//namedWindow(WindowName);
	imshow(WindowName, img2);
}

cv::Mat frame1, frame2;
bool camera1_flag = false;
bool camera2_flag = false;
void camera1_Callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr; // 声明一个CvImage指针的实例
    //cout<<"call camera 1 \n";

    try
    {
        cv_ptr =  cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); //将ROS消息中的图象信息提取，生成新cv类型的图象，复制给CvImage指针
        camera1_flag = true;
        frame1 = cv_ptr->image;
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        cout<<"error reading camera1 /n";
        return;
    }

}

void camera2_Callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr; // 声明一个CvImage指针的实例

    try
    {
        cv_ptr =  cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); //将ROS消息中的图象信息提取，生成新cv类型的图象，复制给CvImage指针
        camera2_flag = true;
        frame2 = cv_ptr->image;
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        cout<<"error reading camera2 /n";
        return;
    }

}

 //Rectify prepare
 int flag_rectify = 0;
 int flag_corner_depth = 0;
 cv::Mat MX1, MY1, MX2, MY2;
 cv::Mat M1, M2, D1, D2, T, R;
 cv::Mat Rl, Rr, Pl, Pr;
 cv::Mat QP;
 Mat view_diff(4, 1, CV_64FC1),world(4, 1, CV_64FC1);
 //Camera Initia
// int camera_W=640,camera_H=480;
 int camera_W=672,camera_H=512;
 double fx_d,px_d,py_d,px_d_2,baseline,offset;    //from rectification
 float constant = 1.0f / fx_d;
 string calib_message;
 float w_x, w_y, w_z;
 cv::Mat frame_LN,frame_RN;
 std::vector<Point2f> corners_L,corners_R;

 void Calculate_world_coord(double x, double y, double dif)
 {
     double X, Y, Z, W;
     X = x - px_d;
     Y = y - py_d;
     Z = fx_d;
     W = (dif - offset) / baseline;
     w_x = X / W;
     w_y = Y / W;
     w_z = Z / W;
 }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_tracker");
    ros::NodeHandle n;
    image_transport::ImageTransport it(n);
    //subscribe video from 2 camera
    image_transport::Subscriber sub_img_1_ = it.subscribe("/camera1/usb_cam1/image_raw", 1, camera1_Callback);
    image_transport::Subscriber sub_img_2_ = it.subscribe("/camera2/usb_cam2/image_raw", 1, camera2_Callback);
//    image_transport::Subscriber sub_img_1_ = it.subscribe("smarteye/image_left", 1, camera1_Callback);
//    image_transport::Subscriber sub_img_2_ = it.subscribe("smarteye/image_right", 1, camera2_Callback);
    ros::Rate loop_rate(50);

    //duration for waiting for cameras
    ros::Duration(1).sleep();
    //read camera once
    ros::spinOnce();
    Mat temp;
    temp = frame1.clone();

	Mat frame_l,frame_r;
	Mat frame_l3, frame_r3;
	Mat greenAreaLeft, greenAreaRight;
	Mat greenAreaLeft_test, greenAreaRight_test;
	Mat L_see, R_see;

	greenAreaLeft_test = temp.clone();
	greenAreaLeft_test.setTo(Scalar(0, 0, 0));
	greenAreaRight_test = temp.clone();
	greenAreaRight_test.setTo(Scalar(0, 0, 0));
	greenAreaLeft.create(temp.rows, temp.cols, CV_8UC3);
	greenAreaLeft.setTo(Scalar(0, 0, 0));
	greenAreaRight.create(temp.rows, temp.cols, CV_8UC3);
	greenAreaRight.setTo(Scalar(0, 0, 0));

	//calibration
	//Prepare
	vector<cv::Point2f> corners;
	Size board_size = Size(11, 8); // checkboard corner size (inside number)
    //Parameter Initial
	bool isSuccess;
	int POS_NUM = 0;
	

	KJCalibration KJcalib(11, 8, 5); // calib checkboard - corner number and size
	KJcalib.setPositionNums(15); // calib image number


	

	//VideoCapture capture(0);
	while (ros::ok())
	{
        loop_rate.sleep();
        ros::spinOnce();

        frame_l = frame1.clone();
        frame_r = frame2.clone();
        frame_LN = frame1.clone();
        frame_RN = frame2.clone();

		//ResizeShow(frame_l, "read_video_l");
		//ResizeShow(frame_r, "read_video_r");
		/*imshow("read_video_l", frame_l);
		imshow("read_video_r", frame_r);*/

		frame_l3 = frame_l.clone();
		frame_r3 = frame_r.clone();

		cvtColor(frame_l, frame_l, CV_BGR2GRAY);
		cvtColor(frame_r, frame_r, CV_BGR2GRAY);//after this, frame_l/r is 1 channel, frame_l/r3 is 3 channals.

		// draw corners for chessboard
		bool found_l = findChessboardCorners(frame_l3, board_size, corners, CALIB_CB_FAST_CHECK);//input 3 
		if (found_l)
		{
			drawChessboardCorners(frame_l3, board_size, corners, found_l);//input 3 
		}
		else
		{
			//cout << "not found in left img!" << endl;
		}
		bool found_r = findChessboardCorners(frame_r3, board_size, corners, CALIB_CB_FAST_CHECK);//input 3 
		if (found_r)
		{
			drawChessboardCorners(frame_r3, board_size, corners, found_r);//input 3 
		}
		else
		{
			//cout << "not found in right img!" << endl;
		}
		
		L_see = frame_l3 + greenAreaLeft; //both 3 (frame_l/r3, L/R_see, greenAreaLeft/Right)
		R_see = frame_r3 + greenAreaRight;
		
		int key = waitKey(50); // for button press time
		if ((char)key == 27) //ESC button for quit
		{
			cout << "u press ESC" << "\n";
			break;
		}
		if ((char)key == 32) // Space button for capture one Image
		{
			cout << "u press space" << "\n";
			isSuccess = KJcalib.setOnePosition(frame_l, frame_r);//input 1(frame_l/r)
			if (true == isSuccess)
			{
				KJcalib.setGreenArea(frame_l, greenAreaLeft); // input(frame_l/r) 1,3(greenAreaLeft/Right)t
				KJcalib.setGreenArea(frame_r, greenAreaRight);
				POS_NUM++;
				cout << "Positions obtained :  " << POS_NUM << "/15" << endl;
			}
			else
			{
				cout << "Invalid, Positions obtained :  " << POS_NUM << "/15" << endl;
			}
		}
		if ((char)key == 'a') // 'a' button for calibrate 97
		{
			cout << "u press a" << "\n";
			KJcalib.calibrate();
		}
        if (key == 113)// 'q' button for stereorectify
        {
            cv::FileStorage fs;
            if (fs.open("/home/trs/catkin_ws/src/stereo_calibration/data/cameraParameter.yml", cv::FileStorage::READ))
            {
                fs["cameraMatrixL"] >> M1;
                fs["cameraDistcoeffL"] >> D1;
                fs["cameraMatrixR"] >> M2;
                fs["cameraDistcoeffR"] >> D2;
                fs["R"] >> R;
                fs["T"] >> T;
                fs.release();
                stereoRectify(M1, D1, M2, D2, Size(camera_W, camera_H), R, T, Rl, Rr, Pl, Pr, QP, 0);
                flag_rectify = 1;
                fx_d = QP.at<double>(2, 3);  // f
                px_d = -QP.at<double>(0, 3); // u0
                py_d = -QP.at<double>(1, 3); // v0
                baseline = 1 / QP.at<double>(3, 2); // -Tx
                offset = - (QP.at<double>(3, 3) / QP.at<double>(3, 2)); //u0-u0'
                cout << QP << endl;
                cout << fx_d << endl;
                cout << px_d << endl;
                cout << py_d << endl;
                cout << baseline << endl;
                cout << offset << endl;
            }
            else
            {
                calib_message = "Open File error!";
            }
        }

        if (key == 119)// 'w' button for corner depth caliculate
        {
            if (flag_rectify > 0)
            {
                flag_corner_depth = 1;
            }
            else
            {
                calib_message = "Please stereorectify first !";
            }
        }


        if (flag_rectify > 0)
        {
            initUndistortRectifyMap(M1, D1, Rl, Pl, frame_LN.size(), CV_32FC1, MX1, MY1);
            initUndistortRectifyMap(M2, D2, Rr, Pr, frame_RN.size(), CV_32FC1, MX2, MY2);
            remap(frame_LN, frame_LN, MX1, MY1, cv::INTER_LINEAR);
            remap(frame_RN, frame_RN, MX2, MY2, cv::INTER_LINEAR);
//            cvtColor(frame_LN, frame_LN, CV_GRAY2BGR);
//            cvtColor(frame_RN, frame_RN, CV_GRAY2BGR);
            L_see = frame_LN.clone();
            R_see = frame_RN.clone();
            L_see.convertTo(L_see, CV_8UC3);
            R_see.convertTo(R_see, CV_8UC3);
            calib_message = "Stereorectify ing!";

            if (flag_corner_depth > 0)
            {
                bool found_L = findChessboardCorners(L_see, board_size, corners_L, CALIB_CB_FAST_CHECK);
                bool found_R = findChessboardCorners(R_see, board_size, corners_R, CALIB_CB_FAST_CHECK);
                if (found_L && found_R)
                {
                    drawChessboardCorners(L_see, board_size, corners_L, found_L);
                    drawChessboardCorners(R_see, board_size, corners_R, found_R);

                    /*     cout << QP << endl;
                         cout << view_diff << endl;
                         cout << world << endl;*/

                    Calculate_world_coord(corners_L[0].x, corners_L[0].y, corners_L[0].x - corners_R[0].x);
                    float a = w_x, b = w_y, c = w_z;
                    Calculate_world_coord(corners_L[1].x, corners_L[1].y, corners_L[1].x - corners_R[1].x);
                    cout << w_x << endl;
                    cout << w_y << endl;
                    cout << w_z << endl;

                    cout << sqrt((a - w_x)*(a - w_x) + (b - w_y)*(b - w_y) + (c - w_z)*(c - w_z)) << endl;
                }
            }
        }

		namedWindow("Video_L", 1);
		ResizeShow(L_see, "Video_L");
		namedWindow("Video_R", 1);
		ResizeShow(R_see, "Video_R");

		//waitKey(10);
		
	}
	//////////////////////////////////////////////
	cout << "Success!" << endl;
	system("pause");
	return 0;
}


