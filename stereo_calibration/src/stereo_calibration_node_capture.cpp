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
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
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
#include <stdlib.h>
using namespace cv;
using namespace std;

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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_tracker");
    ros::NodeHandle n;
    image_transport::ImageTransport it(n);
    //subscribe video from 2 camera
    image_transport::Subscriber sub_img_1_ = it.subscribe("/camera1/usb_cam1/image_raw", 1, camera1_Callback);
    image_transport::Subscriber sub_img_2_ = it.subscribe("/camera2/usb_cam2/image_raw", 1, camera2_Callback);
    ros::Rate loop_rate(50);

    //duration for waiting for cameras
    ros::Duration(1).sleep();
    //read camera once
    ros::spinOnce();
    Mat temp;
    temp = frame1.clone();

	Mat frame_l,frame_r;
	int img_counter = 0;

    string file_path_l = "/home/trs/bhyang/stereo_cali_img/cam_l/";
    string file_path_r = "/home/trs/bhyang/stereo_cali_img/cam_r/";
    string file_name;
    //string name_l,name_r;
    char* name_l = (char*)malloc(sizeof(char)*200);
    char* name_r = (char*)malloc(sizeof(char)*200);

	//VideoCapture capture(0);
	while (ros::ok())
	{
        loop_rate.sleep();
        ros::spinOnce();

        frame_l = frame1.clone();
        frame_r = frame2.clone();

		//ResizeShow(frame_l, "read_video_l");
		//ResizeShow(frame_r, "read_video_r");
		imshow("read_video_l", frame_l);
		imshow("read_video_r", frame_r);
		
		int key = waitKey(50); // for button press time
		if ((char)key == 27) //ESC button for quit
		{
			cout << "u press ESC" << "\n";
			break;
		}
		if ((char)key == 32) // Space button for capture one Image
		{
            img_counter++;
			cout << "u press space" << img_counter << "\n";
			//write img to file
            sprintf(name_l, "leftPic%d.jpg", img_counter);
            sprintf(name_r, "rightPic%d.jpg", img_counter);
            //itoa(img_counter,imgID);
            file_name = file_path_l + name_l;
            imwrite( file_name, frame_l );
            file_name = file_path_r + name_r;
            imwrite( file_name, frame_r );
        }

		//waitKey(10);
		
	}
	//////////////////////////////////////////////
	cout << "Success!" << endl;
	system("pause");
	return 0;
}


