
#pragma once

#include <iostream> 
#include <opencv2/opencv.hpp> 
#include <thread>
#include <chrono>
using namespace std;
using namespace cv;

class KJCalibration
{
public:
    KJCalibration(int boardWidth, int boardHeight, int squareSize);
    ~KJCalibration();
    typedef std::shared_ptr<KJCalibration> Ptr;
    //using callbackLog = std::function<void(std::string)>;

    /**
    * @brief       set images of one position
    * @param       leftImages: images of left camera. rightImages: images of right camera.
    * @return      false: set failed. true: images are ok.
    */
    bool setOnePosition(Mat &leftImage, Mat &rightImage);

    /**
    * @brief       calibrate the system
    * @return      false: failed. true: success and calibration result are generated.
    * @attention   This function should be called after setOnePosition() and position number is enough.
    */
    bool calibrate();

    /**
    * @brief       clear positions that set before
    */
    void clear();

    /**
    * @brief       check whether the image has corners
    * @param       image to check
    * @return      true: image has corners; false: image have no corners
    * @attention   image should be the gray format
    */
    bool fastCheckCorners(cv::Mat &image);
    bool checkAndDrawCorners(cv::Mat &image);
    void setGreenArea(cv::Mat &cornerImage, cv::Mat &greenArea);

    /**
    * @brief       change chessboard size.
    */
    void setBoardWidth(int boardWidth);
    void setBoardHeight(int boardHeight);
    void setBoardSquareSize(int squareSize);


    /**
    * @brief       set position number. default is 9.
    * @attention   image number should be more than 9
    */
    void setPositionNums(int positionNums);
    

private:
    KJCalibration() = delete;
    KJCalibration(const KJCalibration&) = delete;

    void extractCameraCorners(cv::Mat &image, std::vector<cv::Point2f> &corners);

    void calcObjectPoints(std::vector<std::vector<cv::Point3f> > &objectPoints);



    int                     positionNums;
    int                     cornerNums;
    int                     squareSize;
    cv::Size                board_size;
    cv::Size                camera_size;

    std::string     RESULT_DIRECTORY_NAME   = "calibration_result\\";
    std::string     result_directory_path;
    std::string     result_yml                = "result.yml";


    std::vector<cv::Point2f>                camL_points;
    std::vector<cv::Point2f>                camR_points;
    std::vector<std::vector<cv::Point2f> >  camL_corners_all;
    std::vector<std::vector<cv::Point2f> >  camR_corners_all;

};

