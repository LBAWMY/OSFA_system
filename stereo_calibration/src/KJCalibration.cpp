
#include "KJCalibration.h"
#include <experimental/filesystem>
#include <algorithm>


KJCalibration::KJCalibration(int boardWidth, int boardHeight, int squareSize)
    : squareSize(squareSize)
{    
    positionNums = 9;
    cornerNums = boardWidth * boardHeight;

    board_size = cv::Size(boardWidth, boardHeight);

    clear();

    // create result directory


}

KJCalibration::~KJCalibration()
{
}

bool KJCalibration::setOnePosition(cv::Mat &leftImage, cv::Mat &rightImage)
{
    int k = camL_corners_all.size();
    if (k >= positionNums) {
        cout << "positions are enough. Please call calibrate() now." << endl;
        return false;
    }
    

    // check corners
    if (false == fastCheckCorners(leftImage) || false == fastCheckCorners(rightImage)) {
        cout << "leftImage or rightImage has no corners" << endl;
        return false;
    }
    camera_size = cv::Size(leftImage.cols, leftImage.rows);
    

    // left cam_corner
    extractCameraCorners(leftImage, camL_points);

    // right cam_corner
    extractCameraCorners(rightImage, camR_points);

    // push_back data
    camL_corners_all.push_back(camL_points);
    camR_corners_all.push_back(camR_points);

    return true;
}

bool KJCalibration::calibrate()
{
    if (camL_corners_all.size() < positionNums) {
        cout << "positions are not enough." <<endl;
        return false;
    }

    cout << "Start YML build!" <<endl;
    
    std::vector<std::vector<cv::Point3f> > objectPointsAll;
    std::vector<cv::Point3f> object_points;
    calcObjectPoints(objectPointsAll);

    cv::Mat cameraMatrix_L =    cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Cd_L           =    cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
    cv::Mat cameraMatrix_R =    cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Cd_R           =    cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));

    std::vector<cv::Mat> Rc_L, Tc_L,  Rc_R, Tc_R;
    cv::Mat R, T, E, F;
	//cameraMatrix_L/R intrinsic parameter matrix, Cd_L/R: distortion matrix, 
    double camL_error    = cv::calibrateCamera(objectPointsAll, camL_corners_all, camera_size, cameraMatrix_L, Cd_L, Rc_L, Tc_L, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, DBL_EPSILON));
    double camR_error    = cv::calibrateCamera(objectPointsAll, camR_corners_all, camera_size, cameraMatrix_R, Cd_R, Rc_R, Tc_R, 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, DBL_EPSILON));
    //R & T: rotation matrix & translation vector of right camera with respect to left camera
	double cam_two_error = cv::stereoCalibrate(objectPointsAll, camL_corners_all, camR_corners_all, cameraMatrix_L, Cd_L, cameraMatrix_R, Cd_R, camera_size, R, T, E, F, CV_CALIB_USE_INTRINSIC_GUESS, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 150, DBL_EPSILON));

    cout << "stereoL_error : "<< cam_two_error <<endl;

    // save_calibration_yml
    cv::FileStorage fs("/home/trs/catkin_ws/src/stereo_calibration/data/cameraParameter.yml", cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        cout << "fs.isOpened() failed." <<endl;
    }
    else {
        fs << "cameraMatrixL" << cameraMatrix_L << "cameraDistcoeffL" << Cd_L
            << "cameraMatrixR" << cameraMatrix_R << "cameraDistcoeffR" << Cd_R
            << "R" << R << "T" << T << "E" << E << "F" << F << "camL_error" << camL_error << "camR_error" << camR_error << "stereo_error" << cam_two_error;
        fs.release();
    }

    cout << "cameraMatrixL" << cameraMatrix_L <<"\n";
    cout << "cameraDistcoeffL" << Cd_L<<"\n";
    cout << "cameraMatrixR" << cameraMatrix_R<<"\n";
    cout << "cameraDistcoeffR" << Cd_R<<"\n";
    cout << "R" << R <<"\n";
    cout << "T" << T <<"\n";
    cout << "Cam_Two_Error : " << cam_two_error <<endl;
    cout << "YML save Complete!" <<endl;

    return true;
}

void KJCalibration::clear()
{
    camL_corners_all.clear();
    camR_corners_all.clear();
}

bool KJCalibration::fastCheckCorners(cv::Mat &image)
{
    std::vector<cv::Point2f> corners;
    return cv::findChessboardCorners(image, board_size, corners, cv::CALIB_CB_FAST_CHECK);
}

bool KJCalibration::checkAndDrawCorners(cv::Mat &image)
{
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(image, board_size, corners, cv::CALIB_CB_FAST_CHECK);
    if (found) { drawChessboardCorners(image, board_size, corners, found); }

    return found;
}

void KJCalibration::setGreenArea(cv::Mat &cornerImage, cv::Mat &greenArea)
{
    std::vector<cv::Point2f> corners;
    cv::findChessboardCorners(cornerImage, board_size, corners, cv::CALIB_CB_FAST_CHECK);
    std::vector<cv::Point> leftPoints;
    const int index[4] = { 0, board_size.width - 1, board_size.width*board_size.height - 1, board_size.width*board_size.height - board_size.width };
    for (int i = 0; i < 4; i++) {
        leftPoints.push_back(corners[index[i]]);
    }

    // fill poly
    const int greenChannelValue = 80;
    const int numOfPoints = 4;
    cv::Point pointsLeft[1][numOfPoints];

    for (int i = 0; i < numOfPoints; i++) {
        pointsLeft[0][i] = leftPoints[i];
    }

    int npt[1] = { numOfPoints };
    const cv::Point *pptLeft[1] = { pointsLeft[0] };

    fillPoly(greenArea, pptLeft, npt, 1, cv::Scalar(0, greenChannelValue, 0));
}

void KJCalibration::setBoardWidth(int boardWidth)
{
    board_size = cv::Size(boardWidth, board_size.height);
}

void KJCalibration::setBoardHeight(int boardHeight)
{
    board_size = cv::Size(board_size.width, boardHeight);
}

void KJCalibration::setBoardSquareSize(int squareSize)
{
    this->squareSize = squareSize;
}


void KJCalibration::setPositionNums(int positionNums)
{
    const int MINIMUM_POSITION_NUMS = 9;
    if (positionNums < MINIMUM_POSITION_NUMS) 
    {
        cout << "positionNums is less than MINIMUM_POSITION_NUMS" << endl;
        return; 
    }

    this->positionNums = positionNums;
}




void KJCalibration::extractCameraCorners(cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    cv::findChessboardCorners(image, board_size, corners);
    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
}


void KJCalibration::calcObjectPoints(std::vector<std::vector<cv::Point3f> > &objectPoints)
{
    std::vector<cv::Point3f> object_xyz;
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_xyz.push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0));
        }
    }
    for (int i = 0; i < positionNums; ++i) {
        objectPoints.push_back(object_xyz);
    }
}

