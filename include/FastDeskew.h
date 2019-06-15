
#ifndef LPRS_FASTDESKEW_H
#define LPRS_FASTDESKEW_H

#include <math.h>
#include <opencv2/opencv.hpp>

//旋转歪斜的车牌
namespace pr {

    //声明
    cv::Mat fastdeskew(cv::Mat skewImage,int blockSize);

}//namepace pr


#endif //LPRS_FASTDESKEW_H
