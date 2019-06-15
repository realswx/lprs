#ifndef LPRS_PLATEDETECTION_H
#define LPRS_PLATEDETECTION_H

#include <opencv2/opencv.hpp>
#include <PlateInfo.h>
#include <vector>

//检测车牌在图中位置

namespace pr {

    class PlateDetection {

    public:
        PlateDetection(std::string filename_cascade);
        PlateDetection();

        void LoadModel(std::string filename_cascade);
        //粗定位
        void plateDetectionRough(cv::Mat InputImage, std::vector<pr::PlateInfo>  &plateInfos,
                int min_w = 36,int max_w = 800);


    private:
        cv::CascadeClassifier cascade;


    };

}// namespace pr

#endif //LPRS_PLATEDETECTION_H
