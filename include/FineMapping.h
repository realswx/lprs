
#ifndef LPRS_FINEMAPPING_H
#define LPRS_FINEMAPPING_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <string>


namespace pr {

    class FineMapping {
    public:

        //构造函数
        FineMapping();

        FineMapping(std::string prototxt,std::string caffemodel);


        //垂直方向
        static cv::Mat FineMappingVertical(cv::Mat InputProposal, int sliceNum = 15, int upper = 0,
                int lower = -50, int windows_size = 17);
        //水平方向
        cv::Mat FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding);


    private:
        cv::dnn::Net net;

    };




}
#endif //LPRS_FINEMAPPING_H
