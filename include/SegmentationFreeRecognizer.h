#ifndef LPRS_SEGMENTATIONFREERECOGNIZER_H
#define LPRS_SEGMENTATIONFREERECOGNIZER_H

#include "Recognizer.h"
namespace pr{


    class SegmentationFreeRecognizer{
    public:
        const int CHAR_INPUT_W = 14;
        const int CHAR_INPUT_H = 30;
        const int CHAR_LEN = 84;

        SegmentationFreeRecognizer(std::string prototxt, std::string caffemodel);
        std::pair<std::string,float> SegmentationFreeForSinglePlate(cv::Mat plate, std::vector<std::string> mapping_table);


    private:
        cv::dnn::Net net;

    };

}
#endif //LPRS_SEGMENTATIONFREERECOGNIZER_H
