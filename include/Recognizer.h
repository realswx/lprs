#ifndef LPRS_RECOGNIZER_H
#define LPRS_RECOGNIZER_H

#include "PlateInfo.h"
#include "opencv2/dnn.hpp"


namespace pr {
    typedef cv::Mat label;

    class GeneralRecognizer {
    public:
        virtual label recognizeCharacter(cv::Mat character) = 0;

        void SegmentBasedSequenceRecognition(PlateInfo &plateinfo);
        void SegmentationFreeSequenceRecognition(PlateInfo &plateInfo);

    };

}
#endif //LPRS_RECOGNIZER_H
