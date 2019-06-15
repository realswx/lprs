
#ifndef LPRS_CNNRECOGNIZER_H
#define LPRS_CNNRECOGNIZER_H

#include "Recognizer.h"

namespace pr {
    class CNNRecognizer: public GeneralRecognizer{
    public:
        const int CHAR_INPUT_W = 14; //字符输出宽度
        const int CHAR_INPUT_H = 30; //字符输出高度

        CNNRecognizer(std::string prototxt,std::string caffemodel);
        label recognizeCharacter(cv::Mat character);
    private:
        cv::dnn::Net net;

    };

}

#endif //LPRS_CNNRECOGNIZER_H
