
#include "../include/CNNRecognizer.h"


//加载模型预测标签
namespace pr {

    //读取caffe的训练模型
    CNNRecognizer::CNNRecognizer(std::string prototxt, std::string caffemodel){
        net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    }

    label CNNRecognizer::recognizeCharacter(cv::Mat charImage){
        if(charImage.channels()== 3) //如果是rgb图像
            cv::cvtColor(charImage, charImage, cv::COLOR_BGR2GRAY); //将图片转化为黑白图
        /*
         *归一化 归一到０－１ ： 从一系列图像创建4维斑点。
         * 可选地调整大小和作物图像从中心，减去平均值，尺度值按比例因子，交换蓝色和红色通道。
        */
        cv::Mat inputBlob = cv::dnn::blobFromImage(charImage, 1/255.0, cv::Size(CHAR_INPUT_W,CHAR_INPUT_H), cv::Scalar(0,0,0),false);
        net.setInput(inputBlob,"data"); //设置数据
        return net.forward(); //计算输出
    }
}
