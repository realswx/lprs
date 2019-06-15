#include "../include/PlateDetection.h"
#include "util.h"

//探测车牌在图中何处位置

namespace pr {

    PlateDetection::PlateDetection(std::string filename_cascade) {
        cascade.load(filename_cascade);

    };

    //车牌粗略探测位置
    void PlateDetection::plateDetectionRough(cv::Mat InputImage,
            std::vector<pr::PlateInfo>  &plateInfos,int min_w,int max_w) {
        cv::Mat processImage;
        cv::cvtColor(InputImage,processImage,cv::COLOR_BGR2GRAY); //将图片二值化
        std::vector<cv::Rect> platesRegions; //检测物体的矩形框向量组
        cv::Size minSize(min_w,min_w/4); //最小宽度
        cv::Size maxSize(max_w,max_w/4); //最大宽度
        //级联分类器 https://blog.csdn.net/weixin_42309501/article/details/80781293
        //进行扫描
        cascade.detectMultiScale( processImage, platesRegions,
                                  1.1, 3, cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
        //遍历扫描结果
        for(auto plate:platesRegions) {
            /*
               extend rects
               x -= w * 0.14
               w += w * 0.28
               y -= h * 0.6
               h += h * 1.1;
            */
            int zeroadd_w  = static_cast<int>(plate.width*0.30);
            int zeroadd_h = static_cast<int>(plate.height*2);
            int zeroadd_x = static_cast<int>(plate.width*0.15);
            int zeroadd_y = static_cast<int>(plate.height*1);
            plate.x-=zeroadd_x;
            plate.y-=zeroadd_y;
            plate.height += zeroadd_h;
            plate.width += zeroadd_w;
            //复制扫描的Image,复制给plateImage
            cv::Mat plateImage = util::cropFromImage(InputImage,plate);
            PlateInfo plateInfo(plateImage,plate); //创建车牌信息对象
            plateInfos.push_back(plateInfo);

        }
    }
}//namespace pr
