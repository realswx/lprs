#ifndef LPRS_PLATEINFO_H
#define LPRS_PLATEINFO_H
#include <opencv2/opencv.hpp>
namespace pr {

    typedef std::vector<cv::Mat> Character;

    enum PlateColor { BLUE, YELLOW, WHITE, GREEN, BLACK, UNKNOWN}; //车牌颜色，未用
    enum CharType {CHINESE, LETTER, LETTER_NUMS, INVALID}; //车牌字符类型


    class PlateInfo {
        public:
            std::vector<std::pair<CharType, cv::Mat>> plateChars; //字符图片和类型集合
            std::vector<std::pair<CharType, cv::Mat>> plateCoding; //字符编码和类型集合
            float confidence = 0;

            //构造函数
            PlateInfo(const cv::Mat &plateData, std::string plateName,
                    cv::Rect plateRect, PlateColor plateType) { //cv::Rect plateRect 车牌矩形
                    licensePlate = plateData;
                    name = plateName;
                    ROI = plateRect;
                    Type = plateType;
            }

            PlateInfo(const cv::Mat &plateData, cv::Rect plateRect, PlateColor plateType) {
                licensePlate = plateData;
                ROI = plateRect;
                Type = plateType;
            }

            PlateInfo(const cv::Mat &plateData, cv::Rect plateRect) {
                licensePlate = plateData;
                ROI = plateRect;
            }

            PlateInfo() {

            }



            cv::Mat getPlateImage() {
                return licensePlate;
            }
            void setPlateImage(cv::Mat plateImage){
                licensePlate = plateImage;
            }
            cv::Rect getPlateRect() {
                return ROI;
            }
            void setPlateRect(cv::Rect plateRect)   {
                ROI = plateRect;
            }
            cv::String getPlateName() {
                return name;
            }
            void setPlateName(cv::String plateName) {
                name = plateName;
            }
            int getPlateType() {
                return Type;
            }

            void appendPlateChar(const std::pair<CharType,cv::Mat> &plateChar)
            {
                plateChars.push_back(plateChar);
            }

            void appendPlateCoding(const std::pair<CharType,cv::Mat> &charProb){
                plateCoding.push_back(charProb);
            }

            //车牌译码
            std::string decodePlateNormal(std::vector<std::string> mappingTable) {
                std::string decode; //定义字符,用于存放解码结果
                for(auto plate:plateCoding) { //开始循环解码
                    float *prob = (float *)plate.second.data;
                    if(plate.first == CHINESE) { //判断字符为中文
                        //从字符集合取出对应的字符
                        decode += mappingTable[std::max_element(prob, prob + 31) - prob];
                        confidence += *std::max_element(prob, prob + 31);

                    }
                    else if(plate.first == LETTER) { //判断字符为字母
                        //从字符集合取出对应的字符
                        decode += mappingTable[std::max_element(prob + 41, prob + 65)- prob];
                        confidence += *std::max_element(prob + 41, prob + 65);
                    }
                    else if(plate.first == LETTER_NUMS) { //判断字母数字
                        //从字符集合取出对应的字符
                        decode += mappingTable[std::max_element(prob+31,prob+65)- prob];
                        confidence += *std::max_element(prob + 31, prob + 65);
                    }
                    else if(plate.first == INVALID)
                    {
                        decode+='*';
                    }

                }
                name = decode;

                confidence/=7;

                return decode;
            }

    private:
        cv::Mat licensePlate; //需要识别的图像
        cv::Rect ROI; //图像矩形区域
        std::string name ;
        PlateColor Type; //未用
    };
}


#endif //LPRS_PLATEINFO_H
