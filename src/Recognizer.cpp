#include "../include/Recognizer.h"


//挨个识别字符并返回识别结果

namespace pr {


    void GeneralRecognizer::SegmentBasedSequenceRecognition(PlateInfo &plateinfo){
        //循环车牌字符数据
        for(auto char_instance:plateinfo.plateChars) {
            //存放识别出来的字符
            std::pair<CharType,cv::Mat> res;

            if(char_instance.second.rows*char_instance.second.cols > 40) {
                //识别字符数据
                label code_table = recognizeCharacter(char_instance.second);
                //字符类型
                res.first = char_instance.first;
                //识别出来的字符数据 复制到res
                code_table.copyTo(res.second);
                //添加数据
                plateinfo.appendPlateCoding(res);
            } else {
                res.first = INVALID;
                plateinfo.appendPlateCoding(res);
            }
        }
    }
}
