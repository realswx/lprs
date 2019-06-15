
#include <../include/FastDeskew.h>


//快速旋转歪斜的车牌
namespace pr {

    const int ANGLE_MIN = 30 ; //最小旋转角度
    const int ANGLE_MAX = 150 ; //最大旋转角度
    const int PLATE_H = 36; //车牌高
    const int PLATE_W = 136; //车牌宽
    
    int angle(float x,float y) {
        //计算角度
        return atan2(x,y)*180/3.1415;
    }

    /*
	-----------计算一组角度的平均角度(类似平滑卷积)------------
	@angle_list  : 一组角度值;
	@windowsSize : 平滑计算时窗口大小;
	*/
    std::vector<float> avgfilter(std::vector<float> angle_list, int windowsSize) {
        //定义vector angle_list_filtered
        std::vector<float> angle_list_filtered(angle_list.size() - windowsSize + 1);
        //循环计算每个
        for (int i = 0; i < angle_list.size() - windowsSize + 1; i++) {
            float avg = 0.00f; //清空当前angle的均值
            for (int j = 0; j < windowsSize; j++) { //累计计算总的angle
                avg += angle_list[i + j];
            }
            avg = avg / windowsSize; //求平均
            angle_list_filtered[i] = avg; //对应平均值写入angle_list_filtered
        }

        return angle_list_filtered;
    }


    /*
	------------绘制直方图---------------
	@seq 一组数据,需要画到直方图上
	*/
    void drawHist(std::vector<float> seq) {
        cv::Mat image(300,seq.size(),CV_8U); ////直方图声明
        image.setTo(0); ////直方图清零

        //循环,将seq中每个数值画上去
        for(int i = 0;i<seq.size();i++) {
            //先画seq中最大的值
            float l = *std::max_element(seq.begin(),seq.end());
            //高度挤压在0~300
            int p = int(float(seq[i])/l*300);
            //画
            cv::line(image,cv::Point(i,300),cv::Point(i,300-p),cv::Scalar(255,255,255));
        }
        //显示出来
        cv::imshow("vis",image);
    }

    /*
	------------校正车牌图像---------------
	@skewPlate : 需要校正的图像;
	@angle     : 需要旋转的角度;
	@maxAngle  : 最大旋转的角度;
	*/
    cv::Mat  correctPlateImage(cv::Mat skewPlate,float angle,float maxAngle) {
        cv::Mat dst; //声明旋转后的图像
        cv::Size size_o(skewPlate.cols,skewPlate.rows); //获取待处理图像的尺寸size_o
        int extend_padding = 0; //延长填充变量声明
        //计算延长填充大小
        extend_padding = static_cast<int>(skewPlate.rows*tan(cv::abs(angle)/180* 3.14) );
        //计算延展后的图像尺寸size
        cv::Size size(skewPlate.cols + extend_padding ,skewPlate.rows);
        //计算旋转后的宽
        float interval = abs(sin((angle /180) * 3.14)* skewPlate.rows);
        //原图像构成的矩形的四个顶点坐标
        cv::Point2f pts1[4] = {cv::Point2f(0,0), cv::Point2f(0,size_o.height), cv::Point2f(size_o.width,0), cv::Point2f(size_o.width,size_o.height)};

        if(angle>0) { //则逆时针旋转
            //新图像构成的矩形的四个顶点坐标
            cv::Point2f pts2[4] = {cv::Point2f(interval, 0), cv::Point2f(0, size_o.height),
                                   cv::Point2f(size_o.width, 0), cv::Point2f(size_o.width - interval, size_o.height)};
            cv::Mat M  = cv::getPerspectiveTransform(pts1,pts2); //计算变换矩阵
            cv::warpPerspective(skewPlate, dst, M, size); //将skewPlate经M变换为大小为size的图像dst
        }
        else { //顺时针旋转
            //新图像构成的矩形的四个顶点坐标
            cv::Point2f pts2[4] = {cv::Point2f(0, 0), cv::Point2f(interval, size_o.height), cv::Point2f(size_o.width-interval, 0),
                                   cv::Point2f(size_o.width, size_o.height)};
            cv::Mat M  = cv::getPerspectiveTransform(pts1,pts2); //计算变换矩阵
            cv::warpPerspective(skewPlate,dst,M,size,cv::INTER_CUBIC); //将skewPlate经M变换为大小为size的图像dst
        }
        return  dst;
    }

    /*
	------------快速旋转车牌图像---------------
	@skewPlate : 需要校正的图像;
	@blockSize : 角点检测步长;
	*/
    cv::Mat fastdeskew(cv::Mat skewImage,int blockSize) {

        const int FILTER_WINDOWS_SIZE = 5; //过滤的窗口大小
        std::vector<float> angle_list(180); //声明一个angle_list存储角度
        memset(angle_list.data(),0,angle_list.size()*sizeof(int)); //为其分配内存
        cv::Mat bak; //用于备份原图像的图像bak
        skewImage.copyTo(bak); //将原图赋值给bak进行备份
        if(skewImage.channels() == 3) //如果是rgb图像
            cv::cvtColor(skewImage,skewImage,cv::COLOR_RGB2GRAY); //先转化为黑白图
        if(skewImage.channels() == 1) { //若是黑白图
            cv::Mat eigen; //声明特征矩阵eigen
            cv::cornerEigenValsAndVecs(skewImage,eigen,blockSize,5); //计算图像块的特征值和特征向量,用于角点检测,结果保存在eigen
            for(int j = 0; j < skewImage.rows; j += blockSize) { //遍历skewImage的每个像素
                for(int i = 0; i < skewImage.cols; i += blockSize) {
                    //(x2,y2)存储skewImage存在eigen的角点信息
                    float x2 = eigen.at<cv::Vec6f>(j, i)[4];
                    float y2 = eigen.at<cv::Vec6f>(j, i)[5];
                    int angle_cell = angle(x2,y2); //计算角度
                    angle_list[(angle_cell + 180)%180]+=1.0; //在对应角度上作累计计数
                }
            }
        }
        //计算平滑窗口大小为5的平均角度过滤
        std::vector<float> filtered = avgfilter(angle_list,5);
        //计算均角过滤的最大位置
        int maxPos = std::max_element(filtered.begin(),filtered.end()) - filtered.begin() + FILTER_WINDOWS_SIZE/2;
        if(maxPos > ANGLE_MAX) //超过了ANGLE_MAX，即150
            maxPos = (-maxPos+90+180)%180; //作角度变换
        if(maxPos<ANGLE_MIN) //未超过ANGLE_MAX
            maxPos -= 90; //控制在90内
        maxPos = 90 - maxPos; //再变换
        //按maxPos校正图像
        cv::Mat deskewed = correctPlateImage(bak, static_cast<float>(maxPos),60.0f);
        return deskewed;
    }



}//namespace pr
