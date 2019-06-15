
#include "../include/Pipeline.h"

/*
 * 串联程序,串联起了plateDetection、fineMapping、
 * plateSegmentation、recognizer、segmentationFreeRecognizer
 */
namespace pr {

    //定义水平方向填充参数
    const int HorizontalPadding = 4;

    /*
    -----------串接多个功能(构造函数)------------

   @detector_filename            :                         识别模型所在路径;
   @finemapping_prototxt         :   抠出车牌图像模型的prototxt定义文件路径;
   @finemapping_caffemodel       :           抠出车牌图像模型的模型文件路径;
   @segmentation_prototxt        : 分割图像字符的模型的prototxt定义文件路径;
   @segmentation_caffemodel      :         分割图像字符的模型的模型文件路径;
   @charRecognization_proto      :     字符识别的模型的prototxt定义文件路径;
   @charRecognization_caffemodel :             字符识别的模型的模型文件路径;
   @segmentationfree_proto       : 识别单个车牌的模型的prototxt定义文件路径;
   @segmentationfree_caffemodel  :         识别单个车牌的模型的模型文件路径;
   */
    PipelinePR::PipelinePR(std::string detector_filename,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string segmentation_prototxt, std::string segmentation_caffemodel,
                           std::string charRecognization_proto, std::string charRecognization_caffemodel,
                           std::string segmentationfree_proto,std::string segmentationfree_caffemodel) {

        //创建车牌位置检测模型实例指针
        plateDetection = new PlateDetection(detector_filename);
        //创建抠出车牌图像模型实例指针
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        //创建车牌分割模型实例指针
        plateSegmentation = new PlateSegmentation(segmentation_prototxt, segmentation_caffemodel);
        //创建识别模型实例指针
        generalRecognizer = new CNNRecognizer(charRecognization_proto, charRecognization_caffemodel);
        //创建识别单个车牌图像的模型实例指针
        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(segmentationfree_proto,segmentationfree_caffemodel);

    }

    PipelinePR::~PipelinePR() {

        delete plateDetection;
        delete fineMapping;
        delete plateSegmentation;
        delete generalRecognizer;
        delete segmentationFreeRecognizer;


    }



    /*
    -----------识别车牌中各个字符------------

    @plateImage :  包含一个/多个车牌图像;
    @method     : 分割方法的编码;
    */
    std::vector<PlateInfo> PipelinePR:: RunPiplineAsImage(cv::Mat plateImage,int method) {
        //声明结果列表
        std::vector<PlateInfo> results;
        //保存车牌中间信息
        std::vector<pr::PlateInfo> plates;
        //执行车牌粗略探测位置(结果存在plates内)
        plateDetection->plateDetectionRough(plateImage, plates, 36, 700);

        //迭代图中每个车牌
        for (pr::PlateInfo plateinfo:plates) {
            //获取该车牌图像(image_finemapping的finemapping是为了分割出尽量只包含单个车牌的图像)
            cv::Mat image_finemapping = plateinfo.getPlateImage();
            //对图像垂直处理
            image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
            //校正角度
            image_finemapping = pr::fastdeskew(image_finemapping, 5);


            //选择分割车牌字符的方法,选择依据？

            //方法一:基础方法
            if(method==SEGMENTATION_BASED_METHOD) {
                //对图像水平处理
                image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, HorizontalPadding);
                //大小调整
                cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
                //展示
                //cv::imshow("image_finemapping",image_finemapping);
                //cv::waitKey(0);
                //设定为调整后的图像
                plateinfo.setPlateImage(image_finemapping);
                //定义矩形框列表
                std::vector<cv::Rect> rects;
                //对车牌图像的字符分割,结果存在rects
                plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
                //将每个rect中的字符子图又存到plateinfo中去
                plateSegmentation->ExtractRegions(plateinfo, rects);
                //复制图像并且制作边界;处理边界卷积(将image_finemapping的黑色边界填充)
                cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 20, cv::BORDER_REPLICATE);
                //录入plateinfo
                plateinfo.setPlateImage(image_finemapping);
                //进行识别
                generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
                //解码中文字符
                plateinfo.decodePlateNormal(pr::CH_PLATE_CODE);

            }
                //方法二：Segmentation-free
            else if(method==SEGMENTATION_FREE_METHOD) {
                //对图像水平处理
                image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding+3);
                //大小调整
                cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
                //存储图像
                //cv::imwrite("./test.png",image_finemapping);
                // 显示图像
                // cv::imshow("image_finemapping",image_finemapping);
                // cv::waitKey(0);
                // 录入plateinfo
                plateinfo.setPlateImage(image_finemapping);
                //定义矩形框列表
                //std::vector<cv::Rect> rects;
                //对单个图像进行识别
                std::pair<std::string,float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(),pr::CH_PLATE_CODE);
                //获取置信度
                plateinfo.confidence = res.second;
                //车牌识别字符结果
                plateinfo.setPlateName(res.first);
            }
            //结果加入列表
            results.push_back(plateinfo);
        }

        //遍历识别结果
        //for (auto str:results) {
        //输出;
        //std::cout << str << std::endl;
        //}


        //返回结果
        return results;

    }//namespace pr



}
