#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include "CVX/core.h"
#include "CVX/vis.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <direct.h>
#include <string>
using namespace cv;

#include "BFC/stdf.h"
#include "BFC/portable.h"
using namespace ff;

#include <time.h>

#include <iostream>

namespace cv {
    namespace optflow {
        Ptr<DenseOpticalFlow> createOptFlow_DeepFlowX();
    }
}

class RegionAlignScore
{
    struct Pair
    {
        int i, j;
        char val;
    };
    std::vector<Pair> _vPairs;
    Size  _imgSize;
public:
    static void sam(int &x, int &y, const int samWSZ, const Size &imgSize, float dmin=3)
    {
        while (true)
        {
            int dx = (rand() % samWSZ - samWSZ / 2), dy = (rand() % samWSZ - samWSZ / 2);
            if (sqrt(float(dx*dx+dy*dy) < dmin))
                continue;

            int tx = x + dx;
            int ty = y + dy;
            if (uint(tx) < imgSize.width && uint(ty) < imgSize.height)
            {
                x = tx; y = ty;
                break;
            }
        }
    }
    void set(Mat ref, Mat1b mask, int d=3, int samWSZ=15, int minDiff=5)
    {
        if (ref.channels() == 3)
            cvtColor(ref, ref, CV_BGR2GRAY);
        if (mask.step != mask.cols)
            mask = mask.clone();
        CV_Assert(ref.step == ref.cols && mask.step == mask.cols);

        Size size = ref.size();
        for(int y=0; y<ref.rows; y+=d)
            for (int x = 0; x < ref.cols; x+=d)
            {
                int i = y*ref.cols + x;
                if (mask.data[i])
                {
                    int tx = x, ty = y;
                    sam(tx, ty, samWSZ, size);
                    int j = ty*ref.cols + tx;
                    int d = int(ref.data[i]) - ref.data[j];
                    if (i != j && mask.data[j] && abs(d)>=minDiff)
                    {
                        if (d > 0) {
                            _vPairs.push_back({i, j, (1)});
                        } else {
                            _vPairs.push_back({i, j, (0)});
                        }
                    }
                }
            }
        _imgSize = ref.size();
        printf("%d pixel pairs selected\n", (int)_vPairs.size());
    }
    float getScore(Mat img)
    {
        if (img.channels() == 3)
            cvtColor(img, img, CV_BGR2GRAY);
        if (img.step != img.cols)
            img = img.clone();
        CV_Assert(img.size() == _imgSize);

        int nm = 0;
        for (auto &p : _vPairs)
        {
            uchar a = img.data[p.i], b = img.data[p.j];
            if (p.val != 0 && a > b || p.val == 0 && a < b)
                ++nm;
        }
        return float(nm) / _vPairs.size();
    }
};

Mat imcat(Mat v[], int count, int w=5)
{
    int cols = (count-1)*w, rows=v[0].rows;
    for (int i = 0; i < count; ++i)
    {
        CV_Assert(v[i].type() == v[0].type() && v[i].rows == v[0].rows);
        cols += v[i].cols;
    }

    Mat m(Size(cols, rows), v[0].type());
    setMem(m, 0);
    int x = 0;
    for (int i = 0; i < count; ++i)
    {
        copyMem(v[i], m(Rect(x, 0, v[i].cols, rows)));
        x += v[i].cols + w;
    }

    return m;
}
Mat imcat(Mat a, Mat b, int w = 5)
{
    Mat v[] = { a,b };
    return imcat(v, 2, w);
}

class CapAlign
{
    std::string _refFile;
    Mat3b  _ref;
    Mat1b  _refMask;
    Rect   _refROI;

    Mat    _bestFuse, _bestImg, _bestMask, _bestAligned;
    float  _bestScore;
    RegionAlignScore _regionScore;
public:
    void set(const std::string &refFile, int BW=300)
    {
        Mat refx = cv::imread(refFile, -1);
        if (refx.channels() == 4)
        {
            _refMask = getChannel(refx, 3);
            cvtColor(refx, _ref, CV_BGRA2BGR);
        }
        else
        {
            CV_Assert(refx.channels() == 3);
            _ref = refx;
            std::string maskFile = ff::GetDirectory(refFile) + ff::GetFileName(refFile, false) + "_a.png";
            _refMask = cv::imread(maskFile, 0);
        }
        threshold(DWHS(_refMask), DS(_refMask), 127, 0, 255);
        _refFile = refFile;
        _refROI = cv::get_mask_roi(DWHS(_refMask), 127);
        rectAppend(_refROI, BW, BW, BW, BW);
        _refROI = rectOverlapped(_refROI, Rect(0, 0, _ref.cols, _ref.rows));
        _bestScore = -1e6;
        _regionScore.set(_ref(_refROI), _refMask(_refROI));
    }
    static Mat blend(const Mat3b &B, const Mat3b &F, const Mat1b &fmask)
    {
        Mat3b dimg = B.clone();
        for_each_3(DWHN0(dimg), DN0(F), DN1(fmask), [](Vec3b &f, const Vec3b &r, uchar m) {
            if (m)
            {
                for (int i = 0; i < 3; ++i)
                    f[i] = (f[i] + r[i]) / 2;
            }
        });
        return dimg;
    }
    void calcAligned(Mat img0)
    {
        Mat3b ref = _ref(_refROI);
        Mat1b refMask = _refMask(_refROI);
        Mat tar = img0(_refROI);

        Mat1b refGray, tarGray;
        cvtColor(ref, refGray, CV_BGR2GRAY);
        cvtColor(tar, tarGray, CV_BGR2GRAY);

        auto ptr = cv::optflow::createOptFlow_DeepFlowX();

        Mat2f flow;
        ptr->calc(refGray, tarGray, flow);

        Mat1b bmask;
        minFilter(refMask, bmask, 5);
        bmask = refMask - bmask;

        std::vector<Point2f> refPt, tarPt;
        for_each_1c(DWHNC(flow), [&refPt, &tarPt, bmask](const float *f, int x, int y) {
            if (x % 5 == 0 && y % 5 == 0 && /*uint(tx) < bmask.cols && uint(ty) < bmask.rows &&*/ bmask(y, x) != 0)
            {
                refPt.push_back(Point2f(x, y));
                tarPt.push_back(Point2f(x + f[0], y + f[1]));
            }
        });
        Mat1f H = findHomography(refPt, tarPt);

        Mat3b refWarp;
        warpPerspective(ref, refWarp, H, tar.size());

        Mat1b maskWarp;
        warpPerspective(refMask, maskWarp, H, tar.size());

#if 0
        auto clearBG = [](Mat1b &img, const Mat1b &mask) {
			for_each_2(DWHN1(img), DN1(mask), [](uchar &v, uchar m) {
				if (!m)
					v = 0;
			});
		};
		clearBG(refGray, refMask);
		clearBG(tarGray, maskWarp);
		ptr->calc(refGray, tarGray, flow);
#endif
        Mat3b tarWarp(ref.size());
        cv::warp_by_flow_nn(DWHN3(tar), DN(tarWarp), DN(flow));

        Mat vshow[] = { blend(tar, refWarp, maskWarp) , visRegionBoundary(maskWarp, tar), blend(ref,tarWarp,refMask)};

        _bestAligned = imcat(vshow, sizeof(vshow) / sizeof(vshow[0]));


        imshow("aligned", _bestAligned);

        _bestMask = Mat1b::zeros(img0.size());
        copyMem(maskWarp, _bestMask(_refROI));
    }
    void save()
    {
        std::string dir = ff::GetDirectory(_refFile) + ff::GetFileName(_refFile, false) + "/";
        if (!ff::pathExist(dir))
            ff::makeDirectory(dir);

        std::string imdir = dir + "images/";
        if (!ff::pathExist(imdir))
            ff::makeDirectory(imdir);

        std::string file;
        char name[32];
        for (int fi=1; ; ++fi)
        {
            sprintf(name, "%03d", fi);
            file = dir + name +".png";
            if (!ff::pathExist(file))
                break;
        }

        imwrite(file, _bestAligned);
        imwrite(imdir + name + ".png", _bestImg);
        imwrite(imdir + name + "_mask.png", _bestMask);
        _bestScore = 0;
    }
    void onCapture(Mat3b img0, int key)
    {
        //std::cout<<img0.size();
        //std::cout<<_ref.size();
        CV_Assert(img0.size() == _ref.size());

        Mat3b ref = _ref(_refROI);
        Mat1b refMask = _refMask(_refROI);
        Mat3b img = img0(_refROI);
        bool open = false;

        if (key == 'c')
            _bestScore = 0;
        else if (key == ' ')
            this->save();

        Mat3b fuse = blend(img,ref,refMask);
        Mat1b tarMask;

        float score = _regionScore.getScore(img);
        printf("score=%.2f  \r", score);
        //printf("bscore=%.2f  \r", _bestScore);

        if (_bestScore <= 0 || (score > _bestScore-0.01 && score>0.6))
        {
            _bestImg = img0.clone();
            _bestFuse = fuse;
            if(score>_bestScore)
                _bestScore = score;
            printf("bestScore=%.2f  \n", _bestScore);

            calcAligned(_bestImg);
        }
        imshow("frame", imcat(fuse,_bestFuse));
    }
};

void onMouse(int evt, int x, int y, int flags, void* userdata)
{
    if (evt == cv::EVENT_LBUTTONDOWN)
    {
        CapAlign *cap = (CapAlign*)userdata;
        cap->save();
    }
}


//拍照
#if 0
int main()
{
    //file of the template image
    std::string refFile = "D:\\data_model\\3.png";//6 9 10 12 13 14 15 16

    Size imgSize(1920, 1440);

    cv::VideoCapture cap;
    cap.open(1);
    cap.set(CAP_PROP_FRAME_WIDTH, imgSize.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, imgSize.height);
    CapAlign dcap;
    dcap.set(refFile);

    namedWindow("frame",0);
    cvResizeWindow("frame",900,600);
   cv::setMouseCallback("frame", onMouse, &dcap);

   namedWindow("aligned",0);
   cvResizeWindow("aligned",1200,600);

    Mat img;
    while (cap.read(img))
    {
        int c = waitKey(5);
        if (c == 'q')
            break;
        else if (c == 'r')
        {
            imwrite(refFile, img);
        }
        dcap.onCapture(img, c);
    }
    return 0;
}
#endif

//根据光流法的对应关系调整背景图像中目标对象的位置，使之与compose中的对象对齐
#if 1
int main()
{
    int index = 0;
    String refpath = "D:\\1ourDataset\\com\\*.png";
    std::vector<Mat> images;
    std::vector<String> imagenames;
    glob(refpath,imagenames);
    for(int i =0;i < imagenames.size() ;i++){
        String fullname = imagenames[i];

        String name = imagenames[i].substr(fullname.length()-11,7);//001
        //String name = imagenames[i].substr(0,7);


        Mat3b ref_ = imread("D:\\1ourDataset\\com\\"+name+".png",-1);
        Mat1b refMask = imread("D:\\1ourDataset\\mask\\"+name+".png",0);
        Mat tar_ = imread("D:\\1ourDataset\\bg\\"+name+".png",-1);
        //Mat1b tarMask = imread("D:\\data_model\\0\\images\\001_mask.png",0);
        int BW = 30;
        Rect refROI0 = cv::get_mask_roi(DWHS(refMask), 127);

        rectAppend(refROI0, BW, BW, BW, BW);
        Rect refROI = rectOverlapped(refROI0, Rect(0, 0, ref_.cols, ref_.rows));

        Mat3b ref = ref_(refROI);
        Mat tar = tar_(refROI);

        Mat1b refGray, tarGray;
        cvtColor(ref, refGray, CV_BGR2GRAY);
        cvtColor(tar, tarGray, CV_BGR2GRAY);

        auto ptr = cv::optflow::createOptFlow_DeepFlowX();

        Mat2f flow;
        ptr->calc(refGray, tarGray, flow);

        Mat1b bmask = refMask(refROI);

/*
    std::vector<Point2f> refPt, tarPt;
    for_each_1c(DWHNC(flow), [&refPt, &tarPt, bmask](const float *f, int x, int y) {
        if (bmask(y, x) != 0)
        {
            refPt.push_back(Point2f(x, y));
            tarPt.push_back(Point2f(x + f[0], y + f[1]));

        }
    });




    Mat tar_image = Mat::zeros(tar.rows,tar.cols,CV_8UC3);
    for(int index = 0;index<tarPt.size();index++)
    {
        if ( bmask(tarPt[index]) != 0) {
            Vec3f tar_img = tar.at<Vec3f>(tarPt[index]);
            float blue = tar_img.val[0];
            float green = tar_img.val[1];
            float red = tar_img.val[2];
            tar_image.at<Vec3f>(tarPt[index])[0] = blue;
            tar_image.at<Vec3f>(tarPt[index])[1] = green;
            tar_image.at<Vec3f>(tarPt[index])[2] = red;
        }
    }

    Mat ref_image = Mat::zeros(tar.rows,tar.cols,CV_8UC3);

    for(int index = 0;index<refPt.size();index++)
    {
        if (bmask(tarPt[index]) != 0) {
            Vec3f ref_img = ref.at<Vec3f>(refPt[index]);
            float blue = ref_img.val[0];
            float green = ref_img.val[1];
            float red = ref_img.val[2];
            ref_image.at<Vec3f>(tarPt[index])[0] = blue;
            ref_image.at<Vec3f>(tarPt[index])[1] = green;
            ref_image.at<Vec3f>(tarPt[index])[2] = red;
        }
    }


//    Mat int_tar_image = Mat::zeros(tar.rows,tar.cols,CV_8UC3);
//    tar_image.convertTo(int_tar_image,CV_8UC3,1,0);

    imshow("float_tar",tar_image);
    imshow("ref",ref_image);
    waitKey(0);
    std::cout<<tar_image.cols<<std::endl;

*/


        Mat xmap(tar.size(),CV_32FC1);
        Mat ymap(tar.size(),CV_32FC1);
        Mat uv(tar.size(),CV_32FC2);
//        Mat v(tar.size(),CV_32FC1);
        for_each_1c(DWHNC(flow), [&xmap, &ymap, &uv, bmask](const float *f, int x, int y) {
            //if (/*uint(tx) < bmask.cols && uint(ty) < bmask.rows &&*/ bmask(y, x) != 0)
            //{
            xmap.at<float>(y,x) = (float)x + f[0];
            ymap.at<float>(y,x) = (float)y + f[1];
            uv.at<float>(y,x)=*f;
            //}
        });



//        //新建以图片编号为名称的文件夹
//        String folderPath = "G:\\1ourDataset\\compose_flow";
//        if (0 != access(folderPath.c_str(), 0))
//        {
//            // if this folder not exist, create a new one.
//            mkdir(folderPath.c_str());   // 返回 0 表示创建成功，-1 表示失败
//            //换成 ::_mkdir  ::_access 也行，不知道什么意思
//        }

        const char* flowname = ("D:\\1ourDataset\\compose_flow\\"+name).c_str();
        FILE * rgba_file = std::fopen(flowname, "wb");
        fwrite(uv.data, uv.elemSize(), uv.cols * uv.rows, rgba_file);
        fclose(rgba_file);


        Mat3b tar_out(tar.size(),tar.type());
        remap(tar, tar_out, xmap, ymap, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        //tar_(refROI) = tar_out;
        copyMem(tar_out,tar_(refROI));

        imwrite("D:\\1ourDataset\\tarflow\\"+name+".png",tar_);
        std::cout<<"finished: "<<name<<std::endl;

    }



    return 0;
}
#endif