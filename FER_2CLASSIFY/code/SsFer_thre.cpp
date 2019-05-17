/***************************************************************************************************
* @brief: caffe for FER classification c++ interface.
*  
*  @run like: SsFer.bin /home/spark/grocery/FER/codes/caffe_2classify/test/size32_2classify_deploy.prototxt 
*  /home/spark/grocery/FER/codes/caffe_2classify/model/size32_2classify_v4_iter_19500.caffemodel labels.txt
*/
#include "caffe/caffe.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "opencv2/opencv.hpp"
#include<fstream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#ifdef CPU_ONLY
using std::string;
using std::vector;
using caffe::Blob;

typedef std::pair<string, float> Prediction;

class SsFer
{
  public:
    SsFer(string& model_file, 
          string& trained_model,
          string& label_file);
    Prediction Classify(cv::Mat& img, float thresh);

  private:
    cv::Mat PreProcess(cv::Mat& img);
    vector<float> Predict(cv::Mat& inputImg);
  private:
    caffe::Net<float> * net;
    std::vector<string> labels;
    cv::Size input_size;
};

SsFer::SsFer(string& model_file,
             string& trained_model,
             string& label_file){
  net = new caffe::Net<float>(model_file,caffe::TEST); 
  net->CopyTrainedLayersFrom(trained_model);
  
  std::ifstream labels_(label_file.c_str());
  string line;
  while (std::getline(labels_, line))
    labels.push_back(string(line));
  
  //Blob<float>* input_layer = net->input_blobs()[0];
  caffe::shared_ptr<caffe::Blob<float> > input_layer = net->blob_by_name("data");
  int num_channels = input_layer->channels();
  if(num_channels != 1)
    std::cout << "input img channel is error!" << std::endl;
  input_size = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}


static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}


cv::Mat SsFer::PreProcess(cv::Mat& img)
{
    cv::Mat RezImg;
    if(cv::Size(img.cols, img.rows) != input_size)
      {
        //std::cout<<"img size is not according to to proto...,convert now" << std::endl;
        cv::resize(img, RezImg, cv::Size(input_size.width, input_size.height));
      }
    else
      RezImg = img;
		cv::Mat Img_float;
    RezImg.convertTo(Img_float, CV_32FC1);
    return Img_float;
}


vector<float> SsFer::Predict(cv::Mat& InputData)
{
    int length = input_size.height *  input_size.width;
    caffe::shared_ptr<caffe::Blob<float> > inputBlob = net->blob_by_name("data");
    inputBlob->Reshape(1, 1, input_size.height, input_size.width);
    net->Reshape();
    
    float *p = inputBlob->mutable_cpu_data();
		memcpy(p, InputData.data, sizeof(float) * length);
    net->Forward();
    
    /* Copy the output layer to a std::vector */
    //caffe::Blob<float>* output_layer = net->output_blobs()[0]; //index 0 is blob, 1 is diff blob
    caffe::shared_ptr<Blob<float> > output_layer = net->blob_by_name("out");
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);

}

Prediction SsFer::Classify(cv::Mat& img, float thresh)
{
    std::vector<float> out;
    cv::Mat imgProcessed = PreProcess(img); 
    out = Predict(imgProcessed);
		//int N = std::min<int>(labels.size(), topN);
  	//std::vector<int> maxN = Argmax(out, N);
  	Prediction prediction;
    std::string resClass;
    if(out[0] > thresh)
      resClass = "neutral";
    else
      resClass = "non-neutral";

    prediction = make_pair(resClass, out[0]);
    return prediction;
}


int main(int argc, char* argv[])
{
    if(argc != 4)
      {    
        std::cerr << "Usage: " << argv[0]
        << " deploy.prototxt network.caffemodel"
        << " labels.txt"
        << std::endl;
        return 1;
      }

    string model_file = argv[1];
    string trained_model = argv[2];
    string label_file = argv[3];
    float thresh  = 0.1;

    SsFer fer(model_file, trained_model, label_file);
    /*
    for (int i = 1; i < 3; i++)
    {
        std::stringstream ss;
        ss<<i;
        cv::Mat img = cv::imread("./imgs/" + ss.str() + ".jpg", 0);
        cv::TickMeter tm;
        tm.reset();
        tm.start();
        std::vector<Prediction> res =  fer.Classify(img, topN);
        tm.stop();

        std::cout << "Image: " << i << ".jpg" << std::endl;
        std::cout << "Time Cost: "<<
                  tm.getTimeMilli() << " ms" << std::endl;
				
			  // Print the top N predictions. 
  			for (size_t i = 0; i < res.size(); ++i) {
    		Prediction p = res[i];
    		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  			}
	
    }
    */
  std::ifstream fin; 
  string str;
  fin.open("imgs.txt");
  while(!fin.eof())
  {
    getline(fin, str);
    if(str.empty())
      continue;
    cv::Mat img = cv::imread(str, 0);
    Prediction res =  fer.Classify(img, thresh);
    std::cout << std::fixed << std::setprecision(4) << str << " " << res.second << " - \""
               << res.first << "\"" << std::endl;
  }
  fin.close();  
   return 0;
}

#endif //CPU_ONLY
