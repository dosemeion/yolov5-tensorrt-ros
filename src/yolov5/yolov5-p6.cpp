#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include "n_camera_obj/BoundingBox.h"
#include "n_camera_obj/BoundingBoxes.h"
// #include "detectNet.h"
#include <boost/bind.hpp>

#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.5
#define CONF_THRESH 0.45
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


// float* detectNet::Infer (IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize)
// {
//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//     context.enqueue(batchSize, buffers, stream, nullptr);
//     CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);
//     return output;
// }
// detectNet* net = NULL;



void doInference(const sensor_msgs::ImageConstPtr& img_data, IExecutionContext* context, cudaStream_t stream, void **buffers, float* data, float* prob, int batchSize, ros::Publisher& obj_pub) {
    // auto t_start = std::chrono::high_resolution_clock::now();
    // TYPE_8UC3: bgr8编码
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_data, sensor_msgs::image_encodings::TYPE_8UC3);
    // std::cout << "cv_bridge" << cv_ptr->image.size() << std::endl;
    cv::Mat pr_img = preprocess_img(cv_ptr->image, INPUT_W, INPUT_H); // letterbox BGR to RGB
    // std::cout << "preprocess_img" << pr_img.size() << std::endl;

    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[0 + i] = (float)uc_pixel[2] / 255.0;
            data[0 + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[0 + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(buffers[0], data, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);


    std::vector<Yolo::Detection> batch_res;
    nms(batch_res, &prob[0], CONF_THRESH, NMS_THRESH);
    xywh2xyxy(batch_res);
    scale_coords(batch_res, 1280, 720); // 摄像头宽度1280，高度720
    n_camera_obj::BoundingBox box;
    std::vector<n_camera_obj::BoundingBox> boxes;
    n_camera_obj::BoundingBoxes pub_box;

    // auto& res = batch_res;
    //std::cout << res.size() << std::endl;
    // cv::Mat img = cv::imread("/media/hqd/Samsung_T5/code/perception/hqd/src/n_camera_obj/src/yolov5/0.jpg");
    // cv::Mat img = cv_ptr->image;
    int num_obj = 0;
    for (size_t j = 0; j < batch_res.size(); j++) {

        if (batch_res[j].bbox[0] < 0 || batch_res[j].bbox[1] < 0 || batch_res[j].bbox[2] <= 0 || batch_res[j].bbox[3] <= 0) continue;

        box.camera_obj_xmin = batch_res[j].bbox[0];
        box.camera_obj_ymin = batch_res[j].bbox[1];
        box.camera_obj_xmax = batch_res[j].bbox[2];
        box.camera_obj_ymax = batch_res[j].bbox[3];
        box.camera_obj_class = batch_res[j].class_id;
        box.camera_obj_prob = batch_res[j].conf;
        boxes.push_back(box);
        num_obj += 1;
        // cv::Rect r = cv::Rect(batch_res[j].bbox[0], batch_res[j].bbox[1], batch_res[j].bbox[2] - batch_res[j].bbox[0], batch_res[j].bbox[3] - batch_res[j].bbox[1]);
        // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        // cv::putText(img, std::to_string((int)batch_res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }

    // cv::imshow("detect", img);
    // std::cout << "detect" << img.size() << std::endl;
    // cv::imwrite("/media/hqd/Samsung_T5/code/perception/hqd/src/n_camera_obj/src/yolov5/_0.jpg", img);
    // cv::waitKey(1);

    pub_box.count = num_obj;
    pub_box.bounding_boxes = boxes;
    // std::cout << "boxes" << pub_box << std::endl;
    obj_pub.publish(pub_box);
    ROS_INFO("publish %ld camera_obj.", boxes.size());
}


int main(int argc, char** argv) {
    ros::init (argc, argv, "n_camera_obj");
    ros::NodeHandle obj;
    cudaSetDevice(DEVICE);

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    // 创建IRuntime实例
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    // IRuntime::deserializeCudaEngine(存放序列化engine的内存，内存大小)

    std::string wts_name = "";
    std::string engine_name = "/media/hqd/Samsung_T5/code/perception/hqd/src/n_camera_obj/src/yolov5/yolov5s6.engine";
    // float gd = 0.0f, gw = 0.0f;
    std::string img_dir;

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    // 创建tensorRT模型文件流，供加载序列化的engine使用
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // 反序列化加载engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    // 创建上下文
    IExecutionContext* context = engine->createExecutionContext();
    // assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // 创建buffers空间，供输入输出使用
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    // 创建cuda流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));




    ros::Publisher obj_pub = obj.advertise<n_camera_obj::BoundingBoxes>("msg_camera_obj", 1);
    // void doInference(const sensor_msgs::Image& img_data, IExecutionContext* context, cudaStream_t& stream, void **buffers, float* data, float* prob, int batchSize)
    ros::Subscriber obj_sub = obj.subscribe<sensor_msgs::Image>("msg_camera_prep", 1, boost::bind(&doInference, _1, context, stream, buffers, data, prob, BATCH_SIZE, obj_pub));

    ros::spin();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    ROS_INFO("n_camera_obj shutdown.");

    return 0;
}
