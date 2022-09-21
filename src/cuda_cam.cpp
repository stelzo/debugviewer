#include "cuda_cam.hpp"

#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if (signo == SIGINT)
    {
        LogInfo("received SIGINT\n");
        signal_recieved = true;
    }
}

using namespace cv;

int main()
{
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        LogError("can't catch SIGINT\n");

    videoOptions cam_options;
    cam_options.width = 2592;
    cam_options.height = 1944;
    cam_options.frameRate = 30;
    cam_options.deviceType = videoOptions::DeviceType::DEVICE_CSI;
    cam_options.ioType = videoOptions::IoType::INPUT;
    cam_options.codec = videoOptions::Codec::CODEC_RAW;
    cam_options.flipMethod = videoOptions::FlipMethod::FLIP_HORIZONTAL;

    videoSource *camera = videoSource::Create("csi://0", cam_options);

    if (!camera)
    {
        std::cout << "could not init cam" << std::endl;
        SAFE_DELETE(camera);
        return 1;
    }

    if (!camera->Open())
    {
        std::cout << "could not open cam" << std::endl;
        SAFE_DELETE(camera);
        return 1;
    }

    VideoWriter writer(
    "appsrc ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! nvoverlaysink sync=false",
    cv::CAP_GSTREAMER,
    0,   // fourcc
    cam_options.frameRate, // fps
    Size(cam_options.width, cam_options.height),
    {
        cv::VideoWriterProperties::VIDEOWRITER_PROP_IS_COLOR,
        1,
    });

    if (!writer.isOpened())
    {
        exit(-1);
    }

    

    auto size = cv::Size(cam_options.width, cam_options.height);
    cv::Mat dst(cv::Size(1920, 1080), CV_8UC3);
    
    while (!signal_recieved)
    {
        uchar3 *image = NULL; // RGB
        auto start_get_image = high_resolution_clock::now();
        camera->Capture(&image, 10000);
        auto end_get_image = high_resolution_clock::now();
        LogInfo("\ngetting %i ms\n", duration_cast<milliseconds>(end_get_image - start_get_image).count());
 
        if (!camera->IsStreaming())
        {
            signal_recieved = true;
            continue;
        }

        cuda::GpuMat gpu_frame(size, CV_8UC3, (void*) image); // caution, is not really BGR, but RGB in memory. 0ms

        cuda::GpuMat gpu_frame_bgr(size, CV_8UC3);
        cuda::GpuMat gpu_frame_resized(size, CV_8UC3);
        cuda::Stream stream;
       
        // ------- GPU -------
        auto start_gpu_process = high_resolution_clock::now();

        cuda::cvtColor(gpu_frame, gpu_frame_bgr, cv::COLOR_RGB2BGR, 0, stream);
        cuda::resize(gpu_frame_bgr, gpu_frame_resized, cv::Size(1920, 1080), 0, 0, 1, stream);

        stream.waitForCompletion();
        auto end_gpu_process = high_resolution_clock::now();
        LogInfo("gpu %i ms\n", duration_cast<milliseconds>(end_gpu_process - start_gpu_process).count());

        // ------- CPU -------
        auto start_download = high_resolution_clock::now();
        gpu_frame_resized.download(dst); // 5ms
        auto end_download = high_resolution_clock::now();
        LogInfo("download to cpu %i ms\n", duration_cast<milliseconds>(end_download - start_download).count());

        // ------- RENDER -------
        auto start_render_image = high_resolution_clock::now();
        //writer.write(dst);
        cv::imshow("img", dst);
        cv::waitKey(1);
        auto end_render_image = high_resolution_clock::now();
        LogInfo("render %i ms\n", duration_cast<milliseconds>(end_render_image - start_render_image).count());

    }
    
    writer.release();
    SAFE_DELETE(camera);
}