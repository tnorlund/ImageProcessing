/* This measures how quickly the camera can take an image and write it to the disk.
 *
 * TODO
 * - Contact raspicam github
 */
#include <iostream>
#include <thread>
#include <filesystem>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <boost/program_options.hpp>
#include <raspicam/raspicam_cv.h>
#include <raspicam/raspicam_still_cv.h>
#include <opencv2/opencv.hpp>

const std::string DIRECTORY = "images";
const int HEIGHT = 960;
const int WIDTH = 1280;

class FrameRateTest {
    int number_threads;
    bool verbose;
    std::mutex test_mutex;
    raspicam::RaspiCam_Cv Camera;
    std::vector<std::thread> threads;
    std::string file_name;
    std::ofstream file_output; 
    std::ifstream file_input;
    public:
        FrameRateTest( 
            int _num_threads, bool _verbose, std::string _file_name 
        ) {
            verbose = _verbose;
            file_name = _file_name;
            if ( file_name != "" ) { 
                if ( !std::filesystem::exists(file_name) ) {
                    file_input.open(file_name);
                    file_output.open(file_name, std::ios::app); 
                    file_output << "Datetime,Runtime,FPS,NumberFrames,NumberThreads,"
                        + std::string("BatchSize,\n");
                    file_input.close();
                    file_output.close();
                }
            }
            Camera.set( cv::CAP_PROP_FORMAT, CV_8UC3 );
            if ( !Camera.open() ) { throw "Error opening the camera"; }
            if ( _num_threads > std::thread::hardware_concurrency() ) {
                throw std::invalid_argument(
                    "Too many threads given ( " + std::to_string(_num_threads) 
                        + ">" + std::to_string(
                            std::thread::hardware_concurrency()
                        ) + ")"
                );
            }
            if ( _num_threads == 0 ) { number_threads = 1; } 
            else { number_threads = _num_threads; }
            if ( !std::filesystem::exists(DIRECTORY) ) {
                std::filesystem::create_directory(DIRECTORY); 
            }
            // Here, we can reserve the OpenCV matrices to store the images and
            // threads.
            threads.reserve(number_threads);
        }
        ~FrameRateTest() {
            Camera.release();
            std::filesystem::remove_all(DIRECTORY);
        }
        void run( int number_frames, int batch ) {
            std::vector<int> frame_distribution;
            int modulus = number_frames % number_threads;
            time_t timer_begin, timer_end;
            double secondsElapsed = 0;
            // In order to call the threads with the correct distribution of 
            // frames, we must distribute the work between the threads. Note 
            // that the remaining (modulus) frames that cannot be evenly 
            // distributed are applied to the first numbe of threads.
            for ( size_t i = 0; i < number_threads; i++) {
                frame_distribution.push_back(
                    number_frames / number_threads + modulus
                );
                if (modulus > 0) modulus--;
            }
            if ( verbose )
                std::cout << "Going to capture " << number_frames << 
                    " frames with " << number_threads << " threads (" << 
                    batch << " images per batch)" << std::endl;
            // With all of the work distributed among the different threads, we
            // can run the test by running the private function in seperate 
            // threads.
            time ( &timer_begin );
            for (size_t i = 0; i < number_threads; i++) {
                threads.push_back(
                    std::thread(
                        &FrameRateTest::save_captures, this, 
                        frame_distribution[i], batch
                    )
                );
            }
            for( auto &thread : threads ) { thread.join(); }
            time ( &timer_end );
            secondsElapsed = difftime ( timer_end, timer_begin );
            if ( verbose )
                std::cout << secondsElapsed << " seconds for " << 
                    number_frames << " frames : FPS = " <<  
                    ( float ) ( 
                        ( float ) ( number_frames )  / 
                        ( float ) secondsElapsed 
                    ) << std::endl;
            if ( file_name != "" ) { 
                file_input.open(file_name);
                file_output.open(file_name, std::ios::app); 
                file_output << std::to_string(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count()
                    ) + "," + std::to_string(secondsElapsed) + "," 
                    + std::to_string(
                        ( float ) ( 
                            ( float ) ( number_frames )  / 
                            ( float ) secondsElapsed 
                        )
                    ) + "," + std::to_string(number_frames) + "," 
                    + std::to_string(number_threads) + "," 
                    + std::to_string(batch) + ",\n";
                file_input.close();
                file_output.close();
            }
        }
    private:
        void save_captures( int number_frames, int batch ) {
            cv::Mat image = cv::Mat(HEIGHT, WIDTH, CV_8UC3);
            std::vector<cv::Mat> images;
            std::vector<std::string> capture_times;
            // The first thing to do is create a vector of Mat objects to store
            // the grabbed captures from the camera. The times of the captures 
            // must be stored as well. This loop would be a good place to 
            // initialize the strings.
            for (size_t i = 0; i<batch; i++) { 
                images.push_back(image.clone()); 
                capture_times.push_back("");
            }
            // After that, the number of frames to process per batch is 
            // required.
            std::vector<int> batch_sizes(number_frames/batch, batch);
            if ( number_frames%batch != 0 ) { 
                batch_sizes.push_back(number_frames%batch);
            }
            // With the memory allocated and the number of frames distributed 
            // among the different batched-jobs, the different batches can be 
            // run.
            for ( int &batch_size : batch_sizes ) { 
                test_mutex.lock();
                for ( size_t i = 0; i<batch_size; i++ ) {   
                    Camera.grab();
                    capture_times[i] = std::to_string(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count()
                    );
                    Camera.retrieve(images[i]);
                }
                test_mutex.unlock();
                for ( size_t i = 0; i<batch_size; i++ ) { 
                    cv::imwrite(
                        "./" + DIRECTORY + "/" + capture_times[i] + ".png", 
                        images[i]
                    );
                }
            }
        }
};


int main (int argc, char * argv[]) {
    int number_threads, frames, batch;
    std::string file_name;
    bool verbose = false;
    boost::program_options::options_description description("Allowed options");
    description.add_options()
        ("help,h", "Help screen")
        ("threads,t", boost::program_options::value(&number_threads), 
            "The number of threads.\nThis defaults to 1.")
        ("frames,f", boost::program_options::value(&frames), 
            "The number of frames.\nThis defaults to 50.")
        ("batch,b", boost::program_options::value(&batch), 
            "The number of frames to store in memory.\nThis defaults to 1.")
        ("filename,d", boost::program_options::value(&file_name), 
            "The file name used to store the test results.")
        ("verbose,v", "Whether to print messages to screen");
    boost::program_options::variables_map vm;
    boost::program_options::store (
        boost::program_options::command_line_parser (argc, argv)
            .options(description).run (), 
        vm
    );
    boost::program_options::notify (vm);
    if (vm.count("help")) { std::cout << description << std::endl; return 1; }
    if (!vm.count("threads")) { number_threads = 1; }
    if (!vm.count("frames")) { frames = 50; }
    if (!vm.count("batch")) { batch = 1; }
    if (!vm.count("filename")) { file_name = ""; }
    if (vm.count("verbose")) { verbose = true; }
    FrameRateTest test(number_threads, verbose, file_name);
    test.run(frames, batch);
    return 0;
}