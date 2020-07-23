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
    std::mutex m_mutex;
    raspicam::RaspiCam_Cv Camera;
    std::vector<std::thread> threads;
    std::vector<cv::Mat> images;
    public:
        FrameRateTest(int _num_threads, bool _verbose) {
            cv::Mat image;
            verbose = _verbose;
            Camera.set( cv::CAP_PROP_FORMAT, CV_8UC3 );
            if ( !Camera.open() ) { 
                throw "Error opening the camera"; 
            }
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
            if ( ~std::filesystem::exists(DIRECTORY) ) {
                std::filesystem::create_directory(DIRECTORY); 
            }
            // Here, we can reserve the OpenCV matrices to store the images and
            // threads.
            images.reserve(number_threads);
            threads.reserve(number_threads);
            for (size_t i = 0; i < number_threads; i++) { 
                images.push_back(image); 
            }
        }
        ~FrameRateTest() {
            Camera.release();
            std::filesystem::remove_all(DIRECTORY);
        }
        void run(int number_frames) {
            std::vector<int> distribution;
            int modulus = number_frames % number_threads;
            time_t timer_begin, timer_end;
            double secondsElapsed = 0;
            // In order to call the threads with the correct distribution of 
            // frames, we must distribute the work between the threads. Note 
            // that the remaining (modulus) frames that cannot be evenly 
            // distributed are applied to the first numbe of threads.
            for (size_t i = 0; i < number_threads; i++) {
                distribution.push_back(
                    number_frames / number_threads + modulus
                );
                if (modulus > 0) modulus--;
            }
            if ( verbose )
                std::cout << "Going to capture " << number_frames << 
                    " frames with " << number_threads << " threads." << 
                    std::endl;
            // With all of the work distributed among the different threads, we
            // can run the test by running the private function in seperate 
            // threads.
            time ( &timer_begin );
            for (size_t i=0; i < number_threads; i++) {
                threads.push_back(
                    std::thread(
                        &FrameRateTest::save_captures, this, images[i], 
                        distribution[i]
                    )
                );
            }
            for( auto &thread : threads ) { thread.join(); }
            time ( &timer_end );
            secondsElapsed = difftime ( timer_end, timer_begin );
            if ( verbose )
                std::cout << secondsElapsed << " seconds for " << 
                    number_frames << " frames : FPS = " <<  
                    ( float ) ( ( float ) ( number_frames )  / 
                    ( float ) secondsElapsed ) << std::endl;
        }
    private:
        void save_captures(cv::Mat image, int number_frames) {
            
            for (size_t i = 0; i < number_frames; i++) {
                std::string current_time;
                m_mutex.lock();
                Camera.grab();
                current_time = std::to_string(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count()
                );
                Camera.retrieve(image);
                m_mutex.unlock();
                cv::imwrite(
                    "./" + DIRECTORY + "/" + current_time + ".png", 
                    image
                );
            }
        }
};


int main (int argc, char * argv[]) {
    int number_threads, frames;
    bool verbose = false;
    boost::program_options::options_description description("Allowed options");
    description.add_options ()
        ("help,h", "Help screen")
        ("threads,t", boost::program_options::value(&number_threads), "The number of threads.\nThis defaults to 1.")
        ("frames,f", boost::program_options::value(&frames), "The number of frames.\nThis defaults to 50.")
        ("verbose,v", "Whether to print messages to screen");
    // Parse command line arguments
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
    if (vm.count("verbose")) { verbose = true; }
    if (verbose) std::cout << "Starting now!" << std::endl;
    FrameRateTest test(number_threads, verbose);
    test.run(frames);
    return 1;
}