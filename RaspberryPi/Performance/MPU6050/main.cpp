/* This measures how quickly the MPU6050 can be polled and written to disk.
 *
 * The original code is from http://www.electronicwings.com
 * 
 * TODO
 * - Multithread?
 */

#include <algorithm>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <wiringPiI2C.h>
#include <wiringPi.h>
#include <boost/program_options.hpp>

#define Device_Address 0x68	/* Device Address/Identifier for MPU6050 */

#define PWR_MGMT_1   0x6B
#define SMPLRT_DIV   0x19
#define CONFIG       0x1A
#define GYRO_CONFIG  0x1B
#define INT_ENABLE   0x38
#define ACCEL_XOUT_H 0x3B
#define ACCEL_YOUT_H 0x3D
#define ACCEL_ZOUT_H 0x3F
#define GYRO_XOUT_H  0x43
#define GYRO_YOUT_H  0x45
#define GYRO_ZOUT_H  0x47

class MPU6050 {
    int fd;
    float Acc_x, Acc_y, Acc_z;
    float Gyro_x, Gyro_y, Gyro_z;
    float Ax=0, Ay=0, Az=0;
    float Gx=0, Gy=0, Gz=0;
    public:
        MPU6050() {
            // In order to initializes a connection to the MPU6050, the 
            // device's address must be found. This address is then used to
            // write to registers:
            // 1. Sample Rate
            // 2. Power Management
            // 3. Accelerometer Configuration
            // 4. Gyroscope Configuration
            // 5. Interupt
            fd = wiringPiI2CSetup( Device_Address );
            wiringPiI2CWriteReg8( fd, SMPLRT_DIV,  0x07 );
            wiringPiI2CWriteReg8( fd, PWR_MGMT_1,  0x01 );
            wiringPiI2CWriteReg8( fd, CONFIG,      0 );
            wiringPiI2CWriteReg8( fd, GYRO_CONFIG, 24 );
            wiringPiI2CWriteReg8( fd, INT_ENABLE,  0x01 );
        }
        ~MPU6050() {}
        void write_to_console( int seconds ) {
            time_t timer_begin, timer_end;
            bool time_done = false;
            std::cout << "Writing data to console..." << std::endl;
            time( &timer_begin );
            while ( !time_done ) {
                this->read_all_data();
                std::cout << "Gx=" << Gx << " °/s\t" << "Gy=" << Gy << " °/s\t" 
                    << "Gz=" << Gz << " °/s\t" << "Ax=" << Ax << " g\t" << 
                    "Ay=" << Ay << " g\t" << "Az=" << Az << " g" << std::endl;
                time( &timer_end );
                if ( difftime ( timer_end, timer_begin ) >= seconds )
                    time_done = true;
            }
        }
        void write_to_file( int seconds, std::string file_name ) {
            time_t timer_begin, timer_end;
            bool time_done = false;
            // In order to write the recorded data to a file, the file's stream
            // must be declared. 
            std::ofstream file_output;
            std::cout << "file_name: " << file_name << std::endl;
            // If the file does not exist, the header of the ".csv" file needs
            // to be written.
            if ( !std::filesystem::exists( file_name ) ) {
                file_output.open( file_name, atd::ios::out ); 
                file_output << "Datetime,Gx,Gy,Gz,Ax,Ay,Az\n";
            } else { file_output.open( file_name, std::ios::app ); }
            // With the file initialized, the MPU6050 can be read and the data 
            // can be written to the file.
            time( &timer_begin );
            while ( !time_done ) {
                this->read_all_data();
                file_output << std::to_string(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count()
                ) << "," << Gx << "," << Gy << "," << Gz << "," << Ax << "," <<
                    Ay << "," << Az << ",\n";
                time( &timer_end );
                if ( difftime ( timer_end, timer_begin ) >= seconds )
                    time_done = true;
            }
            // After writing the data to the file for the given period of time,
            // the file is closed to cleanup the process.
            file_output.close();
        }
    private:
        short read_raw_data( int addr ) {
            short high_byte, low_byte, value;
            high_byte = wiringPiI2CReadReg8( fd, addr );
            low_byte = wiringPiI2CReadReg8( fd, addr + 1 );
            value = ( high_byte << 8 ) | low_byte;
            return value;
        }
        void read_all_data() {
            // The accelerometer and gyroscopic data is read and scaled to be
            // measured in degrees per second and newtons with respect to
            // Earth's gravity.
            Acc_x = read_raw_data( ACCEL_XOUT_H );
            Acc_y = read_raw_data( ACCEL_YOUT_H );
            Acc_z = read_raw_data( ACCEL_ZOUT_H );
            Gyro_x = read_raw_data( GYRO_XOUT_H );
            Gyro_y = read_raw_data( GYRO_YOUT_H );
            Gyro_z = read_raw_data( GYRO_ZOUT_H );
            Ax = Acc_x / 16384.0; Ay = Acc_y / 16384.0; Az = Acc_z / 16384.0;
            Gx = Gyro_x / 131; Gy = Gyro_y / 131; Gz = Gyro_z / 131;
        }
};

int main( int argc, char * argv[] ) {
    int time; std::string file_name;
    MPU6050 this_sensor;
    boost::program_options::variables_map vm;
    boost::program_options::options_description description("Allowed options");
    description.add_options()
        ("help,h", "Help screen")
        ("time,t", boost::program_options::value(&time), 
            "The amount of time to run the MPU6050 in seconds.\nThis defaults "
            "to 10.")
        ("disk,d", 
            "Whether to write the data to the disk. Otherwise, the data is "
            "written to the screen.")
        ("filename,f", boost::program_options::value(&file_name), 
            "The file name used to store the test results.\nThis defaults to "
            "the UNIX time with a \".csv\" file extension appended.");
    
    boost::program_options::store(
        boost::program_options::command_line_parser (argc, argv)
            .options(description).run (), 
        vm
    );
    boost::program_options::notify( vm );
    if ( vm.count( "help" ) ) {
        std::cout << description << std::endl; return 1; 
    }
    if ( !vm.count( "time" ) ) { time = 10; }
    if ( !vm.count( "filename" ) ) { 
        file_name = std::to_string(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        ) + ".csv"; 
    }
    if ( vm.count( "disk" ) ) { 
        this_sensor.write_to_file( time, file_name );
    } else {
        this_sensor.write_to_console( time ); 
    }
    return 0;
}
