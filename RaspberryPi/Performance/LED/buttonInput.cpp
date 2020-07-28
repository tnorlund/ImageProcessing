/* This turns on an LED light at the push of a button.
 *
 */
#include <iostream>
#include <wiringPi.h>
#include <boost/program_options.hpp>

#define LIGHT_PIN  0x07
#define SWITCH_PIN 0x00

class LEDButton {
    public:
        LEDButton() {
            // The pin modes that set the light and switch are set to output
            // and input, respectively. The switch pin is set to pull up. The
            // light starts as OFF.
            wiringPiSetup();
            pinMode( LIGHT_PIN, OUTPUT ); digitalWrite( LIGHT_PIN, LOW );
            pinMode( SWITCH_PIN, INPUT );
            pullUpDnControl( SWITCH_PIN, PUD_UP );
        }
        ~LEDButton() {
            // The pin that turns the light on and off is set so that the light
            // is off.
            digitalWrite( LIGHT_PIN, LOW );
        }
        void run( int run_time ) {
            // The LED button runs for a certain period of time. When the time 
            // is reached, the function ends.
            time_t timer_begin, timer_end, button_begin, button_end;
            int count;
            bool time_done = false; bool button_pushed = false;
            time( &timer_begin );
            while( !time_done ) {
                if ( !digitalRead( SWITCH_PIN ) ) {
            //     // The switch is read as LOW when the button is pushed.
            //     if ( !digitalRead( SWITCH_PIN ) ) {
            //         button_pushed = true;
            //         // Loop here while the button is considered to be pushed.
            //         while( button_pushed ) {
            //             std::cout << "The button is being pushed." << std::endl;
                        digitalWrite( LIGHT_PIN, HIGH );
                    //     if( digitalRead( SWITCH_PIN ) ) {
                    //         time( &button_begin );
                    //         while( digitalRead( SWITCH_PIN ) && !button_pushed ) {
                    //             std::cout << "The button was let go of." << std::endl;
                    //             time( &button_end );
                    //             if ( 
                    //                 difftime( button_end, button_begin ) >= 0.5 
                    //             ) {
                    //                 digitalWrite( LIGHT_PIN, LOW );
                    //                 button_pushed = false;
                    //             }
                    //             delay( 100 );
                    //         }
                    //     }
                    //     delay( 100 );
                        
                    // }
                    // // delay(50);
                }
                else {
                    digitalWrite( LIGHT_PIN, LOW );
                    std::cout << "The button is being pressed." << std::endl;
                }
                // digitalWrite( LIGHT_PIN, !digitalRead(SWITCH_PIN) );
                delay(100);
                time ( &timer_end );
                if ( difftime ( timer_end, timer_begin ) >= run_time )
                    time_done = true;
            }
        }
};

int main ( int argc, char * argv[] ) {
    int seconds;
    LEDButton button;
    boost::program_options::options_description description( "Allowed options" );
    description.add_options()
        ( "help,h", "Help screen" )
        ( 
            "time,t", boost::program_options::value( &seconds ),
            "The length of time to run the program.\nDefaults to 15 seconds."
        );
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::command_line_parser( argc, argv )
            .options( description ).run(), 
        vm
    );
    boost::program_options::notify( vm );
    if ( vm.count( "help" ) ) { 
        std::cout << description << std::endl; return 1; 
    }
    if ( !vm.count( "time" ) ) { seconds = 15; }
    button.run( seconds );
    return 0 ;
}