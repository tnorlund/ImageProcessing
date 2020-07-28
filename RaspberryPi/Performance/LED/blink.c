#include <wiringPi.h>
int main (void)
{
  int GPIO_pin = 7;
  wiringPiSetup () ;
  pinMode (GPIO_pin, OUTPUT) ;
  for (int i=0; i<15; i++)
  {
    digitalWrite (GPIO_pin, HIGH) ; delay (500) ;
    digitalWrite (GPIO_pin,  LOW) ; delay (500) ;
  }
  return 0 ;
}