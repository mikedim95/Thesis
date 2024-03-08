use std::error::Error;
use std::thread;
use std::time::Duration;

use rppal::gpio::Gpio;
use rppal::system::DeviceInfo;

// Gpio uses BCM pin numbering. BCM GPIO 23 is tied to physical pin 16.
const GPIO_LED: u8 = 4;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Blinking an LED on a {}.", DeviceInfo::new()?.model());

    let mut pin = Gpio::new()?.get(GPIO_LED)?.into_output();

    // Blink the LED by setting the pin's logic level high for 500 ms.
   loop {
        // Blink the LED by setting the pin's logic level high for 500 ms.
        println!("Led on");
        pin.set_high();
        thread::sleep(Duration::from_millis(100));

        // Turn off the LED by setting the pin's logic level low for 500 ms.
        println!("Led off");
        pin.set_low();
        thread::sleep(Duration::from_millis(100));
    }
   
}