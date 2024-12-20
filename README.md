# AirTest Mobile Automation

AirTest Mobile Automation is an object-oriented, multi-process control framework for mobile apps based on the AirTest framework. It is designed for stability and compatibility, making it ideal for automating tasks in games like Honor of Kings.

## Features

* **Enhanced Stability**: This framework enhances the reliability of testing and automation by implementing rigorous connection status checks and intelligent automatic reconnection mechanisms. Instead of throwing errors upon failure, it encapsulates AirTest functions to attempt reconnections, ensuring continuous operation even in the face of transient network issues or screenshot retrieval failures. In rare cases, such as with iOS devices, it may prompt for a physical reconnect of the device to be recognized by `tidevice list`. Additionally, it addresses and corrects issues with the `start_app` function that could occur on certain Android systems, providing a smoother and more resilient automation experience.
* **Automated Operation**: Capable of fully automated processes with unattended operation. In case of errors, it automatically restarts the app or the control endpoint, such as Docker or an Android emulator.
* **Time Management**: Utilizes the UTC/GMT+08:00 time zone for task scheduling aligned with Chinese game refresh cycles.
* **Formatted Output**: Displays information in a formatted manner, e.g.,  `[MM-DD HH:MM:SS] info`.

## Modules

### Device Management ( `deviceOB` )

Handles device management in an object-oriented approach, supporting various clients and control endpoints.

| Clients | Control Endpoints | Management Capabilities |
|---------|-------------------|-------------------------|
| [BlueStacks](https://www.bluestacks.com/download.html) / [LDPlayer](https://www.ldplayer.net/)/ [MuMu](https://mumu.163.com/) | Windows | Start, stop, and restart emulators |
| [Docker](https://hub.docker.com/r/redroid/redroid) | Linux | Start, stop, and restart containers |
| iOS | Mac | Reconnect with tidevice, restart iOS |
| USB-connected Android phones | ALL | Reboot Android system |
| WIFI-connected phones | ALL | Hot reboot Android system |

### APP Management ( `appOB` )

Manages the opening, stopping, and restarting of apps.

### Tools ( `DQWheel` )

* A utility for multi-process support based on the file system, including synchronization, broadcasting, file management, and enhanced time management.
* Utilizes files and dictionaries to store and retrieve image coordinates, reducing the time to locate element coordinates repeatedly. It also allows for selecting specific positions, such as the least proficient hero in Honor of Kings based on proficiency.

## Development Examples

Below are some examples of development scripts that demonstrate the capabilities of the project.

- An automation script for [Honor of Kings](https://github.com/cndaqiang/WZRY) .
- Automate daily sign-ins and claim gift packages on Android with [autoansign](https://github.com/MobileAutoFlow/autoansign).

For the configuration file format, see [this guide](https://wzry-doc.pages.dev/guide/config/).

## Acknowledgements

The historical versions of this script, namely [WZRY_AirtestIDE_XiaoMi11@cndaqiang](https://github.com/cndaqiang/WZRY_AirtestIDE_XiaoMi11), [WZRY_AirtestIDE_emulator@cndaqiang](https://github.com/cndaqiang/WZRY_AirtestIDE_emulator), and [WZRY@cndaqiang](https://github.com/cndaqiang/WZRY) have also been instrumental in the development process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

&copy; 2024 cndaqiang. All rights reserved.
