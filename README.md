# message-logging-kernels
Implementing MPI message logging capabilities as HPC kernel extensions

Requires ULFM 2.0 (or newer) configured with --with-ft
https://fault-tolerance.org/ulfm/downloads/

If power-saving is enabled with message logging protocols, Mammut is required:
https://github.com/DanieleDeSensi/mammut

We have used Mammut with acpi_cpufreq kernel driver, and msr-safe for permissions with MSR registers: 
https://github.com/LLNL/msr-safe
