#include <stdio.h>
#include <stdarg.h>

#include "nnom_port.h"


void NNOM_PORT_LOG(const char * fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
