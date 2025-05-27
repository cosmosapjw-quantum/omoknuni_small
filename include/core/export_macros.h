#ifndef ALPHAZERO_CORE_EXPORT_MACROS_H
#define ALPHAZERO_CORE_EXPORT_MACROS_H

// Simple export macros for shared library building
#ifdef _WIN32
    #ifdef ALPHAZERO_EXPORTS
        #define ALPHAZERO_API __declspec(dllexport)
    #else
        #define ALPHAZERO_API __declspec(dllimport)
    #endif
#else
    #define ALPHAZERO_API __attribute__((visibility("default")))
#endif

#endif // ALPHAZERO_CORE_EXPORT_MACROS_H