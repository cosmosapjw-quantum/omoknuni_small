#ifndef ALPHAZERO_CORE_EXPORT_MACROS_H
#define ALPHAZERO_CORE_EXPORT_MACROS_H

// Define export/import macros for shared library functionality
#if defined(_MSC_VER) || defined(WIN32)
    // Windows platform with MSVC or MinGW
    #if defined(ALPHAZERO_EXPORTS) || defined(alphazero_EXPORTS)
        #define ALPHAZERO_API __declspec(dllexport)
    #else
        #define ALPHAZERO_API __declspec(dllimport)
    #endif
#else
    // Linux/Unix platform
    #if defined(ALPHAZERO_EXPORTS) || defined(alphazero_EXPORTS)
        #define ALPHAZERO_API __attribute__((visibility("default")))
    #else
        #define ALPHAZERO_API
    #endif
#endif

#endif // ALPHAZERO_CORE_EXPORT_MACROS_H 