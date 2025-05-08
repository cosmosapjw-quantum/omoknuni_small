#ifndef ALPHAZERO_CORE_EXPORT_MACROS_H
#define ALPHAZERO_CORE_EXPORT_MACROS_H

// Define export/import macros for DLL functionality
#if defined(_MSC_VER)
    #if defined(ALPHAZERO_EXPORTS)
        #define ALPHAZERO_API __declspec(dllexport)
    #else
        #define ALPHAZERO_API __declspec(dllimport)
    #endif
#else
    #define ALPHAZERO_API
#endif

#endif // ALPHAZERO_CORE_EXPORT_MACROS_H 