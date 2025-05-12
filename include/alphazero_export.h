#pragma once

/**
 * @file
 * @brief Export macros for AlphaZero library
 *
 * This file contains macros for properly exporting symbols from shared libraries.
 * Supports both Windows and Linux platforms.
 */

// Define export/import macros for the AlphaZero library
#ifndef ALPHAZERO_API
    #ifdef ALPHAZERO_EXPORTS
        // We are building the library
        #define ALPHAZERO_API __attribute__((visibility("default")))
    #else
        // We are using the library
        #define ALPHAZERO_API 
    #endif
#endif

// Macro to use in class declarations to properly export/import them
#define ALPHAZERO_EXPORT ALPHAZERO_API 
