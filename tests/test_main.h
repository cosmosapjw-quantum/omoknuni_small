#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <gtest/gtest.h>

// Only include main function if not using a custom main
#ifndef CUSTOM_MAIN_USED
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif // CUSTOM_MAIN_USED

#endif // TEST_MAIN_H 