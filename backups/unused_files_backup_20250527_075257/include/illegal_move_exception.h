// File: core/illegal_move_exception.h
#ifndef ALPHAZERO_CORE_ILLEGAL_MOVE_EXCEPTION_H
#define ALPHAZERO_CORE_ILLEGAL_MOVE_EXCEPTION_H

#include <stdexcept> // For std::runtime_error
#include <string>    // For std::string
#include <sstream>   // For std::stringstream

namespace alphazero {
namespace core {

/**
 * @brief Exception class for illegal moves attempted in a game state.
 *
 * This exception is thrown when an operation like makeMove encounters
 * an action that is not valid according to the game's rules or current state
 * (e.g., moving to an occupied cell, violating specific rules like Ko or
 * Renju forbidden moves).
 */
class IllegalMoveException : public std::runtime_error {
public:
    /**
     * @brief Constructs an IllegalMoveException.
     *
     * @param message A descriptive message explaining why the move is illegal.
     * @param attempted_action The integer representation of the action that was attempted illegally. Defaults to -1 if action is not applicable.
     */
    explicit IllegalMoveException(const std::string& message, int attempted_action = -1)
        : std::runtime_error(build_what_message(message, attempted_action)), // Construct base class with the full message
          action_(attempted_action),
          base_message_(message) 
          {}

    /**
     * @brief Constructs an IllegalMoveException using a C-style string.
     *
     * @param message A C-style string message explaining why the move is illegal.
     * @param attempted_action The integer representation of the action that was attempted illegally. Defaults to -1 if action is not applicable.
     */
    explicit IllegalMoveException(const char* message, int attempted_action = -1)
        : std::runtime_error(build_what_message(message, attempted_action)), // Construct base class with the full message
          action_(attempted_action),
          base_message_(message ? message : "") // Ensure base_message_ is initialized even if message is null
          {}


    /**
     * @brief Gets the integer representation of the illegal action attempted.
     *
     * @return The action index, or -1 if not applicable/provided.
     */
    int getAction() const noexcept {
        return action_;
    }

    /**
     * @brief Gets the base error message provided during construction (without the action details).
     * * @return const std::string& The base error message.
     */
    const std::string& getBaseMessage() const noexcept {
        return base_message_;
    }

    /**
     * @brief Returns the explanatory string (inherited from std::exception -> std::runtime_error).
     *
     * Overriding is optional as the base class constructor is already initialized with the full message.
     * This ensures `what()` returns the message created by `build_what_message`.
     * * @return const char* Pointer to a C-style string with the error description.
     */
    // const char* what() const noexcept override {
    //     // Implementation inherited from std::runtime_error which stores the message passed to its constructor.
    //     return std::runtime_error::what(); 
    // }


private:
    int action_; // The illegal action attempted (-1 if not applicable)
    std::string base_message_; // Store the original message part

    /**
     * @brief Helper function to build the final message for the std::runtime_error base class.
     */
    static std::string build_what_message(const std::string& message, int action) {
        std::stringstream ss;
        ss << "Illegal Move: " << message;
        if (action != -1) {
            ss << " (Action: " << action << ")";
        }
        return ss.str();
    }
     /**
     * @brief Helper function overload for C-style strings.
     */
    static std::string build_what_message(const char* message, int action) {
         std::stringstream ss;
         ss << "Illegal Move: " << (message ? message : "[No message provided]");
         if (action != -1) {
             ss << " (Action: " << action << ")";
         }
         return ss.str();
     }

};

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_ILLEGAL_MOVE_EXCEPTION_H