#!/usr/bin/env python3
"""
Test script for ELO rating system.
"""

import os
import sys
import json
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

from python.alphazero.elo_system import ELORatingSystem, RandomPolicyModel, create_elo_report


def test_basic_elo_computation():
    """Test basic ELO computation."""
    print("Testing basic ELO computation...")
    
    elo = ELORatingSystem()
    
    # Add models
    elo.add_model("model_a", rating=1500)
    elo.add_model("model_b", rating=1500)
    
    # Test expected score calculation
    expected = elo.expected_score(1500, 1500)
    assert abs(expected - 0.5) < 0.001, f"Expected 0.5, got {expected}"
    
    expected = elo.expected_score(1500, 1100)  # 400 point advantage
    assert abs(expected - 0.909) < 0.01, f"Expected ~0.909, got {expected}"
    
    # Test rating update
    old_a = elo.get_rating("model_a")
    old_b = elo.get_rating("model_b")
    
    # Model A wins 60 games, B wins 40
    elo.update_ratings("model_a", "model_b", wins_a=60, wins_b=40, draws=0)
    
    new_a = elo.get_rating("model_a")
    new_b = elo.get_rating("model_b")
    
    # A should gain rating, B should lose
    assert new_a > old_a, "Winner should gain rating"
    assert new_b < old_b, "Loser should lose rating"
    assert abs((new_a - old_a) + (new_b - old_b)) < 0.1, "Rating changes should sum to ~0"
    
    print("✅ Basic ELO computation test passed!")


def test_random_baseline():
    """Test random policy baseline."""
    print("\nTesting random policy baseline...")
    
    elo = ELORatingSystem()
    
    # Random policy should be at ELO 0
    random_rating = elo.get_rating("random_policy")
    assert random_rating == 0.0, f"Random policy should be at ELO 0, got {random_rating}"
    
    # Test calibration against random
    elo.add_model("strong_model")
    elo.calibrate_against_random("strong_model", wins=90, losses=5, draws=5)
    
    strong_rating = elo.get_rating("strong_model")
    assert strong_rating > 200, f"Strong model should have high rating, got {strong_rating}"
    
    # Test weak model
    elo.add_model("weak_model")
    elo.calibrate_against_random("weak_model", wins=20, losses=75, draws=5)
    
    weak_rating = elo.get_rating("weak_model")
    assert weak_rating < -200, f"Weak model should have low rating, got {weak_rating}"
    
    print("✅ Random baseline test passed!")


def test_k_factor_adaptation():
    """Test K-factor adaptation for new vs established models."""
    print("\nTesting K-factor adaptation...")
    
    elo = ELORatingSystem(k_factor_new=32, k_factor_established=16, games_until_established=20)
    
    elo.add_model("new_model", 1500)
    elo.add_model("old_model", 1500)
    
    # Set old model as established
    elo.game_counts["old_model"] = 25
    
    # New model should use higher K-factor
    assert elo.get_k_factor("new_model") == 32
    assert elo.get_k_factor("old_model") == 16
    
    # After 20 games, new model should use lower K-factor
    elo.game_counts["new_model"] = 20
    assert elo.get_k_factor("new_model") == 16
    
    print("✅ K-factor adaptation test passed!")


def test_save_load():
    """Test saving and loading ELO data."""
    print("\nTesting save/load functionality...")
    
    elo1 = ELORatingSystem()
    elo1.add_model("model_x", 1234)
    elo1.add_model("model_y", 1567)
    elo1.update_ratings("model_x", "model_y", wins_a=5, wins_b=3, draws=2)
    
    # Save
    test_file = "test_elo_ratings.json"
    elo1.save_ratings(test_file)
    
    # Load into new system
    elo2 = ELORatingSystem()
    elo2.load_ratings(test_file)
    
    # Verify ratings match
    assert elo2.get_rating("model_x") == elo1.get_rating("model_x")
    assert elo2.get_rating("model_y") == elo1.get_rating("model_y")
    assert elo2.game_counts["model_x"] == elo1.game_counts["model_x"]
    
    # Cleanup
    os.remove(test_file)
    
    print("✅ Save/load test passed!")


def test_random_policy_model():
    """Test random policy model."""
    print("\nTesting random policy model...")
    
    random_model = RandomPolicyModel(board_size=15)
    
    # Test prediction
    import numpy as np
    dummy_board = np.zeros((1, 19, 15, 15))
    
    policy, value = random_model.predict(dummy_board)
    
    # Check policy is uniform
    assert len(policy) == 15 * 15
    assert abs(policy.sum() - 1.0) < 0.001
    assert all(abs(p - 1/225) < 0.001 for p in policy)
    
    # Check value is neutral
    assert value[0] == 0.0
    
    print("✅ Random policy model test passed!")


def test_integration():
    """Test full integration scenario."""
    print("\nTesting integration scenario...")
    
    elo = ELORatingSystem()
    
    # More realistic scenario: only update based on actual matches
    # Initial model is calibrated against random
    elo.add_model("initial")
    elo.calibrate_against_random("initial", wins=70, losses=25, draws=5)
    
    # Each subsequent model plays against the previous champion
    models = [
        ("iter_1", "initial", 55, 40, 5),    # iter_1 beats initial 55-40
        ("iter_2", "iter_1", 52, 43, 5),     # iter_2 beats iter_1 52-43
        ("iter_3", "iter_2", 48, 47, 5),     # iter_3 barely beats iter_2
        ("iter_4", "iter_3", 45, 50, 5),     # iter_4 loses to iter_3
        ("iter_5", "iter_3", 58, 37, 5),     # iter_5 plays iter_3 (the champion)
    ]
    
    current_champion = "initial"
    
    for new_model, opponent, wins, losses, draws in models:
        # Add new model with default rating
        elo.add_model(new_model)
        
        # Play match
        elo.update_ratings(new_model, opponent, wins_a=wins, wins_b=losses, draws=draws)
        
        # Update champion if new model wins
        if wins > losses:
            current_champion = new_model
            print(f"{new_model} becomes new champion")
    
    # Check progression
    ratings = elo.get_sorted_ratings()
    print("\nFinal ratings:")
    for model, rating in ratings:
        print(f"  {model}: {rating:.1f}")
    
    # Verify random is still at 0
    assert elo.get_rating("random_policy") == 0.0
    
    # Check that we have reasonable progression
    # The best models should have positive ratings
    best_model = ratings[0][0]
    assert ratings[0][1] > 100  # Best model should be significantly above random
    
    # Create report
    create_elo_report(elo, "test_elo_report.txt")
    assert os.path.exists("test_elo_report.txt")
    os.remove("test_elo_report.txt")
    
    print("✅ Integration test passed!")


def main():
    """Run all tests."""
    print("Starting ELO system tests...")
    
    try:
        test_basic_elo_computation()
        test_random_baseline()
        test_k_factor_adaptation()
        test_save_load()
        test_random_policy_model()
        test_integration()
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()