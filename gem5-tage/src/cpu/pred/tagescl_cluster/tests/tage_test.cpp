// Standard C++ headers
#include <cassert>
#include <cstdint>  // for uint32_t, uint64_t
#include <iomanip>
#include <iostream>
#include <vector>

// Project headers
#include "../tage.hpp"
#include "../tagescl_configs.hpp"
#include "../utils.hpp"

// Commented out alternative paths for reference
// #include "../../tagescl/tage.hpp"
// #include "../../tagescl/tagescl_configs.hpp"
// #include "../../tagescl/utils.hpp"
// Function to print prediction information
void print_prediction(uint64_t pc,
    const tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>& info) {
    std::cout << "Prediction for PC 0x"
        << std::hex << pc << std::dec << ":" << std::endl;
    std::cout << "  Predicted: "
        << (info.prediction ? "Taken" : "Not taken") << std::endl;
    std::cout << "  Confidence: "
              << (info.high_confidence ? "High" :
                 (info.medium_confidence ? "Medium" :
                 (info.low_confidence ? "Low" : "None"))) << std::endl;
    std::cout << "  Hit bank: " << info.hit_bank << std::endl;
    std::cout << "  Alt bank: " << info.alt_bank << std::endl;
}

// Simple test program
int main() {
    std::cout << "Starting TAGE unit test..." << std::endl;

    // Create random number generator
    tagescl::Random_Number_Generator rng;

    // Set up initial values for the random number generator
    int64_t dummy_phist = 0;
    int64_t dummy_tghist = 0;
    rng.phist_ptr_ = &dummy_phist;
    rng.ptghist_ptr_ = &dummy_tghist;

    // Create TAGE predictor instance using CONFIG_64KB
    constexpr int MAX_IN_FLIGHT_BRANCHES = 32;
    tagescl::Tage<tagescl::CONFIG_64KB::TAGE> tage_predictor(rng,
        MAX_IN_FLIGHT_BRANCHES);

    // Define some branch patterns for testing
    const int NUM_BRANCHES = 10;
    std::vector<uint64_t> branch_pcs = {
        0x1000, 0x1008, 0x1010, 0x1018, 0x1020,
        0x2000, 0x2008, 0x2010, 0x2018, 0x2020
    };
    std::vector<bool> branch_outcomes = {
        true, false, true, true, false,
        true, true, false, false, true
    };

    // Statistics for verification
    int correct_predictions = 0;
    int total_predictions = 0;

    // // Test 0: Original TAGE
    // std::cout << "\n==== Test 0: Original TAGE ====\n" << std::endl;

    // // Train the predictor with the branch pattern
    // for (int j = 0; j < 1; j++) {
    // for (int i = 0; i < NUM_BRANCHES; i++) {
    //     uint64_t pc = branch_pcs[i];
    //     bool outcome = branch_outcomes[i];
    //     uint32_t cluster_id = 0;  // Default cluster
    //     bool is_h2p = false;

    //     // Create prediction info object
    //     tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>
    //         prediction_info;

    //     // Get prediction
    //     tage_predictor.get_prediction(pc, &prediction_info);

    //     // Add debugging information
    //     std::cout << "\nDebug info for PC 0x" << std::hex << pc
    //         << std::dec << ":" << std::endl;
    //     std::cout << "  Bimodal prediction: "
    //         << (prediction_info.alt_prediction ? "Taken" : "Not taken")
    //         << std::endl;
    //     std::cout << "  Bimodal confidence: "
    //         << (prediction_info.alt_confidence ? "High" : "Low")
    //         << std::endl;

    //     std::cout << "  Indices: ";
    //     for (int i = 1;
    //         i <= 2 * tagescl::CONFIG_64KB::TAGE::NUM_HISTORIES; i++) {
    //         std::cout << prediction_info.indices[i] << " ";
    //     }
    //     std::cout << std::endl;

    //     std::cout << "  Tags: ";
    //     for (int i = 1; i <= 2 * tagescl::CONFIG_64KB::TAGE::NUM_HISTORIES;
    //         i++) {
    //         std::cout << prediction_info.tags[i] << " ";
    //     }
    //     std::cout << std::endl;

    //     std::cout << "  Hit bank: " << prediction_info.hit_bank
    //         << std::endl;
    //     std::cout << "  Alt bank: " << prediction_info.alt_bank
    //         << std::endl;
    //     std::cout << "  Final prediction: "
    //         << (prediction_info.prediction ? "Taken" : "Not taken")
    //         << std::endl;

    //     print_prediction(pc, prediction_info);
    //     // Update stats
    //     if (prediction_info.prediction == outcome) {
    //         correct_predictions++;
    //     }
    //     total_predictions++;

    //     // Branch metadata
    //     // Conditional, not indirect
    //     tagescl::Branch_Type branch_type = {true, false};
    //     uint64_t branch_target = pc + 8;  // Next instruction

    //     // Update speculative state
    //     tage_predictor.update_speculative_state(
    //         pc, branch_target, branch_type,
    //         prediction_info.prediction, &prediction_info);

    //     // Commit state
    //     tage_predictor.commit_state(pc, outcome, prediction_info, outcome);

    //     // Retire
    //     tage_predictor.commit_state_at_retire(prediction_info);
    // }
    // }

    // // Report initial accuracy
    // std::cout << "Initial training accuracy: " << correct_predictions << "/"
    //           << total_predictions << " ("
    //           << (100.0 * correct_predictions / total_predictions) << "%)"
    //           << std::endl;

    // // Reset stats
    // correct_predictions = 0;
    // total_predictions = 0;

    // Test 1: Basic prediction accuracy
    std::cout << "\n==== Test 1: Basic Prediction Accuracy ====\n"
        << std::endl;

    // Train the predictor with the branch pattern
    for (int j = 0; j < 1; j++) {
    for (int i = 0; i < NUM_BRANCHES; i++) {
        uint64_t pc = branch_pcs[i];
        bool outcome = branch_outcomes[i];
        uint32_t cluster_id = 0;  // Default cluster
        bool is_h2p = false;

        // Create prediction info object
        tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>
            prediction_info;

        // Get prediction
        tage_predictor.
            get_prediction(pc, cluster_id, is_h2p, &prediction_info);
        print_prediction(pc, prediction_info);
        // Update stats
        if (prediction_info.prediction == outcome) {
            correct_predictions++;
        }
        total_predictions++;

        // Branch metadata
        // Conditional, not indirect
        tagescl::Branch_Type branch_type = {true, false};
        uint64_t branch_target = pc + 8;  // Next instruction

        // Update speculative state
        tage_predictor.update_speculative_state(
            pc, branch_target, branch_type,
            prediction_info.prediction, &prediction_info);

        // Commit state
        tage_predictor.commit_state(pc, outcome, prediction_info, outcome);

        // Retire
        tage_predictor.commit_state_at_retire(prediction_info);
    }
    }

    // Report initial accuracy
    std::cout << "Initial training accuracy: " << correct_predictions << "/"
              << total_predictions << " ("
              << (100.0 * correct_predictions / total_predictions) << "%)"
              << std::endl;

    // Reset stats
    correct_predictions = 0;
    total_predictions = 0;

    // Test the predictor again after training
    std::cout << "\n==== Test 2: Prediction After Training ====\n"
        << std::endl;

    for (int i = 0; i < NUM_BRANCHES; i++) {
        uint64_t pc = branch_pcs[i];
        bool outcome = branch_outcomes[i];
        uint32_t cluster_id = 0;
        bool is_h2p = false;

        tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>
            prediction_info;
        tage_predictor
            .get_prediction(pc, cluster_id, is_h2p, &prediction_info);

        print_prediction(pc, prediction_info);

        if (prediction_info.prediction == outcome) {
            correct_predictions++;
        }
        total_predictions++;

        tagescl::Branch_Type branch_type = {true, false};
        uint64_t branch_target = pc + 8;

        tage_predictor.update_speculative_state(
            pc, branch_target, branch_type,
            prediction_info.prediction, &prediction_info);

        tage_predictor.commit_state(pc, outcome, prediction_info, outcome);
        tage_predictor.commit_state_at_retire(prediction_info);
    }

    // Report trained accuracy
    std::cout << "\nTrained accuracy: " << correct_predictions << "/"
              << total_predictions << " ("
              << (100.0 * correct_predictions / total_predictions) << "%)"
              << std::endl;

    // Test 3: Cluster-specific predictions (if supported)
    std::cout << "\n==== Test 3: Cluster-Specific Predictions ====\n"
        << std::endl;

    // Try a prediction with a specific cluster ID
    uint64_t test_pc = 0x3000;
    uint32_t test_cluster_id = 2;  // Use cluster 1 (bit 1 set)
    bool test_is_h2p = true;       // Use history-to-prediction path

    tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>
        cluster_prediction_info;
    tage_predictor.get_prediction(test_pc, test_cluster_id, test_is_h2p,
        &cluster_prediction_info);

    std::cout << "Cluster-specific prediction:" << std::endl;
    print_prediction(test_pc, cluster_prediction_info);

    // Test 4: Recovery from mispeculation
    std::cout << "\n==== Test 4: Recovery from Mispeculation ====\n"
        << std::endl;

    // Scenario: branch predicted one way, but actually goes the other way
    uint64_t recovery_pc = 0x4000;
    uint32_t recovery_cluster_id = 0;
    bool recovery_is_h2p = false;

    tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE> recovery_info;
    tage_predictor.get_prediction(recovery_pc, recovery_cluster_id,
        recovery_is_h2p, &recovery_info);

    std::cout << "Initial prediction:" << std::endl;
    print_prediction(recovery_pc, recovery_info);

    // Update speculative state with prediction
    tagescl::Branch_Type recovery_type = {true, false};
    uint64_t recovery_target = recovery_pc + 8;

    tage_predictor.update_speculative_state(
        recovery_pc, recovery_target, recovery_type,
        recovery_info.prediction, &recovery_info);

    // Now recover state because prediction was wrong
    bool actual_outcome = !recovery_info.prediction; // Opposite of prediction

    // Recover from mispeculation
    tage_predictor.global_recover_speculative_state(recovery_info);

    // Show prediction after recovery
    tagescl::Tage_Prediction_Info<tagescl::CONFIG_64KB::TAGE>
        post_recovery_info;
    tage_predictor.get_prediction(recovery_pc, recovery_cluster_id,
        recovery_is_h2p, &post_recovery_info);

    std::cout << "\nPrediction after recovery:" << std::endl;
    print_prediction(recovery_pc, post_recovery_info);

    // Now commit the correct outcome
    tage_predictor.update_speculative_state(
        recovery_pc, recovery_target, recovery_type,
        actual_outcome, &post_recovery_info);

    tage_predictor.commit_state(recovery_pc, actual_outcome,
        post_recovery_info, actual_outcome);
    tage_predictor.commit_state_at_retire(post_recovery_info);

    std::cout << "\nTAGE unit test completed successfully!" << std::endl;
    return 0;
}
