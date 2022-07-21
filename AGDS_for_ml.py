from collections import Counter
import numpy as np
import copy
from itertools import combinations_with_replacement
from numpy.random import uniform

from joblib import Parallel, delayed


def calculate_neighbour_weight(val1, val2, val_max, val_min):
    if val_max == val_min:
        return abs(val1 - val2)
    return 1 - (abs(val2 - val1) / (val_max - val_min))


def calculate_object_attr_weight(number_of_att_value_in_all_obj):
    return 1 / number_of_att_value_in_all_obj


def find_neighbours(aggregated_feature_list, node):
    if node is None:
        return None
    index_of_node = aggregated_feature_list.index(node)
    left_node = None
    right_node = None

    if index_of_node != 0:
        left_node = aggregated_feature_list[index_of_node - 1]

    if index_of_node != len(aggregated_feature_list) - 1:
        right_node = aggregated_feature_list[index_of_node + 1]
    return left_node, right_node


def propagate_left(feature_nodes, assigned_weights, current_nodes, feature_index):
    left_node, right_node = find_neighbours(feature_nodes[feature_index]['sorted'], current_nodes[feature_index])
    # print(f"Propagating feature: {feature_index} | current: {current_nodes[feature_index]} | left {left_node}")
    # print(f"Assigned_weights: {assigned_weights[feature_index]}")
    # print(f"Feature nodes: {feature_nodes}")
    if left_node or left_node == 0:
        assigned_weights[feature_index][left_node] = \
            assigned_weights[feature_index][current_nodes[feature_index]] * calculate_neighbour_weight(
                current_nodes[feature_index],
                left_node, max(feature_nodes[feature_index]['sorted']),
                min(feature_nodes[feature_index]['sorted']))
        current_nodes[feature_index] = left_node
        propagate_left(feature_nodes, assigned_weights, current_nodes, feature_index)

    else:
        return


def propagate_right(feature_nodes, assigned_weights, current_nodes, feature_index):
    left_node, right_node = find_neighbours(feature_nodes[feature_index]['sorted'], current_nodes[feature_index])
    # print(f"Propagating feature: {feature_index} | current: {current_nodes[feature_index]} | right {right_node}")
    # print(f"Assigned_weights: {assigned_weights[feature_index]}")
    # print(f"Feature nodes: {feature_nodes}")
    if right_node or right_node == 0:
        assigned_weights[feature_index][right_node] = \
            assigned_weights[feature_index][current_nodes[feature_index]] * calculate_neighbour_weight(
                current_nodes[feature_index],
                right_node, max(feature_nodes[feature_index]['sorted']),
                min(feature_nodes[feature_index]['sorted']))
        current_nodes[feature_index] = right_node
        propagate_right(feature_nodes, assigned_weights, current_nodes, feature_index)
    else:
        return


def find_nearest_left_value(list_, value):
    output_val = None
    for idx, val in enumerate(list_):
        if val <= value:
            output_val = val
        else:
            break
    return output_val


def find_nearest_right_value(list_, value):
    output_val = None
    for idx, val in enumerate(list_):
        if val >= value:
            output_val = val
            break
    return output_val


def apply_initial_stimulation(feature_nodes, assigned_weights, current_nodes):
    left_initial_state = current_nodes.copy()
    right_initial_state = current_nodes.copy()

    for feature_index in current_nodes:
        if current_nodes[feature_index] not in feature_nodes[feature_index]['sorted']:
            # print(f"{current_nodes[feature_index]} not in {feature_nodes[feature_index]}")
            left_initial_state[feature_index] = find_nearest_left_value(feature_nodes[feature_index]['sorted'],
                                                                        current_nodes[feature_index])
            right_initial_state[feature_index] = find_nearest_right_value(feature_nodes[feature_index]['sorted'],
                                                                          current_nodes[feature_index])

            if left_initial_state[feature_index] or left_initial_state[feature_index] == 0:
                assigned_weights[feature_index][left_initial_state[feature_index]] = \
                    calculate_neighbour_weight(left_initial_state[feature_index], current_nodes[feature_index],
                                               max(feature_nodes[feature_index]['sorted']),
                                               min(feature_nodes[feature_index]['sorted']))

            if right_initial_state[feature_index] or right_initial_state[feature_index] == 0:
                assigned_weights[feature_index][right_initial_state[feature_index]] = \
                    calculate_neighbour_weight(right_initial_state[feature_index], current_nodes[feature_index],
                                               max(feature_nodes[feature_index]['sorted']),
                                               min(feature_nodes[feature_index]['sorted']))
    return left_initial_state, right_initial_state


def propagate_to_neighbours(feature_nodes, assigned_weights, current_nodes):
    left_initial_state, right_initial_state = apply_initial_stimulation(feature_nodes, assigned_weights, current_nodes)
    # print(f"Initial left : {left_initial_state}")
    # print(f"Initial right : {right_initial_state}")

    for feature_index in current_nodes:
        if left_initial_state[feature_index] is not None:
            left_node, _ = find_neighbours(feature_nodes[feature_index]['sorted'], left_initial_state[feature_index])
            # print(left_node)
            # print(f"Initial: {initial_nodes}")
            if left_node or left_node == 0:  # not 0 = False but we need 0
                try:
                    propagate_left(feature_nodes, assigned_weights, left_initial_state.copy(), feature_index)
                except:
                    raise Exception(f"LEFT Assigned_weights: {assigned_weights}")
        if right_initial_state[feature_index] is not None:
            _, right_node = find_neighbours(feature_nodes[feature_index]['sorted'], right_initial_state[feature_index])
            # print(right_node)
            # print(f"Initial: {initial_nodes}")
            if right_node or right_node == 0:
                try:
                    propagate_right(feature_nodes, assigned_weights, right_initial_state.copy(), feature_index)
                except:
                    raise Exception(f"RIGHT Assigned_weights: {assigned_weights}")


def run_similarity_search(node, feature_nodes, all_objects):
    original_node = node
    assigned_weights = {
        i: {original_node[i]: 1}
        for i in range(len(node))
    }
    current_nodes = {
        i: original_node[i]
        for i in range(len(node))
    }
    # print(assigned_weights)

    propagate_to_neighbours(feature_nodes, assigned_weights, current_nodes)

    object_scores = []
    # print(assigned_weights)
    for obj_ in all_objects:
        obj_score = 0.0
        for feature_index in range(len(obj_)):
            # print(f"obj[feature_idx] {obj_[feature_index]}")
            if feature_index not in assigned_weights:
                continue
            curr_weight = assigned_weights[feature_index][obj_[feature_index]]
            feature_node_weight = feature_nodes[feature_index]['counter'][obj_[feature_index]]
            # try:
            obj_score += curr_weight * 1 / feature_node_weight
            # except:
            # pass
        object_scores.append(obj_score)
    return assigned_weights, object_scores


def run_AGDS_algorithm_minimise(min_goal_function, max_iter, n_random_params_per_epoch, n_initial_population,
                                n_population, n_units_for_exploration=2, explore_always=False, evenly_distribute=False,
                                random_param_function=None):
    n_initial_population = n_initial_population
    n_population = n_population

    top_n_to_average = 1

    parameters_value: [tuple] = []  # represents current knowledge

    to_train = [random_param_function()
                for _ in range(n_initial_population)
                ]  # initialising of epoch
    #   print(f"Initialised: {to_train}")
    epochs_min_accuracy = []

    explore = True
    evals_counter = 0
    for _ in range(max_iter):  # number of populations
        # We are training what we need to train
        # print("Training of new models!")
        if evals_counter >= max_iter:
            break
        for model_params in to_train:
            if evals_counter >= max_iter:
                break
            model_accuracy = min_goal_function(model_params)
            parameters_value.append((model_params, model_accuracy))
            # print((model_params, model_accuracy))
            evals_counter += 1
            # print(f"Evals: {evals_counter}")

        # Evaluate of epoch
        current_best_model_acc = min([el[1] for el in parameters_value])
        epochs_min_accuracy.append(current_best_model_acc)
        # print(f"Current best model: {current_best_model_acc}")

        # if current_best_model_acc > 1.99:
        #   print(f"FINISHED! evals: {evals_counter}")
        #   break

        if explore_always:
            explore = True
        elif len(epochs_min_accuracy) > 2 and epochs_min_accuracy[-2] < epochs_min_accuracy[-1]:
            explore = False
        else:
            explore = True

        #     print(parameters_value)

        vectorised_params = [list(el[0]) + [el[1]] for el in parameters_value]

        # Building AGDS
        # print("Building AGDS")
        feature_nodes = {
            i: {'sorted': sorted(list(set([el[i] for el in vectorised_params]))),
                'counter': Counter([el[i] for el in vectorised_params])
                }
            for i in range(len(vectorised_params[0]))
        }

        # generate new models for initial evaluation. Much larger than n_population!
        #     params_to_check = copy.copy(all_space)
        params_to_check = [random_param_function() for _ in range(n_random_params_per_epoch)]
        # print(params_to_check[0])

        current = set([tuple(el1[0]) for el1 in parameters_value])

        params_to_check = [el for el in params_to_check if tuple(el) not in current]

        # print(params_to_check[0])
        # find most similar objects and get their expected accuracy
        # params_probable_score = []
        # all_similarities = []
        #
        def f(params):
            """
            returns similarity score (highest) and probable score of model
            :return:
            """
            assigned_weights, similarities = run_similarity_search(params, feature_nodes, vectorised_params)
            most_similar_ones = np.argsort(similarities)[::-1]
            return similarities[most_similar_ones[0]], np.sum(
                [
                    vectorised_params[idx][-1] * similarities[idx] for idx in most_similar_ones[:top_n_to_average]
                ]
            ) / np.sum([similarities[idx] for idx in most_similar_ones[:top_n_to_average]])

        results = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(params) for params in params_to_check)
        all_similarities = [el[0] for el in results]
        params_probable_score = [el[1] for el in results]
        top_indexes_from_new_generation = np.argsort(params_probable_score)[:n_population]
        to_train = [params_to_check[idx] for idx in top_indexes_from_new_generation]
        if explore:
            lest_similar = np.argsort(all_similarities)
            to_train = to_train + [params_to_check[idx] for idx in lest_similar[:n_units_for_exploration]]

    return min([el[1] for el in parameters_value])
