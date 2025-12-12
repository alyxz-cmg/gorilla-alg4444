"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from pathlib import Path
from pprint import pformat, pprint
import monai.metrics as mm
# from scipy.spatial import distance
from scipy.spatial import cKDTree
import numpy as np

from helpers import run_prediction_processing
from multiprocessing import Pool
from helpers import get_max_workers

import os
INPUT_DIRECTORY = Path(f"{os.getcwd()}/test/input")
OUTPUT_DIRECTORY = Path(f"{os.getcwd()}/test/output")
GROUND_TRUTH_DIRECTORY = Path(f"{os.getcwd()}/ground_truth")

# for docker building
# INPUT_DIRECTORY = Path("/input")
# OUTPUT_DIRECTORY = Path("/output")
# GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

SPACING_LEVEL0 = 0.24199951445730394
GT_MM = True

def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_detected_lymphocytes = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-lymphocytes",
    )
    location_detected_monocytes = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-monocytes",
    )
    location_detected_inflammatory_cells = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-inflammatory-cells",
    )

    # Secondly, read the results
    result_detected_lymphocytes = load_json_file(
        location=location_detected_lymphocytes,
    )

    result_detected_monocytes = load_json_file(
        location=location_detected_monocytes,
    )

    result_detected_inflammatory_cells = load_json_file(
        location=location_detected_inflammatory_cells,
    )

    if not GT_MM:
        result_detected_inflammatory_cells = convert_mm_to_pixel(result_detected_inflammatory_cells)
        result_detected_monocytes = convert_mm_to_pixel(result_detected_monocytes)
        result_detected_lymphocytes = convert_mm_to_pixel(result_detected_lymphocytes)

    # Thirdly, retrieve the input image name to match it with an image in your ground truth
    file_id = get_image_name(
        values=job["inputs"],
        slug="kidney-transplant-biopsy",
    )
    file_id = file_id.split("_PAS")[0]
    # Fourthly, load your ground truth
    # Include it in your evaluation container by placing it in ground_truth/
    gt_lymphocytes = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_lymphocytes.json")
    gt_monocytes = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_monocytes.json")
    gt_inf_cells = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_inflammatory-cells.json")

    # compare the results to your ground truth and compute some metrics
    radius_lymph = 0.004 if GT_MM else int(4 / SPACING_LEVEL0) # margin for lymphocytes is 4um at spacing 0.25 um / pixel
    radius_mono = 0.005 if GT_MM else int(5 / SPACING_LEVEL0) # margin for monocytes is 5um at spacing 0.25 um / pixel
    radius_infl = 0.005 if GT_MM else int(5 / SPACING_LEVEL0) # margin for inflammatory cells is 5um at spacing 0.24 um / pixel
    lymphocytes_eval = get_froc_vals_pr(gt_lymphocytes, result_detected_lymphocytes,
                                        radius=radius_lymph)
    monocytes_eval = get_froc_vals_pr(gt_monocytes, result_detected_monocytes,
                                      radius=radius_mono)
    inflamm_eval = get_froc_vals_pr(gt_inf_cells, result_detected_inflammatory_cells, radius=radius_infl)

    report += "Lymphocytes eval:\n" + pformat({k: v for k, v in lymphocytes_eval.items() if type(v) is not list}) + "\n"
    report += "Monocytes eval:\n" + pformat({k: v for k, v in monocytes_eval.items() if type(v) is not list}) + "\n"
    report += "Inflammatory cells eval:\n" + pformat({k: v for k, v in inflamm_eval.items() if type(v) is not list}) + "\n"

    print(report)

    # Finally, calculate by comparing the ground truth to the actual results
    return (file_id, {
        'lymphocytes': lymphocytes_eval,
        'monocytes': monocytes_eval,
        'inflammatory-cells': inflamm_eval
    })


def get_froc_vals_pr(gt_dict, result_dict, radius: int):
    """
    Computes the Free-Response Receiver Operating Characteristic (FROC) values for given ground truth and result data.
    Using https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing FROC metrics such as sensitivity, false positives per mm²,
              true positive probabilities, false positive probabilities, total positives,
              area in mm², and FROC score.
    """
    # in case there are no predictions
    if len(result_dict['points']) == 0:
        return {'sensitivity_slide': [0], 'fp_per_mm2_slide': [0], 'fp_probs_slide': [0],
                'tp_probs_slide': [0], 'total_pos_slide': 0, 'area_mm2_slide': 0, 'froc_score_slide': 0}
    if len(gt_dict['points']) == 0:
        return {}

    gt_coords = [i['point'] for i in gt_dict['points']]
    gt_rois = [i['polygon'] for i in gt_dict['rois']]
    # compute the area of the polygon in roi
    area_mm2 = SPACING_LEVEL0 * SPACING_LEVEL0 * gt_dict["area_rois"] / 1000000
    result_prob = [i['probability'] for i in result_dict['points']]
    result_coords = [[i['point'][0], i['point'][1]] for i in result_dict['points']]

    # prepare the data for the FROC curve computation with monai
    true_positives, false_negatives, false_positives, tp_probs, fp_probs = match_coordinates(gt_coords, result_coords,
                                                                                             result_prob, radius)
    total_pos = len(gt_coords)

    pr_40 = compute_precision_recall_threshold(tp_probs, fp_probs, total_pos, threshold=0.4)
    pr_90 = compute_precision_recall_threshold(tp_probs, fp_probs, total_pos, threshold=0.9)

    # the metric is implemented to normalize by the number of images, we however want to have it by mm2, so we set
    # num_images = ROI area in mm2
    sensitivity, fp_per_mm2_slide, froc_score = get_froc_score(fp_probs, tp_probs, total_pos, area_mm2)

    return {'sensitivity_slide': list(sensitivity), 'fp_per_mm2_slide': list(fp_per_mm2_slide),
            'fp_probs_slide': list(fp_probs), 'tp_probs_slide': list(tp_probs), 'total_pos_slide': total_pos,
            'area_mm2_slide': area_mm2, 'froc_score_slide': float(froc_score),
            'presicion_recall_threshold=0_4_slide': pr_40, 'presicion_recall_threshold=0_9_slide': pr_90,}


def compute_precision_recall_threshold(tp_probs, fp_probs, nb_gt, threshold = 0.5):
    # threshold_pred = min(min(tp_probs), min(fp_probs))
    y_score_tp_fp = np.concatenate([tp_probs, fp_probs])
    y_true_tp_fp = [1] * len(tp_probs) + [0] * len(fp_probs)

    # Compute probabilities for true positives and false positives
    threshold_filter = y_score_tp_fp > threshold
    y_true_tp_fp_filtered = np.array(y_true_tp_fp)[threshold_filter]
    y_score_tp_fp_filtered = y_score_tp_fp[threshold_filter]

    tp = sum(y_true_tp_fp_filtered)
    fp = len(y_score_tp_fp_filtered) - tp
    fn = nb_gt - tp

    # compute the precision and recall values
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def match_coordinates(ground_truth, predictions, pred_prob, margin):
    """
    Faster matching using cKDTree for nearest neighbor search.

    Args:
        ground_truth (list of tuples): GT points (x, y).
        predictions (list of tuples): Predicted points (x, y).
        pred_prob (list of floats): Probabilities of predictions.
        margin (float): Max distance allowed for matching.

    Returns:
        true_positives (int), false_negatives (int), false_positives (int),
        tp_probs (np.array), fp_probs (np.array)
    """

    if len(ground_truth) == 0 and len(predictions) == 0:
        return 0, 0, 0, np.array([]), np.array([])

    if len(predictions) == 0:
        return 0, len(ground_truth), 0, np.array([]), np.array([])

    if len(ground_truth) == 0:
        return 0, 0, len(predictions), np.array([]), np.array(pred_prob)

    # Build KDTree from predictions
    pred_array = np.array(predictions)
    pred_prob_array = np.array(pred_prob)
    pred_tree = cKDTree(pred_array)

    # Query ground truth points against prediction tree
    gt_array = np.array(ground_truth)
    distances, indices = pred_tree.query(gt_array, distance_upper_bound=margin)

    matched_pred_indices = set()
    tp_probs = []

    for dist, pred_idx in zip(distances, indices):
        if pred_idx < len(pred_array) and dist <= margin:
            if pred_idx not in matched_pred_indices:
                matched_pred_indices.add(pred_idx)
                tp_probs.append(pred_prob_array[pred_idx])

    true_positives = len(matched_pred_indices)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(predictions) - true_positives

    # Get the probs for false positives
    fp_probs = [pred_prob_array[i] for i in range(len(predictions)) if i not in matched_pred_indices]

    return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs)


def get_froc_score(fp_probs, tp_probs, total_pos, area_mm2):
    eval_thresholds = (10, 20, 50, 100, 200, 300)

    fp_per_mm2, sensitivity = mm.compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    if len(fp_per_mm2) == 0 and len(sensitivity) == 0:
        return sensitivity, fp_per_mm2, 0
    if len(sensitivity) == 1:
        # we only have one true positive point, we have to compute the FROC values a bit differently
        sensitivity = [1]
        fp_per_mm2 = [len(fp_probs)/area_mm2]
        froc_score = np.mean([int(fp_per_mm2[0] < i) for i in eval_thresholds])
    else:
        # area_under_froc = auc(fp_per_mm2, sensitivity)
        froc_score = mm.compute_froc_score(fp_per_mm2, sensitivity, eval_thresholds=eval_thresholds)

    return sensitivity, fp_per_mm2, froc_score


def main():
    print_inputs()
    predictions = read_predictions()
    metrics = {}

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    max_workers = get_max_workers()
    with Pool(processes=max_workers) as pool:
        results = pool.map(process, predictions)
    file_ids = [r[0] for r in results]
    metrics_per_slide = [r[1] for r in results]
    metrics['per_slide'] = {file_id: metrics_per_slide[i] for i, file_id in enumerate(file_ids)}

    # We have the results per prediction, we can aggregate over the results and
    # generate an overall score(s) for this submission
    lymphocytes_metrics = format_metrics_for_aggr(metrics_per_slide, 'lymphocytes')
    monocytes_metrics = format_metrics_for_aggr(metrics_per_slide, 'monocytes')
    inflammatory_cells_metrics = format_metrics_for_aggr(metrics_per_slide, 'inflammatory-cells')
    aggregated_metrics = {
        'lymphocytes': get_aggr_froc(lymphocytes_metrics),
        'monocytes': get_aggr_froc(monocytes_metrics),
        'inflammatory-cells': get_aggr_froc(inflammatory_cells_metrics)
    }

    # clean up the per-file metrics
    for file_id, file_metrics in metrics['per_slide'].items():
        for cell_type in ['lymphocytes', 'monocytes', 'inflammatory-cells']:
            for i in ['fp_probs_slide', 'tp_probs_slide', 'total_pos_slide']:
                if i in file_metrics[cell_type]:
                    del file_metrics[cell_type][i]

    # Aggregate the metrics_per_slide
    metrics["aggregates"] = aggregated_metrics

    # Split the metrics into metrics.json and monkey-evaluation-details.json
    metrics_leaderboard = {}
    for cell_type in ['lymphocytes', 'monocytes', 'inflammatory-cells']:
        metrics_leaderboard[cell_type] = {
            'froc_score': aggregated_metrics[cell_type]['froc_score_aggr'],
            'presicion_threshold=0_4': aggregated_metrics[cell_type]['presicion_recall_threshold=0_4_aggr'][0],
            'recall_threshold=0_4': aggregated_metrics[cell_type]['presicion_recall_threshold=0_4_aggr'][1],
            'presicion_threshold=0_9': aggregated_metrics[cell_type]['presicion_recall_threshold=0_9_aggr'][0],
            'recall_threshold=0_9': aggregated_metrics[cell_type]['presicion_recall_threshold=0_9_aggr'][1]
        }
    write_metrics(metrics=metrics_leaderboard)
    write_extended_evaluations(dictionary=metrics)

    return 0


def get_aggr_froc(metrics_dict):
    if len(metrics_dict) == 0:
        return {'sensitivity_aggr': [0], 'fp_aggr': [0], 'area_mm2_aggr': 0, 'froc_score_aggr': 0}
    # https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    fp_probs = np.array([item for sublist in metrics_dict['fp_probs_slide'] for item in sublist])
    tp_probs = np.array([item for sublist in metrics_dict['tp_probs_slide'] for item in sublist])
    total_pos = sum(metrics_dict['total_pos_slide'])
    area_mm2 = sum(metrics_dict['area_mm2_slide'])
    if total_pos == 0:
        return {'sensitivity_aggr': [0], 'fp_per_mm2_aggr': [0], 'area_mm2_aggr': area_mm2, 'froc_score_aggr': 0}

    # compute precision and recall at 0.5 and 0.9
    pr_40 = compute_precision_recall_threshold(tp_probs, fp_probs, total_pos, threshold=0.4)
    pr_90 = compute_precision_recall_threshold(tp_probs, fp_probs, total_pos, threshold=0.9)

    # sensitivity, fp_overall = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    sensitivity_overall, fp_per_mm2, froc_score_overall = get_froc_score(fp_probs, tp_probs, total_pos, area_mm2)

    return {'sensitivity_aggr': list(sensitivity_overall), 'fp_per_mm2_aggr': list(fp_per_mm2),
            'area_mm2_aggr': area_mm2, 'froc_score_aggr': float(froc_score_overall),
            'presicion_recall_threshold=0_4_aggr': pr_40, 'presicion_recall_threshold=0_9_aggr': pr_90}

    # return {'area_mm2_aggr': area_mm2,
    #         'froc_score_aggr': float(froc_score_overall)}


def format_metrics_for_aggr(metrics_list, cell_type):
    """
    Formats the metrics dictionary to be used in the aggregation function.
    """
    aggr = {}
    for d in [i[cell_type] for i in metrics_list]:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the collapsed_dict, initialize it with an empty list
            if key not in aggr:
                aggr[key] = []
            # Append the value to the list corresponding to the key
            # Remove None values in list
            if type(value) is list:
                value = [x for x in value if x is not None]
            aggr[key].append(value)

    return aggr


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    print(INPUT_DIRECTORY)
    with open("test/predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def convert_mm_to_pixel(data_dict, spacing=SPACING_LEVEL0):
    # Converts a distance in mm to pixels: coord in mm * 1000 * spacing
    points_pixels = []
    for d in data_dict['points']:
        if len(d['point']) == 2:
            d['point'] = [mm_to_pixel(d['point'][0]), mm_to_pixel(d['point'][1]), 0]
        else:
            d['point'] = [mm_to_pixel(d['point'][0]), mm_to_pixel(d['point'][1]), mm_to_pixel(d['point'][2])]
        points_pixels.append(d)
    data_dict['points'] = points_pixels
    return data_dict


def mm_to_pixel(dist, spacing=SPACING_LEVEL0):
    spacing = spacing / 1000
    dist_px = int(round(dist / spacing))
    return dist_px


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


def write_extended_evaluations(*, dictionary):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "monkey-evaluation-details.json", "w") as f:
        f.write(json.dumps(dictionary, indent=4))



if __name__ == "__main__":
    raise SystemExit(main())
