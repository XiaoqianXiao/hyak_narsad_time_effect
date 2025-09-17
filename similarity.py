# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

import numpy as np
from nilearn.image import index_img, load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.input_data import NiftiMasker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import nibabel as nib
from joblib import Parallel, delayed
import os
import re
import logging

logger = logging.getLogger(__name__)


def searchlight_similarity(bold_4d, mask_img, radius=6, trial_pairs=None, similarity='pearson', n_jobs=12,
                           batch_size=1000):
    """
    Compute voxel-wise similarity for multiple trial pairs using a cubic searchlight approach.

    Parameters:
        bold_4d: nib.Nifti1Image - 4D BOLD image (x, y, z, trials)
        mask_img: nib.Nifti1Image - binary mask image
        radius: int - half the side length of the cube in mm (cube side = 2 * radius)
        trial_pairs: list of (i, j) - trial index pairs to compare
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for voxel processing
        batch_size: int - number of voxels to process per batch

    Returns:
        list of (i, j, nib.Nifti1Image) - similarity maps for each trial pair
    """
    logger.info(
        f"Starting searchlight similarity with cube side={2 * radius}mm, similarity={similarity}, n_jobs={n_jobs}, {len(trial_pairs)} pairs")
    try:
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        logger.info(f"Masker fitted, mask shape: {masker.mask_img_.shape}")
        bold_data = masker.transform(bold_4d)  # Shape: (n_voxels, n_trials)
        logger.info(f"Transformed BOLD shape: {bold_data.shape}")
    except Exception as e:
        logger.error(f"Error in masker setup or transform: {e}")
        raise

    coordinates = np.argwhere(masker.mask_img_.get_fdata() > 0)
    logger.info(f"Number of voxels to process: {len(coordinates)}")
    affine = mask_img.affine
    voxel_size = np.abs(affine[0, 0])  # Assuming isotropic voxels
    half_side_voxels = int(np.round(radius / voxel_size))

    def compute_batch_similarity(coords, bold_data, bold_4d, mask_img, half_side_voxels, trial_pairs, batch_idx,
                                 total_batches):
        """
        Compute similarities for a batch of voxels across all trial pairs.
        """
        try:
            img_shape = bold_4d.shape[:3]
            mask_data = mask_img.get_fdata()
            batch_results = {pair: [] for pair in trial_pairs}
            bold_4d_data = bold_4d.get_fdata()

            for voxel_num, coord in enumerate(coords, 1):
                x, y, z = coord
                x_min, x_max = max(0, x - half_side_voxels), x + half_side_voxels + 1
                y_min, y_max = max(0, y - half_side_voxels), y + half_side_voxels + 1
                z_min, z_max = max(0, z - half_side_voxels), z + half_side_voxels + 1
                x_max = min(x_max, img_shape[0])
                y_max = min(y_max, img_shape[1])
                z_max = min(z_max, img_shape[2])

                mask_cube = mask_data[x_min:x_max, y_min:y_max, z_min:z_max]
                valid_voxels = mask_cube > 0
                n_voxels = np.sum(valid_voxels)
                if n_voxels < 2:
                    for pair in trial_pairs:
                        batch_results[pair].append(np.nan)
                    continue

                for i, j in trial_pairs:
                    img1_cube = bold_4d_data[x_min:x_max, y_min:y_max, z_min:z_max, i][valid_voxels].ravel()
                    img2_cube = bold_4d_data[x_min:x_max, y_min:y_max, z_min:z_max, j][valid_voxels].ravel()
                    if len(img1_cube) < 2 or np.any(np.isnan(img1_cube)) or np.any(np.isnan(img2_cube)):
                        batch_results[(i, j)].append(np.nan)
                        continue
                    if similarity == 'pearson':
                        sim = pearsonr(img1_cube, img2_cube)[0]
                    elif similarity == 'cosine':
                        sim = cosine_similarity(img1_cube.reshape(1, -1), img2_cube.reshape(1, -1))[0, 0]
                    else:
                        raise ValueError("similarity must be 'pearson' or 'cosine'")
                    batch_results[(i, j)].append(sim)

            logger.info(f"Processed batch {batch_idx}/{total_batches}, {len(coords)} voxels")
            return batch_results
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            return {pair: [np.nan] * len(coords) for pair in trial_pairs}

    # Split coordinates into batches
    total_voxels = len(coordinates)
    batches = [coordinates[i:i + batch_size] for i in range(0, total_voxels, batch_size)]
    total_batches = len(batches)
    logger.info(f"Processing {total_voxels} voxels in {total_batches} batches of {batch_size} voxels")

    # Process batches in parallel
    batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_batch_similarity)(batch_coords, bold_data, bold_4d, mask_img, half_side_voxels, trial_pairs,
                                          idx + 1, total_batches)
        for idx, batch_coords in enumerate(batches)
    )

    # Aggregate results into similarity maps
    similarity_maps = {pair: np.full(mask_img.shape, np.nan, dtype=np.float32) for pair in trial_pairs}
    for batch_idx, batch_result in enumerate(batch_results):
        batch_coords = batches[batch_idx]
        for pair in trial_pairs:
            for i, coord in enumerate(batch_coords):
                similarity_maps[pair][tuple(coord)] = batch_result[pair][i]

    # Convert to Nifti images
    results = [(i, j, nib.Nifti1Image(similarity_maps[(i, j)], mask_img.affine)) for i, j in trial_pairs]
    logger.info(
        f"Computed {len(results)} similarity maps, skipped {sum(np.isnan(similarity_maps[trial_pairs[0]]).ravel())} voxels")
    return results


def roi_similarity(bold_4d, atlas_img, roi_labels, trial_pairs, similarity='pearson', n_jobs=4):
    """
    Compute pairwise ROI similarities for multiple trial pairs from a 4D BOLD image.

    Parameters:
        bold_4d: nib.Nifti1Image - 4D BOLD image (x, y, z, trials)
        atlas_img: nib.Nifti1Image - labeled atlas image (ROIs > 0)
        roi_labels: list - list of valid ROI labels
        trial_pairs: list of (i, j) - trial index pairs to compare
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for ROI pairs

    Returns:
        list of (i, j, np.ndarray) - similarity matrices (n_rois, n_rois) for each trial pair
    """
    logger.info(f"Starting ROI similarity with {len(roi_labels)} ROIs, {len(trial_pairs)} trial pairs, similarity={similarity}, n_jobs={n_jobs}")
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, detrend=False)
        bold_ts = masker.fit_transform(bold_4d)  # Shape: (n_voxels, n_trials)
        logger.info(f"ROI time-series shape: {bold_ts.shape}")
    except Exception as e:
        logger.error(f"Error in ROI masker setup or transform: {e}")
        raise

    n_rois = len(roi_labels)
    # Validate ROIs
    valid_rois = []
    for i in range(n_rois):
        if bold_ts.shape[0] < 2 or np.all(bold_ts[:, i] == 0) or np.any(np.isnan(bold_ts[:, i])):
            logger.warning(f"ROI {i} (label {roi_labels[i]}) has invalid data: shape={bold_ts[:, i].shape}, all zeros={np.all(bold_ts[:, i] == 0)}, NaNs={np.any(np.isnan(bold_ts[:, i]))}")
            continue
        valid_rois.append(i)
    logger.info(f"Valid ROIs: {len(valid_rois)}/{n_rois}")

    def compute_trial_pair(i, j, bold_ts, valid_rois, similarity, pair_num, total_pairs):
        try:
            sim_matrix = np.full((n_rois, n_rois), np.nan, dtype=np.float32)
            for ri, rj in [(ri, rj) for ri in valid_rois for rj in valid_rois]:
                ts1 = bold_ts[:, ri]
                ts2 = bold_ts[:, rj]
                if np.any(np.isnan(ts1)) or np.any(np.isnan(ts2)) or len(ts1) < 2 or len(ts2) < 2:
                    logger.debug(f"Trial pair {pair_num}/{total_pairs} ({i} vs {j}), ROI {ri} vs {rj} skipped: invalid data")
                    continue
                if np.all(ts1 == 0) or np.all(ts2 == 0):
                    logger.debug(f"Trial pair {pair_num}/{total_pairs} ({i} vs {j}), ROI {ri} vs {rj} skipped: all zeros")
                    continue
                if similarity == 'pearson':
                    sim = pearsonr(ts1, ts2)[0]
                elif similarity == 'cosine':
                    sim = cosine_similarity(ts1.reshape(1, -1), ts2.reshape(1, -1))[0, 0]
                else:
                    raise ValueError("similarity must be 'pearson' or 'cosine'")
                sim_matrix[ri, rj] = sim
            logger.debug(f"Computed trial pair {pair_num}/{total_pairs} ({i} vs {j})")
            return (i, j, sim_matrix)
        except Exception as e:
            logger.error(f"Error computing trial pair {i} vs {j} ({pair_num}/{total_pairs}): {e}")
            return (i, j, np.full((n_rois, n_rois), np.nan, dtype=np.float32))

    logger.info(f"Computing {len(trial_pairs)} trial pairs")
    pair_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_trial_pair)(i, j, bold_ts, valid_rois, similarity, idx + 1, len(trial_pairs))
        for idx, (i, j) in enumerate(trial_pairs)
    )

    logger.info(f"Computed {len(pair_results)} trial pair similarity matrices")
    return pair_results


def load_roi_names(names_file_path: str, roi_labels: list) -> dict:
    """
    Load ROI names from a file with alternating lines.
    """
    logger.info(f"Loading ROI names from {names_file_path}")

    def format_name(name: str) -> str:
        s = name.strip()
        m = re.match(r"^(.+)-(rh|lh)$", s, flags=re.IGNORECASE)
        if m:
            region, hemi = m.group(1), m.group(2).lower()
            return f"{hemi}_{region}"
        m = re.match(r"^7Networks_(LH|RH)_(.+)$", s, flags=re.IGNORECASE)
        if m:
            hemi = m.group(1).lower()
            rest = m.group(2)
            m_idx = re.match(r"^(.*)_(\d+)$", rest)
            if m_idx:
                base, idx = m_idx.group(1), m_idx.group(2)
                return f"{hemi}_{base}-{idx}"
            return f"{hemi}_{rest}"
        return s

    default_names = {int(l): f"combined_ROI_{int(l)}" for l in roi_labels if l is not None}
    if not os.path.exists(names_file_path):
        logger.warning(f"ROI names file not found: {names_file_path}. Using numerical labels.")
        return default_names

    intlabel_to_rawname = {}
    try:
        with open(names_file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                logger.warning(f"Incomplete pair at line {i + 1}: missing label data")
                continue
            name_line = lines[i]
            nums = lines[i + 1].split()
            if not nums:
                logger.warning(f"Empty label line at {i + 2}")
                continue
            try:
                label_int = int(nums[0])
                intlabel_to_rawname[label_int] = name_line
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid label format at line {i + 2}: {lines[i + 1]} - {e}")
                continue
    except Exception as e:
        logger.error(f"Error reading ROI names file {names_file_path}: {e}. Using numerical labels.")
        return default_names

    roi_names = {}
    for lab in roi_labels:
        try:
            lab_int = int(lab)
            raw = intlabel_to_rawname.get(lab_int)
            roi_names[lab_int] = format_name(raw) if raw is not None else f"combined_ROI_{lab_int}"
        except (ValueError, TypeError):
            logger.warning(f"Invalid ROI label: {lab}. Skipping.")
            continue

    if not roi_names:
        logger.warning("No valid ROI names loaded. Using numerical labels.")
        return default_names

    logger.info(f"Loaded {len(roi_names)} ROI names. Example: {list(roi_names.items())[:10]}")
    return roi_names


def get_roi_labels(atlas_img, atlas_name):
    logger.info(f"Extracting ROI labels from {atlas_name}")
    atlas_data = atlas_img.get_fdata()
    roi_labels = np.unique(atlas_data)[np.unique(atlas_data) > 0]
    if len(roi_labels) == 0:
        logger.error(f"No valid ROIs found in {atlas_name} atlas (all values <= 0)")
        raise ValueError(f"No valid ROIs found in {atlas_name} atlas")
    logger.info(f"Found {len(roi_labels)} ROI labels")
    return roi_labels
