#!/usr/bin/env python3
"""
match_object.py
================

Find a reference object (given as a small cut-out image) inside a larger
scene image, regardless of the object's scale, rotation, or mirroring
(left/right flip) in the scene.

Approach
--------
Plain OpenCV `cv2.matchTemplate` only works when the object in the scene has
(roughly) the same scale and orientation as the template, so it is not used
here. Instead this script uses *feature based* matching:

1. Detect SIFT keypoints/descriptors in the reference image and in the scene.
   SIFT descriptors are invariant to uniform scaling and in-plane rotation,
   which gives us scale/rotation invariance "for free".
2. SIFT descriptors are NOT invariant to mirroring, so the reference image is
   also matched in a horizontally-flipped version. Whichever version (normal
   or flipped) produces more/better matches tells us whether the object was
   mirrored in the scene.
3. Good matches (Lowe's ratio test) are passed to `skimage`'s robust
   `ransac` estimator fitting a `SimilarityTransform` (scale + rotation +
   translation). This directly gives us the scale factor, rotation angle and
   translation that map the reference onto the object found in the scene,
   while being robust to outlier matches.
4. The estimated transform is used to (a) draw the detected object's outline
   in the scene and (b) warp/rectify the region back out of the scene so it
   is oriented exactly like the original reference image (including
   un-mirroring it if necessary).

Requirements
------------
    pip install opencv-contrib-python scikit-image numpy

    (opencv-contrib-python or opencv-python >= 4.4 both work, SIFT has been
    patent-free and included in the main `opencv-python` package since then;
    opencv-contrib-python is a safe choice that always includes it.)

Usage
-----
    python match_object.py reference.png scene.png \
        --output result.png --rectified-output rectified.png

Output
------
* Prints the estimated scale, rotation (degrees), translation and whether
  the object was mirrored.
* Saves `result.png`: the scene with the found object outlined.
* Saves `rectified.png`: the found object warped/cropped out of the scene
  and re-oriented to match the reference image's orientation.
"""

import argparse

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import SimilarityTransform, warp
from typing import Optional
from pathlib import Path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def get_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "cv2.SIFT_create() is not available. Install a recent OpenCV "
            "build with `pip install --upgrade opencv-contrib-python`."
        )
    return cv2.SIFT.create()


def detect_and_describe(gray_img):
    sift = get_sift()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors


def resize_for_detection(gray_img, max_dim):
    """Downscale gray_img so its longest side is at most max_dim (no-op if it
    already is). Returns (image_used_for_detection, scale), where
    scale = detection_size / original_size -- SIFT builds its pyramid from a
    2x-upsampled copy of its input, so running it directly on a
    multi-thousand-pixel image (e.g. a stitched line-scan capture) can need
    gigabytes for that one buffer alone. SIFT descriptors are scale-invariant,
    so detecting on a smaller copy doesn't hurt matching quality, and the
    scale returned here lets the caller map the resulting transform back onto
    the original full-resolution image."""
    h, w = gray_img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return gray_img, 1.0
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(gray_img, new_size, interpolation=cv2.INTER_AREA), scale


def match_descriptors(desc_ref, desc_scene, ratio=0.75):
    """Lowe's ratio-test matching using a brute-force L2 matcher."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc_ref, desc_scene, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def fit_similarity_ransac(src_pts, dst_pts, residual_threshold=4.0,
                           max_trials=2000, min_samples=2):
    """Robustly fit a similarity transform (scale+rotation+translation)."""
    if len(src_pts) < min_samples:
        return None, None
    model, inliers = ransac(
        (src_pts, dst_pts),
        SimilarityTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )
    return model, inliers


def try_orientation(kp_ref, desc_ref, kp_scene, desc_scene, ratio, residual_threshold):
    """Try to match one version (normal or flipped) of the reference against
    the scene and robustly fit a similarity transform. Returns None if the
    match is too weak to trust."""
    if desc_ref is None or desc_scene is None:
        return None

    matches = match_descriptors(desc_ref, desc_scene, ratio=ratio)
    if len(matches) < 4:
        return None

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches])

    model, inliers = fit_similarity_ransac(
        src_pts, dst_pts, residual_threshold=residual_threshold
    )
    if model is None or inliers is None or inliers.sum() < 4:
        return None

    return {
        "model": model,
        "inlier_count": int(inliers.sum()),
        "match_count": len(matches),
        "inliers": inliers,
    }


def summarize_transform(model, mirrored):
    rotation_deg = np.degrees(model.rotation)
    rotation_deg = (rotation_deg + 180) % 360 - 180  # normalize to [-180, 180]
    return {
        "scale": float(model.scale),
        "rotation_deg": float(rotation_deg),
        "translation": (float(model.translation[0]), float(model.translation[1])),
        "mirrored": mirrored,
    }


def draw_detection(scene_bgr, ref_shape, model):
    """Draw the reference's outline, projected into the scene, plus its
    center point."""
    h, w = ref_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    projected = model(corners)

    out = scene_bgr.copy()
    pts = projected.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    center = projected.mean(axis=0).astype(int)
    cv2.circle(out, tuple(center), 6, (0, 0, 255), -1)

    # little arrow showing the "up" direction of the found object, for a
    # quick visual sanity check of the rotation
    top_mid = ((projected[0] + projected[1]) / 2).astype(int)
    cv2.arrowedLine(out, tuple(center), tuple(top_mid), (255, 0, 0), 2, tipLength=0.2)
    return out


def rectify_object(scene_bgr, model, ref_shape, mirrored):
    """Warp the region of the scene that contains the found object back out
    so that it has the exact size/orientation of the original (unmirrored)
    reference image."""
    h, w = ref_shape[:2]
    # `warp`'s map argument goes output-coords -> input-coords. Our output
    # is reference-sized, our input is the (larger) scene, and `model` maps
    # reference-space -> scene-space, so `model` itself is the correct map
    # here (NOT model.inverse).
    warped = warp(scene_bgr, model, output_shape=(h, w), preserve_range=True)
    warped = warped.astype(scene_bgr.dtype)

    if mirrored:
        # `model` maps the *flipped* reference onto the scene, so the crop
        # we just extracted is still mirrored w.r.t. the original reference.
        # Flip it back once more to match the original orientation.
        warped = cv2.flip(warped, 1)

    return warped


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Find a reference object in a scene, regardless of "
                    "scale, rotation, or mirroring."
    )
    parser.add_argument("reference", help="Path to the reference (cut-out) object image")
    parser.add_argument("scene", help="Path to the scene image to search in")
    parser.add_argument("--output", default="result.png",
                         help="Path to save the annotated scene (default: result.png)")
    parser.add_argument("--rectified-output", default="rectified.png",
                         help="Path to save the found object, re-oriented to match "
                              "the reference (default: rectified.png)")
    parser.add_argument("--ratio", type=float, default=0.75,
                         help="Lowe's ratio test threshold (default: 0.75)")
    parser.add_argument("--residual-threshold", type=float, default=4.0,
                         help="RANSAC inlier distance threshold in pixels (default: 4.0)")
    parser.add_argument("--max-dim", type=int, default=2000,
                         help="Downscale images so their longest side is at most this many "
                              "pixels before running SIFT (default: 2000). SIFT is "
                              "scale-invariant, so this only trades off speed/memory, not "
                              "match quality -- the found object is still drawn/rectified at "
                              "the original full resolution. Raise this if matching fails on "
                              "small or low-detail objects; set to a huge value to disable.")
    return parser


def findObject(reference=None, scene=None, output:Optional[str|Path]="result.png",
         rectified_output:Optional[str|Path]="rectified.png", ratio=0.75,
         residual_threshold=4.0, max_dim=2000):
    """Find `reference` in `scene`; save the annotated scene and rectified crop.

    Callable directly, e.g. ``main(reference="ref.png", scene="scene.png")``, or run
    as a CLI script with no arguments, in which case they're parsed from ``sys.argv``
    instead of the keyword defaults above.

    Returns:
        dict: scale/rotation/translation/mirrored info (see `summarize_transform`),
        plus the `output` and `rectified_output` paths written to.
    """
    if reference is None and scene is None:
        args = _build_arg_parser().parse_args()
        reference = args.reference
        scene = args.scene
        output = args.output
        rectified_output = args.rectified_output
        ratio = args.ratio
        residual_threshold = args.residual_threshold
        max_dim = args.max_dim
    elif reference is None or scene is None:
        raise TypeError(
            "main() requires both 'reference' and 'scene' when called directly "
            "(or neither, to parse them from the command line)."
        )

    ref_bgr = load_image(reference)
    scene_bgr = load_image(scene)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    # Detect on downscaled copies (see resize_for_detection), then rescale the
    # fitted transform back to full resolution below so drawing/rectifying
    # still operate on the original images.
    scene_gray_small, scene_scale = resize_for_detection(scene_gray, max_dim)
    ref_gray_small, ref_scale = resize_for_detection(ref_gray, max_dim)
    # ref_flipped_gray_small = cv2.flip(ref_gray_small, 1)

    kp_scene, desc_scene = detect_and_describe(scene_gray_small)
    if desc_scene is None or len(kp_scene) == 0:
        raise RuntimeError("No features could be detected in the scene image.")

    kp_ref, desc_ref = detect_and_describe(ref_gray_small)
    # kp_ref_flip, desc_ref_flip = detect_and_describe(ref_flipped_gray_small)

    # if desc_ref is None and desc_ref_flip is None:
    if desc_ref is None:
        raise RuntimeError("No features could be detected in the reference image.")

    candidates = []

    result_normal = try_orientation(kp_ref, desc_ref, kp_scene, desc_scene,
                                     ratio, residual_threshold)
    if result_normal is not None:
        result_normal["mirrored"] = False
        candidates.append(result_normal)

    # result_flipped = try_orientation(kp_ref_flip, desc_ref_flip, kp_scene, desc_scene,
    #                                   ratio, residual_threshold)
    # if result_flipped is not None:
    #     result_flipped["mirrored"] = True
    #     candidates.append(result_flipped)

    if not candidates:
        raise RuntimeError(
            "Could not find the reference object in the scene (not enough reliable matches)."
        )

    best = max(candidates, key=lambda r: r["inlier_count"])
    mirrored = best["mirrored"]

    # best["model"] maps downscaled-reference -> downscaled-scene coordinates
    # (both detection passes ran on the resized copies above). Compose with
    # pure-scale transforms on each side to get the equivalent mapping
    # between the *original* full-resolution images.
    full_model = (
        SimilarityTransform(scale=ref_scale)
        + best["model"]
        + SimilarityTransform(scale=1.0 / scene_scale)
    )
    info = summarize_transform(full_model, mirrored)

    print("=== Object found ===")
    print(f"Inlier / total matches : {best['inlier_count']} / {best['match_count']}")
    print(f"Scale (scene / reference) : {info['scale']:.4f}x")
    print(f"Rotation                  : {info['rotation_deg']:.2f} degrees "
          "(counter-clockwise, reference -> scene)")
    print(f"Translation (x, y)        : "
          f"({info['translation'][0]:.1f}, {info['translation'][1]:.1f}) px")
    print(f"Mirrored                  : "
          f"{'Yes (horizontally flipped)' if mirrored else 'No'}")


    if output is not None:
        annotated = draw_detection(scene_bgr, ref_bgr.shape, full_model)
        cv2.imwrite(output, annotated)
        print(f"\nAnnotated scene saved to: {output}")

    if rectified_output is not None:
        rectified = rectify_object(scene_bgr, full_model, ref_bgr.shape, mirrored)
        cv2.imwrite(rectified_output, rectified)
        print(f"Rectified (re-oriented) object saved to: {rectified_output}")

    return {**info, "output": output, "rectified_output": rectified_output}


if __name__ == "__main__":
    findObject()