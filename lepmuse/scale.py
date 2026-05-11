from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage import color
import numpy as np
from scipy import ndimage as ndi
from joblib import Memory
import matplotlib.patches as patches

from .cache import memory

RULER_TOP = 0.1
RULER_BOT = 0.9
RULER_LEFT = 0.1
RULER_RIGHT = 0.4
FIRST_INDEX_THRESHOLD = 0.9
LINE_WIDTH = 40
# A column/row must contain at least this fraction of the peak column/row
# fill to be considered part of the "dense" ruler strip.  Sparse UNet noise
# pixels far from the physical ruler can inflate the bounding box; trimming
# at fixed percentages of that inflated box misses the actual tick region.
RULER_DENSITY_THRESHOLD = 0.05


def binarize_ruler(ruler_rgb):
    """Returns a binarized version of the image.

    Parameters
    ----------
    ruler_rgb : (M, N) ndarray
        Input image containing the ruler.

    Returns
    -------
    ruler : (M, N) ndarray
        Ruler as a binarized image.

    Notes
    -----
    This performs differently than the U-net; while the U-net returns the
    location of the ruler, this returns the binarized ruler and its elements.
    """
    gray = color.rgb2gray(ruler_rgb)
    thresh = threshold_otsu(gray)
    ruler = gray > thresh

    return ruler


def remove_numbers(focus):
    """Returns a ruler image with the numbers stripped away.

    Parameters
    ----------
    focus : 2D array
        Binary image of the ruler.

    Returns
    -------
    focus_numbers_filled : 2D array
        Binary image of the ruler without numbers.

    Notes
    -----
    The numbers are stripped away to improve the results of the Fourier
    transform, which will process the ruler ticks.
    """
    focus_numbers_markers, _ = ndi.label(focus, ndi.generate_binary_structure(2, 1))
    focus_numbers_regions = regionprops(focus_numbers_markers)
    focus_numbers_region_areas = [region.filled_area for region in focus_numbers_regions]
    focus_numbers_avg_area = np.mean(focus_numbers_region_areas)

    focus_numbers_filled = np.copy(focus)
    for region in focus_numbers_regions:
        if region.eccentricity < 0.99 and region.filled_area > focus_numbers_avg_area:
            min_row, min_col, max_row, max_col = region.bbox
            focus_numbers_filled[min_row:max_row, min_col:max_col] = 0

    return focus_numbers_filled


def fourier(signal, axes=None):
    """Performs a Fourier transform to find the distance in pixels
    between two ticks of the ruler.

    Parameters
    ----------
    signal : 1D array
        Array representing the value of the ticks in space.

    Returns
    -------
    T_space : float
        Distance in pixels between two ticks (0.5 mm).
    """
    signal_thresholded = signal > 0

    if signal_thresholded.sum() == 0:
        raise ValueError(
            "Ruler tick signal is all-zero after binarization. "
            "The ruler crop may be empty, mislocated, or entirely erased by remove_numbers(). "
            "Check that the ruler mask covers the tick-mark area."
        )

    fourier_coefs = np.fft.rfft(signal_thresholded)
    mod = np.abs(fourier_coefs)
    mod[0:10] = 0  # discard DC and very-low-frequency components
    freq = np.fft.rfftfreq(len(signal_thresholded))

    peak_idx = np.argmax(mod)
    f_space = freq[peak_idx]

    if f_space == 0:
        raise ValueError(
            "No dominant tick frequency found in ruler signal (FFT peak at f=0). "
            "The ruler projection axis may be mismatched with ruler orientation — "
            "ensure the ruler mask is aligned with the tick direction."
        )

    T_space = 1.0 / f_space

    if axes and axes[4]:
        axes[4].plot(signal, linewidth=0.5)
        axes[5].axvline(x=f_space, color='r', linestyle='dotted', linewidth=1)
        axes[5].plot(freq, mod, linewidth=0.5)

    return T_space


@memory.cache(ignore=['axes'])
def main(image_rgb, ruler_bin, axes=None):
    """Finds the distance between ticks

    Parameters
    ----------
    image_rgb : array
        array representing the image
    ax : array
        array of Axes that show subplots

    Returns
    -------
    t_space : float
        distance between two ticks (.5 mm)
    """
    # preparing figure.
    if axes and axes[0]:
        axes[0].set_title('Final output')
        axes[0].imshow(image_rgb)
        if axes[3]:
            axes[3].set_title('Image structure')
            axes[4].set_title('Ruler signal')
            axes[5].set_title('Fourier transform of ruler signal')
            axes[3].imshow(image_rgb)

    # detecting the bounding box of the ruler mask.
    ruler_row, ruler_col = np.nonzero(ruler_bin)
    top_ruler = int(ruler_row.min())
    side_ruler = int(ruler_col.min())
    ruler_h = int(ruler_row.max() - ruler_row.min())
    ruler_w = int(ruler_col.max() - ruler_col.min())

    if ruler_h < 5 or ruler_w < 5:
        raise ValueError(
            f"Ruler bounding box too small ({ruler_h}×{ruler_w} px) — "
            "ruler mask may be empty or the specimen dilation erased it."
        )

    # Orientation: horizontal ruler (width >> height) vs vertical (height >= width).
    # Tick marks repeat along the ruler's long axis.  We project onto that axis
    # (sum perpendicular slices) to get a 1-D periodic signal for the FFT.
    is_horizontal = ruler_w > ruler_h
    print(f"Ruler bounding box: {ruler_h}h × {ruler_w}w  →  {'horizontal' if is_horizontal else 'vertical'}")

    r_min, r_max = int(ruler_row.min()), int(ruler_row.max())
    c_min, c_max = int(ruler_col.min()), int(ruler_col.max())
    roi_mask_full = ruler_bin[r_min:r_max, c_min:c_max]

    # ── Locate dense ruler strip ───────────────────────────────────────────────
    # The raw bounding box can be inflated by sparse UNet noise pixels far from
    # the physical ruler.  Fixed-percentage crops of an inflated box will miss
    # the actual tick region entirely.  We find the contiguous "dense" band
    # (rows or columns whose fill exceeds RULER_DENSITY_THRESHOLD × peak fill)
    # and restrict the ROI to that band before any further processing.
    if is_horizontal:
        # For a horizontal ruler tick marks vary along x → locate dense ROWS.
        axis_dens = roi_mask_full.sum(axis=1)          # pixels per row
        dense_idx = np.where(axis_dens > axis_dens.max() * RULER_DENSITY_THRESHOLD)[0]
        dr0, dr1 = int(dense_idx[0]),  int(dense_idx[-1]) + 1
        dc0, dc1 = 0, roi_mask_full.shape[1]
    else:
        # For a vertical ruler tick marks vary along y → locate dense COLUMNS.
        axis_dens = roi_mask_full.sum(axis=0)          # pixels per column
        dense_idx = np.where(axis_dens > axis_dens.max() * RULER_DENSITY_THRESHOLD)[0]
        dc0, dc1 = int(dense_idx[0]),  int(dense_idx[-1]) + 1
        dr0, dr1 = 0, roi_mask_full.shape[0]

    # Absolute image coordinates of the dense strip's origin (used for plotting).
    dense_r_abs = r_min + dr0
    dense_c_abs = c_min + dc0
    dense_h = dr1 - dr0
    dense_w  = dc1 - dc0
    print(f"Dense ruler strip: {dense_h}h × {dense_w}w  abs=({dense_r_abs},{dense_c_abs})")

    # Crop image and mask to the dense strip only, then blank non-ruler pixels.
    # This guarantees Otsu sees only ruler paper vs ruler tick values.
    roi_rgb  = image_rgb[r_min + dr0 : r_min + dr1,
                         c_min + dc0 : c_min + dc1].copy()
    roi_mask = roi_mask_full[dr0:dr1, dc0:dc1]
    roi_rgb[~roi_mask] = 255   # white → False after ~binarize_ruler
    focus = ~binarize_ruler(roi_rgb)

    # Removing the numbers in the ruler to denoise the fourier transform analysis
    focus_numbers_filled = remove_numbers(focus)

    if is_horizontal:
        # Horizontal ruler: trim top/bottom within dense strip, project onto x.
        up_trim    = int(RULER_TOP   * focus_numbers_filled.shape[0])
        down_trim  = int(RULER_BOT   * focus_numbers_filled.shape[0])
        left_focus = int(RULER_LEFT  * focus_numbers_filled.shape[1])
        right_focus= int(RULER_RIGHT * focus_numbers_filled.shape[1])
        focus_numbers_filled = focus_numbers_filled[up_trim:down_trim, left_focus:right_focus]
        means = np.mean(focus_numbers_filled, axis=0)
        first_index = np.argmax(means > FIRST_INDEX_THRESHOLD * means.max())
        sums = np.sum(focus_numbers_filled, axis=0)
        side_ruler = dense_c_abs + left_focus
    else:
        # Vertical ruler: trim top/bottom and left/right within dense strip, project onto y.
        up_trim    = int(RULER_TOP   * focus_numbers_filled.shape[0])
        down_trim  = int(RULER_BOT   * focus_numbers_filled.shape[0])
        left_focus = int(RULER_LEFT  * focus_numbers_filled.shape[1])
        right_focus= int(RULER_RIGHT * focus_numbers_filled.shape[1])
        focus_numbers_filled = focus_numbers_filled[up_trim:down_trim, left_focus:right_focus]
        means = np.mean(focus_numbers_filled, axis=1)
        first_index = np.argmax(means > FIRST_INDEX_THRESHOLD * means.max())
        sums = np.sum(focus_numbers_filled, axis=1)
        side_ruler = dense_c_abs + left_focus

    if focus_numbers_filled.size == 0:
        raise ValueError(
            f"Ruler crop is empty after trimming dense strip ({dense_h}h×{dense_w}w). "
            "Check RULER_TOP/BOT/LEFT/RIGHT constants."
        )

    # Fourier transform analysis to give us the pixels between the 1mm ticks
    t_space = 1 * fourier(sums, axes)
    print("Estimated ruler t-unit space in pixels - {:.2f}".format(t_space))
    if axes and axes[4]:
        axes[4].set_title(f'Ruler signal (t-unit = {t_space:.1f} px)')
    # Single grading on the ruler
    y_single = [side_ruler + first_index,
                side_ruler + first_index + t_space]
    # 10 units of grading on the ruler 
    y_mult = [side_ruler + first_index,
              side_ruler + first_index + t_space * 10]

    # plotting.
    x_side = np.array([y_single[1], y_single[1]])
    if axes and axes[0]:
        axes[0].fill_betweenx(y_single, x_side, x_side + LINE_WIDTH, color='red', linewidth=0, alpha=0.7)
        axes[0].fill_betweenx(y_mult, x_side - LINE_WIDTH, x_side, color='blue', linewidth=0, alpha=0.7)

    if axes and axes[3]:
        rect = patches.Rectangle((side_ruler, dense_r_abs + up_trim),
                                 right_focus,
                                 down_trim,
                                 linewidth=1, edgecolor='red', facecolor='none')
        axes[3].axvline(x=side_ruler, color='blue', linestyle='dashed')
        axes[3].add_patch(rect)

    return t_space, side_ruler
