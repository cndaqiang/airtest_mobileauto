#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################
# Author : cndaqiang             #
# Update : 2025-01-13            #
# Build  : 2025-01-13            #
# What   : ManyExists 多图并行识别   #
#################################

"""
Optimized multiple image detection module for Airtest.
Provides manyexists function that takes a single screenshot and checks multiple templates against it.

This module combines and optimizes code from the following Airtest functions:
- airtest.core.cv.loop_find: Parameter design (timeout, threshold, interval, intervalfunc)
- airtest.core.cv.Template.match_in: Template matching logic for each image
- G.DEVICE.snapshot: Screenshot capture mechanism
- airtest.core.settings.Settings (ST): Configuration constants (FIND_TIMEOUT, SNAPSHOT_QUALITY)

Unlike calling exists() multiple times (which takes one screenshot per template),
manyexists takes a single screenshot and checks all templates against it,
significantly improving performance when checking multiple images.
"""

import concurrent.futures
import time
from typing import List, Union, Optional, Tuple
import numpy as np

from airtest.core.cv import Template
from airtest.core.helper import G, logwrap
from airtest.core.settings import Settings as ST
from airtest.utils.transform import TargetPos


def _match_template_single(template: Template, screen: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Match a single template against a screen.

    Args:
        template: Template object to match
        screen: Screenshot as numpy array

    Returns:
        Match result - None if not found or (x, y) coordinates if found
    """
    try:
        match_result = template.match_in(screen)
        return match_result
    except Exception as e:
        G.LOGGING.debug(f"Template matching failed for {template}: {e}")
        return None




@logwrap
def manyexists(templates: List[Template],
               screen: Optional[np.ndarray] = None,
               timeout: float = ST.FIND_TIMEOUT,
               threshold: Optional[float] = None,
               interval: float = 0.5,
               intervalfunc: Optional[callable] = None,
               use_concurrent: bool = False,
               max_workers: Optional[int] = None,
               return_screen: bool = True) -> Union[List[Optional[Tuple[float, float]]],
                                                   Tuple[List[Optional[Tuple[float, float]]], np.ndarray]]:
    """
    Check whether multiple templates exist on device screen with a single screenshot.

    This function optimizes the detection process by taking only one screenshot and
    checking all templates against it, rather than taking a screenshot for each template.

    Args:
        templates: image templates to be found in screenshot
        screen: Optional pre-captured screenshot. If None, will take a new screenshot with timeout
        timeout: time interval how long to look for the screenshot
        threshold: default is None
        interval: sleep interval before next attempt to find the image template
        intervalfunc: function that is executed after unsuccessful attempt to find the image template
        use_concurrent: Whether to use concurrent.futures for parallel template matching.
            Can be slightly faster, but not much faster for only a few images. Default: False
        max_workers: Maximum number of worker threads for concurrent execution
        return_screen: Whether to return the screenshot along with results

    Returns:
        If return_screen is False: List of match results for each template.
            Each result is either False (template not found) or (x, y) coordinates (template found).
        If return_screen is True: Tuple of (results_list, screenshot)

    Examples:
        # Basic usage - check multiple templates with single screenshot (returns results and screen by default)
        templates = [Template("button1.png"), Template("button2.png")]
        results, screenshot = manyexists(templates)

        # With pre-captured screen (only returns results)
        screen = G.DEVICE.snapshot()
        results = manyexists(templates, screen=screen, return_screen=False)

        # Enable concurrent processing for better performance with many templates
        results, screenshot = manyexists(templates, use_concurrent=True, max_workers=4)

        # Disable screen return
        results = manyexists(templates, return_screen=False)

        # Custom timeout for screenshot capture (default is ST.FIND_TIMEOUT = 20 seconds)
        results, screenshot = manyexists(templates, timeout=5.0)

        # Override threshold for all templates
        results, screenshot = manyexists(templates, threshold=0.8)

        # Custom interval for screenshot retry
        results, screenshot = manyexists(templates, interval=0.2)

        # Custom interval function for screenshot retry attempts
        def my_interval_func():
            print("Retrying screenshot...")
        results, screenshot = manyexists(templates, intervalfunc=my_interval_func)
    """
    # Ensure templates is a list first
    if not isinstance(templates, list):
        templates = [templates]

    # Take screenshot if not provided
    if screen is None:
        start_time = time.time()
        screen = None

        # Try to take screenshot until successful or timeout
        while screen is None:
            try:
                screen = G.DEVICE.snapshot(filename=None, quality=ST.SNAPSHOT_QUALITY)
                if screen is None:
                    G.LOGGING.warning("Screen is None, may be locked or device not ready")
            except Exception as e:
                G.LOGGING.warning(f"Failed to take screenshot: {e}")
                screen = None

            # Check timeout - use timeout parameter (default: ST.FIND_TIMEOUT_TMP)
            if (time.time() - start_time) > timeout:
                G.LOGGING.error(f"Failed to take screenshot within {timeout} seconds")
                # Return all False results if screenshot failed
                results = [False] * len(templates)
                if return_screen:
                    return results, screen
                else:
                    return results

            # Execute intervalfunc if provided (like loop_find)
            if intervalfunc is not None:
                intervalfunc()

            # Sleep interval before retry (use interval parameter)
            if screen is None:
                time.sleep(interval)

    # Apply threshold override if provided (like loop_find does)
    if threshold is not None:
        # Store original thresholds to restore later
        original_thresholds = [t.threshold for t in templates]
        for template in templates:
            template.threshold = threshold

    results = []

    if use_concurrent and len(templates) > 1:
        # Use ThreadPoolExecutor for concurrent template matching
        if max_workers is None:
            max_workers = min(len(templates), 4)  # Default to 4 workers max

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all template matching tasks
            futures = [executor.submit(_match_template_single, template, screen) for template in templates]

            # Collect results in order
            results = [future.result() for future in futures]
    else:
        # Sequential processing
        for template in templates:
            match_result = _match_template_single(template, screen)
            results.append(match_result)

    # Convert None results to False for consistency with airtest's exists() function
    results = [False if result is None else result for result in results]

    # Restore original thresholds if they were overridden
    if threshold is not None:
        for template, orig_threshold in zip(templates, original_thresholds):
            template.threshold = orig_threshold

    if return_screen:
        return results, screen
    else:
        return results