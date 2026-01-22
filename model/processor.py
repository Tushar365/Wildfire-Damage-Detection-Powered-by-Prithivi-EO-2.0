"""
GeoTIFF processor for handling 6-band HLS format imagery.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageDraw
import torch
from typing import Tuple, Dict, Optional
import os
from scipy.ndimage import label


class GeoTIFFProcessor:
    """Handles loading, preprocessing, and saving of GeoTIFF files."""

    def __init__(self):
        self.expected_bands = 6

    def load_geotiff(self, file_path: str) -> Dict:
        """
        Load a 6-band GeoTIFF file.

        Args:
            file_path: Path to the GeoTIFF file

        Returns:
            Dictionary containing:
                - data: numpy array of shape (bands, height, width)
                - profile: rasterio profile with metadata
                - transform: affine transform
                - crs: coordinate reference system
        """
        with rasterio.open(file_path) as src:
            # Read all bands
            data = src.read()

            # Validate band count
            if data.shape[0] != self.expected_bands:
                raise ValueError(
                    f"Expected {self.expected_bands} bands, got {data.shape[0]}. "
                    f"Please provide 6-band HLS format imagery."
                )

            return {
                'data': data,
                'profile': src.profile,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'shape': (src.height, src.width)
            }

    def validate_alignment(self, pre_data: Dict, post_data: Dict) -> None:
        """
        Validate that pre and post images are spatially aligned.

        Args:
            pre_data: Pre-event image data
            post_data: Post-event image data

        Raises:
            ValueError: If images are not aligned
        """
        if pre_data['shape'] != post_data['shape']:
            raise ValueError(
                f"Image dimensions don't match: "
                f"pre {pre_data['shape']} vs post {post_data['shape']}"
            )

        # Check CRS
        if pre_data['crs'] != post_data['crs']:
            raise ValueError(
                f"Coordinate reference systems don't match: "
                f"pre {pre_data['crs']} vs post {post_data['crs']}"
            )

        # Check bounds (allowing small floating point differences)
        pre_bounds = pre_data['bounds']
        post_bounds = post_data['bounds']
        tolerance = 1e-6

        for i, (pb, pob) in enumerate(zip(pre_bounds, post_bounds)):
            if abs(pb - pob) > tolerance:
                raise ValueError(
                    f"Image bounds don't match. "
                    f"Images must cover the same geographic area."
                )

    def normalize_reflectance(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize reflectance values to 0-1 range.

        HLS data typically ranges from 0-10000 (scaled reflectance).

        Args:
            data: Input array of shape (bands, height, width)

        Returns:
            Normalized array
        """
        # Typical HLS data is scaled reflectance (0-10000)
        # We use a 99th percentile clip to handle extreme outliers/bright spots
        if data.max() > 1.1:
            # Clip to a reasonable range (0-10000) but also use percentile for robustness
            p99 = np.percentile(data, 99)
            limit = max(10000.0, p99)
            data = np.clip(data, 0, limit) / limit

        return data.astype(np.float32)

    def prepare_model_input(
        self,
        pre_data: np.ndarray,
        post_data: np.ndarray,
        normalize: bool = True,
        target_size: Optional[Tuple[int, int]] = (512, 512)
    ) -> torch.Tensor:
        """
        Prepare stacked temporal input for Prithivi model.

        Args:
            pre_data: Pre-event data (6, H, W)
            post_data: Post-event data (6, H, W)
            normalize: Whether to normalize reflectance values
            target_size: Target (H, W) to resize to (model expects square usually)

        Returns:
            Torch tensor of shape (1, 12, H, W)
        """
        if normalize:
            pre_data = self.normalize_reflectance(pre_data)
            post_data = self.normalize_reflectance(post_data)

        # Convert to torch and add batch dimension
        pre_tensor = torch.from_numpy(pre_data).float().unsqueeze(0)
        post_tensor = torch.from_numpy(post_data).float().unsqueeze(0)

        # Resize to square if requested (Fixes Einops square-guess bug in terratorch)
        if target_size:
            pre_tensor = torch.nn.functional.interpolate(
                pre_tensor, size=target_size, mode='bilinear', align_corners=False
            )
            post_tensor = torch.nn.functional.interpolate(
                post_tensor, size=target_size, mode='bilinear', align_corners=False
            )

        # Stack pre and post temporally (channels first)
        stacked = torch.cat([pre_tensor, post_tensor], dim=1)  # (1, 12, H, W)

        return stacked

    def create_rgb_visualization(
        self,
        data: np.ndarray,
        bands: Tuple[int, int, int] = (3, 2, 1)
    ) -> np.ndarray:
        """
        Create RGB visualization from multispectral data.

        Args:
            data: Input data of shape (bands, height, width)
            bands: Which bands to use for RGB (default: 3,2,1 for natural color)

        Returns:
            RGB array of shape (height, width, 3) with values 0-255
        """
        # Select RGB bands (convert to 0-indexed)
        r = data[bands[0] - 1, :, :]
        g = data[bands[1] - 1, :, :]
        b = data[bands[2] - 1, :, :]

        # Stack into RGB
        rgb = np.stack([r, g, b], axis=-1)

        # Normalize to 0-1
        rgb = self.normalize_reflectance(rgb)

        # Apply contrast stretch (2nd to 98th percentile)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        # Convert to 0-255
        rgb = (rgb * 255).astype(np.uint8)

        return rgb

    def create_change_mask_visualization(
        self,
        mask: np.ndarray,
        colormap: str = 'red'
    ) -> np.ndarray:
        """
        Create colored visualization of change mask.

        Args:
            mask: Binary mask (H, W) with 1 for changed areas
            colormap: Color scheme ('red', 'fire', or 'grayscale')

        Returns:
            RGB array of shape (H, W, 3) with values 0-255
        """
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Normalize to 0-1
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / mask.max()

        # Create RGB visualization
        if colormap == 'red':
            # Red for damage
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[:, :, 0] = (mask * 255).astype(np.uint8)  # Red channel
        elif colormap == 'purple':
            # Purple/Pink for damage (high R and B, low G)
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[:, :, 0] = (mask * 255).astype(np.uint8)  # Red
            rgb[:, :, 1] = (mask * 50).astype(np.uint8)   # Slight Green for pinker tone
            rgb[:, :, 2] = (mask * 255).astype(np.uint8)  # Blue
        elif colormap == 'fire':
            # Improved fire color scheme: Dark Red -> Orange -> Bright Yellow
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            
            # Red channel: quickly saturates
            rgb[:, :, 0] = (np.clip(mask * 1.5, 0, 1) * 255).astype(np.uint8)
            
            # Green channel: creates orange/yellow (slower ramp)
            rgb[:, :, 1] = (np.clip(mask * 1.0, 0, 1) * 255).astype(np.uint8)
            
            # Blue channel: small amount for bright white-hot center if mask is very high
            rgb[:, :, 2] = (np.clip((mask - 0.8) * 5, 0, 1) * 255).astype(np.uint8)
        else:  # grayscale
            gray = (mask * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)

        return rgb

    def save_results(
        self,
        pre_rgb: np.ndarray,
        post_rgb: np.ndarray,
        mask_rgb: np.ndarray,
        mask_geotiff: np.ndarray,
        output_dir: str,
        profile: Dict,
        prefix: str = 'result'
    ) -> Dict[str, str]:
        """
        Save all outputs (PNG visualizations and GeoTIFF mask).

        Args:
            pre_rgb: Pre-event RGB visualization
            post_rgb: Post-event RGB visualization
            mask_rgb: Change mask RGB visualization
            mask_geotiff: Change mask with geospatial metadata
            output_dir: Directory to save outputs
            profile: Rasterio profile for GeoTIFF output
            prefix: Filename prefix

        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)

        paths = {}

        # Save PNG visualizations
        pre_path = os.path.join(output_dir, f'{prefix}_pre.png')
        Image.fromarray(pre_rgb).save(pre_path)
        paths['pre_png'] = pre_path

        post_path = os.path.join(output_dir, f'{prefix}_post.png')
        Image.fromarray(post_rgb).save(post_path)
        paths['post_png'] = post_path

        mask_path = os.path.join(output_dir, f'{prefix}_mask.png')
        Image.fromarray(mask_rgb).save(mask_path)
        paths['mask_png'] = mask_path

        # Save GeoTIFF mask with spatial metadata
        mask_geotiff_path = os.path.join(output_dir, f'{prefix}_mask.tif')

        # Update profile for single-band output
        out_profile = profile.copy()
        out_profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        })

        # Ensure mask is 2D
        if mask_geotiff.ndim > 2:
            mask_geotiff = mask_geotiff.squeeze()

        with rasterio.open(mask_geotiff_path, 'w', **out_profile) as dst:
            dst.write(mask_geotiff.astype(np.float32), 1)

        paths['mask_geotiff'] = mask_geotiff_path

        return paths

    def create_overlay_visualization(
        self,
        base_rgb: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create soft heatmap overlay of change mask on base image.

        Args:
            base_rgb: Base RGB image (H, W, 3)
            mask: Confidence map or binary mask (H, W)
            alpha: Maximum transparency of overlay (0-1)

        Returns:
            RGB array with soft overlay
        """
        # Ensure mask is 2D and float
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = mask.astype(np.float32)

        # Create purple overlay
        # We use a pinkish-purple colormap as requested
        overlay = self.create_change_mask_visualization(mask, colormap='purple')

        # Apply a non-linear boost to make the overlay "deeper" and more visible
        # Power law transformation (gamma < 1 makes dark values brighter)
        boosted_mask = np.power(mask, 0.7)
        
        # Ensure mask is 2D and float
        mask_3d = np.expand_dims(boosted_mask, axis=-1)

        result = (
            base_rgb.astype(np.float32) * (1 - mask_3d * alpha) +
            overlay.astype(np.float32) * mask_3d * alpha
        ).astype(np.uint8)

        return result

    def create_boxed_visualization(
        self,
        base_rgb: np.ndarray,
        mask: np.ndarray,
        padding: int = 5,
        min_pixels: int = 20
    ) -> np.ndarray:
        """
        Create visualization with red squares (bounding boxes) around detected areas.

        Args:
            base_rgb: Base RGB image (H, W, 3)
            mask: Binary mask or confidence map
            padding: Padding around the bounding box in pixels
            min_pixels: Minimum pixel count for a component to get a box

        Returns:
            RGB array with red bounding boxes
        """
        # Ensure mask is binary and 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Binary threshold if it's a confidence map
        binary_mask = (mask > 0.3).astype(np.int32)
        
        # Find connected components
        labeled_array, num_features = label(binary_mask)
        
        # Create a copy of the base image to draw on
        result_img = Image.fromarray(base_rgb.copy())
        draw = ImageDraw.Draw(result_img)
        
        # Draw boxes for each feature
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            pixel_count = component_mask.sum()
            
            if pixel_count < min_pixels:
                continue
                
            # Find coordinates
            rows = np.any(component_mask, axis=1)
            cols = np.any(component_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            ymin = max(0, ymin - padding)
            ymax = min(base_rgb.shape[0] - 1, ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(base_rgb.shape[1] - 1, xmax + padding)
            
            # Draw rectangle (Red, width=3)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=3)
            
        return np.array(result_img)
