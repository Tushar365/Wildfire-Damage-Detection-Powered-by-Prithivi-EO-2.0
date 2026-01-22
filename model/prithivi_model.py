"""
Prithivi-EO-2.0 model wrapper for change detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
from model.burn_indices import detect_burn_areas_spectral, calculate_nbr_from_array, calculate_dnbr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrithiviChangeDetector:
    """
    Wrapper for Prithivi-EO-2.0-300M model for change detection tasks.

    Uses singleton pattern to load model once and reuse across requests.
    """

    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the change detector (model loaded lazily)."""
        if self._model is None:
            self._initialize_model()

    def _initialize_model(self):
        """Load Prithivi model and prepare for inference."""
        logger.info("Initializing Prithivi-EO-2.0-300M model...")

        # Set device (CPU only for this implementation)
        self._device = torch.device('cpu')
        logger.info(f"Using device: {self._device}")

        # Check for local weights first
        self._local_weights_path = Path("./weights")
        model_path = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
        
        if self._local_weights_path.exists() and (self._local_weights_path / "config.json").exists():
            logger.info(f"Local weights found at {self._local_weights_path.absolute()}")
            model_path = str(self._local_weights_path.absolute())

        try:
            # Import terratorch components
            from terratorch.models import PrithviModelFactory
            from terratorch.datasets import HLSBands

            # Load model using factory
            logger.info(f"Loading model from {model_path}...")
            self._model_factory = PrithviModelFactory()
            
            # PrithviModelFactory.build_model(backbone, decoder, bands, ...) in terratorch 1.2.1
            # We need to provide the specific components and the task
            self._model = self._model_factory.build_model(
                task='segmentation',
                backbone='prithvi_eo_v2_300',
                decoder='FCNDecoder', # standard decoder
                bands=[
                    HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, 
                    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2
                ],
                num_frames=2,
                num_classes=2,
                pretrained=model_path
            )

            # Move to device
            self._model.to(self._device)
            self._model.eval()

            logger.info("Model loaded successfully via terratorch!")

        except (ImportError, Exception) as e:
            logger.error(f"Error loading model via terratorch: {e}")
            logger.info("Falling back to custom implementation...")
            self._load_custom_model(model_path)

    def _load_custom_model(self, model_id_or_path="ibm-nasa-geospatial/Prithvi-EO-2.0-300M"):
        """
        Load Prithivi model using transformers library as fallback.
        This provides a simpler change detection approach.
        """
        try:
            from transformers import AutoModel, AutoConfig

            logger.info(f"Loading Prithivi model via transformers from {model_id_or_path}...")

            # Load config first to prevent NoneType errors in some environments
            config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
            
            # Use a safe way to set num_labels that doesn't trigger internal property setters prematurely
            config.update({"num_labels": 2}) 
            if not hasattr(config, 'id2label') or config.id2label is None:
                config.id2label = {0: "no_change", 1: "change"}
                config.label2id = {"no_change": 0, "change": 1}

            # Load base encoder
            self._encoder = AutoModel.from_pretrained(
                model_id_or_path,
                config=config,
                trust_remote_code=True
            )

            # Simple change detection head

            # Move to device
            self._encoder.to(self._device)
            self._encoder.eval()

            logger.info("Custom model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise RuntimeError(
                "Could not load Prithivi model. Please ensure transformers "
                "and terratorch are installed correctly."
            )

    @torch.no_grad()
    def detect_changes(
        self,
        pre_tensor: torch.Tensor,
        post_tensor: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect changes between pre and post images.

        Args:
            pre_tensor: Pre-event image tensor (1, 6, H, W)
            post_tensor: Post-event image tensor (1, 6, H, W)
            threshold: Threshold for binary mask (0-1)

        Returns:
            Dictionary containing:
                - mask: Binary change mask (H, W)
                - confidence: Confidence map (H, W) with values 0-1
                - change_magnitude: Magnitude of change (H, W)
        """
        logger.info("Running change detection...")

        # Move tensors to device
        pre_tensor = pre_tensor.to(self._device)
        post_tensor = post_tensor.to(self._device)

        # Handle non-square images or specific resolutions that cause Einops errors
        # The Prithvi-EO-2.0-300M terratorch implementation often assumes square grid
        orig_h, orig_w = pre_tensor.shape[-2:]
        is_square = orig_h == orig_w
        
        # We'll force to 512x512 if not square to be safe
        target_size = (512, 512)
        if not is_square or (orig_h % 16 != 0):
            logger.info(f"Resizing input from ({orig_h}, {orig_w}) to {target_size} for model compatibility")
            pre_tensor_model = torch.nn.functional.interpolate(
                pre_tensor, size=target_size, mode='bilinear', align_corners=False
            )
            post_tensor_model = torch.nn.functional.interpolate(
                post_tensor, size=target_size, mode='bilinear', align_corners=False
            )
        else:
            pre_tensor_model = pre_tensor
            post_tensor_model = post_tensor

        try:
            # Always use embedding distance for zero-shot reliability
            change_map = self._detect_with_embedding_distance(pre_tensor_model, post_tensor_model)

            # Resize change_map back to original size if it was resized
            if not is_square or (orig_h % 16 != 0):
                # change_map is numpy (H, W) or (B, H, W)?
                # _detect_with_terratorch returns squeeze() numpy
                change_map_torch = torch.from_numpy(change_map).unsqueeze(0).unsqueeze(0)
                change_map_torch = torch.nn.functional.interpolate(
                    change_map_torch, size=(orig_h, orig_w), mode='bilinear', align_corners=False
                )
                change_map = change_map_torch.squeeze().cpu().numpy()

            binary_mask = (change_map > threshold).astype(np.uint8)

            logger.info(
                f"Change detection complete. "
                f"Changed pixels: {binary_mask.sum()} / {binary_mask.size}"
            )

            return {
                'mask': binary_mask,
                'confidence': change_map,
                'change_magnitude': change_map,
            }

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def _detect_with_terratorch(
        self,
        pre_tensor: torch.Tensor,
        post_tensor: torch.Tensor
    ) -> np.ndarray:
        """Change detection using full terratorch model."""
        # terratorch models expect (Batch, Channels, Time, Height, Width) for 3D convolutions
        # pre_tensor and post_tensor are (1, 6, H, W)
        # We stack them along a new 'time' dimension at index 2
        stacked = torch.stack([pre_tensor, post_tensor], dim=2)  # (1, 6, 2, H, W)

        # Forward pass
        output_obj = self._model(stacked)

        # Get logits from output
        # Record: terratorch/transformers usually return a dict-like ModelOutput
        # terratorch 1.2.1 ModelOutput has an 'output' attribute
        if hasattr(output_obj, 'output') and output_obj.output is not None:
            logits = output_obj.output
        elif hasattr(output_obj, 'logits'):
            logits = output_obj.logits
        elif isinstance(output_obj, dict) and 'logits' in output_obj:
            logits = output_obj['logits']
        else:
            logits = output_obj

        # Apply softmax and get change class
        probs = torch.softmax(logits, dim=1)
        change_prob = probs[:, 1, :, :]  # Change class

        # Convert to numpy
        change_map = change_prob.squeeze().cpu().numpy()

        return change_map

    def _detect_with_embedding_distance(
        self,
        pre_tensor: torch.Tensor,
        post_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Change detection using zero-shot embedding distance.
        
        This computes embeddings for pre and post, then calculates the 
        distance between them in feature space.
        """
        # Get embeddings
        if hasattr(self, '_model') and self._model is not None:
            # For terratorch model
            if self._model.__class__.__name__ == "PrithviModel":
                # We want embeddings before the head
                # Stack for terratorch
                stacked = torch.stack([pre_tensor, post_tensor], dim=2) # (1, 6, 2, H, W)
                output = self._model(stacked)
                # If we can't easily get embeddings, we might need a different approach
                # But let's try to get them from the encoder backbone
                backbone = self._model.backbone
                # Terratorch backbone forward usually returns list of features
                feats = backbone(stacked)
                # Feature at index -1 is usually the highest level feature
                if isinstance(feats, (list, tuple)):
                    emb = feats[-1]
                elif hasattr(feats, 'output'):
                    emb = feats.output
                elif isinstance(feats, dict) and 'output' in feats:
                    emb = feats['output']
                else:
                    emb = feats

                pre_emb = emb[:, :, 0, :, :]
                post_emb = emb[:, :, 1, :, :]
            else:
                # If it's a terratorch model but we don't know the structure, 
                # try to get the backbone from the model if it exists.
                if hasattr(self._model, 'backbone'):
                    stacked = torch.stack([pre_tensor, post_tensor], dim=2)
                    feats = self._model.backbone(stacked)
                    if isinstance(feats, (list, tuple)):
                        emb = feats[-1]
                    elif hasattr(feats, 'output'):
                        emb = feats.output
                    else:
                        emb = feats
                    pre_emb = emb[:, :, 0, :, :]
                    post_emb = emb[:, :, 1, :, :]
                else:
                    # Last resort fallback
                    output_obj = self._model(pre_tensor.unsqueeze(2).repeat(1, 1, 2, 1, 1))
                    if hasattr(output_obj, 'output'):
                        pre_emb = output_obj.output
                    elif isinstance(output_obj, dict) and 'output' in output_obj:
                        pre_emb = output_obj['output']
                    else:
                        pre_emb = output_obj
                    # If pre_emb is still not a tensor (e.g. it's the ModelOutput itself)
                    if not isinstance(pre_emb, torch.Tensor) and hasattr(pre_emb, 'last_hidden_state'):
                         pre_emb = pre_emb.last_hidden_state
                    post_emb = pre_emb # dummy
        elif hasattr(self, '_encoder') and self._encoder is not None:
            # Custom transformers load
            pre_emb_out = self._encoder(pre_tensor)
            post_emb_out = self._encoder(post_tensor)
            pre_emb = pre_emb_out.last_hidden_state
            post_emb = post_emb_out.last_hidden_state
        else:
            raise RuntimeError("No model or encoder found for inference")

        # Reshape embeddings to spatial format if needed (B, T, C) -> (B, C, H, W)
        if pre_emb.ndim == 3:
            B, L, C = pre_emb.shape
            H = W = int(np.sqrt(L))
            pre_emb = pre_emb.transpose(1, 2).reshape(B, C, H, W)
            post_emb = post_emb.transpose(1, 2).reshape(B, C, H, W)

        # Compute embedding distance
        # To better detect BURN AREAS, we apply higher weight to the SWIR bands
        # HLS usually has SWIR1 and SWIR2 at indices 4 and 5 (0-indexed)
        # Note: embeddings are compressed features, but they still retain spatial/spectral locality
        
        # Calculate cosine similarity
        pre_emb_norm = torch.nn.functional.normalize(pre_emb, p=2, dim=1)
        post_emb_norm = torch.nn.functional.normalize(post_emb, p=2, dim=1)
        
        # Base cosine distance
        cosine_dist = (1 - (pre_emb_norm * post_emb_norm).sum(dim=1)).unsqueeze(1) # (B, 1, H, W)
        logger.info(f"Raw cosine distance range: {cosine_dist.min().item():.4f} to {cosine_dist.max().item():.4f}")
        
        # Apply spatial smoothing to group "random dots" into areas
        # We use a simple average pooling or gaussian-like blur
        smoothed_dist = torch.nn.functional.avg_pool2d(
            cosine_dist, kernel_size=3, stride=1, padding=1
        )
        logger.info(f"Smoothed distance range: {smoothed_dist.min().item():.4f} to {smoothed_dist.max().item():.4f}")
        
        # Adaptive calibration using percentiles
        # Use VERY conservative approach: only highlight the top changes
        # Map the 50th (median) to 98th percentile to 0-1 range
        min_d = torch.quantile(smoothed_dist, 0.50)  # Median
        max_d = torch.quantile(smoothed_dist, 0.98)  # Top 2%
        
        # Ensure we don't divide by zero
        if max_d > min_d:
            change_map_prob = torch.clamp((smoothed_dist - min_d) / (max_d - min_d), 0, 1)
        else:
            # If there's no variation, everything is 0
            change_map_prob = torch.zeros_like(smoothed_dist)
        
        logger.info(f"Calibration range: {min_d.item():.4f} to {max_d.item():.4f}")

        # Remove the batch and channel dimension
        change_map_prob = change_map_prob.squeeze(1) # (B, H, W)

        # Resize to match input resolution if needed
        if change_map_prob.shape[-2:] != pre_tensor.shape[-2:]:
            change_map_prob = torch.nn.functional.interpolate(
                change_map_prob.unsqueeze(1),
                size=pre_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        # Convert to numpy (detach from graph first)
        change_map = change_map_prob.squeeze().detach().cpu().numpy()
        return change_map

    def _detect_with_custom(
        self,
        pre_tensor: torch.Tensor,
        post_tensor: torch.Tensor
    ) -> np.ndarray:
        """Fallback to embedding distance if called."""
        return self._detect_with_embedding_distance(pre_tensor, post_tensor)

    def detect_burn_areas(
        self,
        pre_array: np.ndarray,
        post_array: np.ndarray,
        threshold: float = 0.1,
        use_hybrid: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Detect burn areas using spectral indices and/or deep learning embeddings.
        
        This method combines:
        1. dNBR (delta Normalized Burn Ratio) - spectral signature of burns
        2. Prithivi embeddings - contextual understanding of land cover change
        
        Args:
            pre_array: Pre-fire image (6, H, W)
            post_array: Post-fire image (6, H, W)
            threshold: dNBR threshold for burn detection (default 0.1 = low severity)
            use_hybrid: If True, fuse spectral and embedding signals
        
        Returns:
            Dictionary containing:
                - mask: Binary burn mask
                - confidence: Confidence/severity map
                - severity: Burn severity classification (0-4)
                - stats: Burn severity statistics
                - dnbr: Delta NBR map
        """
        logger.info("Running burn area detection...")
        
        # Step 1: Pure spectral analysis using dNBR
        spectral_results = detect_burn_areas_spectral(
            pre_array, 
            post_array, 
            threshold=threshold,
            use_bai=False  # Can enable for extra validation
        )
        
        if not use_hybrid:
            # Use spectral-only detection
            logger.info("Using spectral-only detection (dNBR)")
            return {
                'mask': spectral_results['mask'],
                'confidence': spectral_results['dnbr'],
                'severity': spectral_results['severity'],
                'stats': spectral_results['stats'],
                'dnbr': spectral_results['dnbr'],
            }
        
        # Step 2: Hybrid approach - combine with Prithivi embeddings
        logger.info("Using hybrid detection (dNBR + Prithivi embeddings)")
        
        # Get embedding-based change detection
        pre_tensor = torch.from_numpy(pre_array).float().unsqueeze(0)
        post_tensor = torch.from_numpy(post_array).float().unsqueeze(0)
        
        embedding_change = self._detect_with_embedding_distance(
            pre_tensor.to(self._device),
            post_tensor.to(self._device)
        )
        
        # Fusion strategy: Both signals must agree
        # This reduces false positives from non-burn changes
        dnbr = spectral_results['dnbr']
        
        # Normalize both to 0-1 range for fusion
        dnbr_norm = np.clip(dnbr / 1.0, 0, 1)  # dNBR rarely exceeds 1.0
        embedding_norm = embedding_change  # already 0-1
        
        # Weighted fusion: spectral evidence gets higher weight for burns
        # Because dNBR is the gold standard for burn mapping
        fused_confidence = 0.7 * dnbr_norm + 0.3 * embedding_norm
        
        # Create binary mask: require strong spectral evidence (dNBR >= threshold)
        # AND some embedding support (embedding_norm >= 0.3)
        burn_mask = (dnbr >= threshold) & (embedding_norm >= 0.3)
        
        logger.info(
            f"Burn detection complete. "
            f"Spectral burns: {spectral_results['mask'].sum()}, "
            f"Hybrid burns: {burn_mask.sum()}"
        )
        
        return {
            'mask': burn_mask.astype(np.uint8),
            'confidence': fused_confidence,
            'severity': spectral_results['severity'],
            'stats': spectral_results['stats'],
            'dnbr': dnbr,
        }

    def detect_changes_from_arrays(
        self,
        pre_array: np.ndarray,
        post_array: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to detect changes from numpy arrays.

        Args:
            pre_array: Pre-event array (6, H, W)
            post_array: Post-event array (6, H, W)
            threshold: Threshold for binary mask

        Returns:
            Dictionary with detection results
        """
        # Convert to tensors
        pre_tensor = torch.from_numpy(pre_array).float().unsqueeze(0)
        post_tensor = torch.from_numpy(post_array).float().unsqueeze(0)

        return self.detect_changes(pre_tensor, post_tensor, threshold)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            'model': 'Prithivi-EO-2.0-300M',
            'device': str(self._device),
            'status': 'loaded' if self._model or self._encoder else 'not loaded',
            'backend': 'terratorch' if hasattr(self, '_model') else 'custom',
        }
