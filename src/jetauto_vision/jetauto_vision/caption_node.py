"""
Scene captioning node using Florence-2.

Subscribes to a camera image topic, periodically runs Florence-2 inference,
and publishes natural-language scene descriptions to /scene_caption.

Uses a dedicated inference thread to avoid blocking ROS callbacks.
Only publishes when the scene description changes significantly.
"""

import time
import threading

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class CaptionNode(LifecycleNode):
    """ROS2 lifecycle node for Florence-2 scene captioning."""

    def __init__(self):
        super().__init__('caption_node')

        # -- Declare parameters --
        self.declare_parameter('model_name', 'microsoft/Florence-2-base')
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('caption_interval', 15.0)
        self.declare_parameter('task_prompt', '<MORE_DETAILED_CAPTION>')
        self.declare_parameter('similarity_threshold', 0.6)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('max_new_tokens', 256)

        # -- State --
        self.model = None
        self.processor = None
        self.torch_dtype = None
        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._prev_caption = ''
        self._caption_thread = None
        self._shutdown_event = threading.Event()

        self.get_logger().info('CaptionNode created (inactive — waiting for configure)')

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks
    # ------------------------------------------------------------------ #

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Read parameters and load the Florence-2 model."""
        self._read_parameters()

        if not self._load_model():
            return TransitionCallbackReturn.FAILURE

        self.pub_caption = self.create_publisher(String, '/scene_caption', 1)

        self.get_logger().info(
            f'Configured — model={self.model_name}, '
            f'interval={self.caption_interval}s, task={self.task_prompt}'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Create image subscription and start the captioning loop."""
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._image_callback, 1
        )

        self._shutdown_event.clear()
        self._caption_thread = threading.Thread(
            target=self._caption_loop, daemon=True, name='caption_loop'
        )
        self._caption_thread.start()

        self.get_logger().info('Activated — captioning started')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop captioning."""
        self._shutdown_event.set()
        if self._caption_thread and self._caption_thread.is_alive():
            self._caption_thread.join(timeout=5.0)
        if hasattr(self, 'sub_image'):
            self.destroy_subscription(self.sub_image)
        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release model."""
        self._release_model()
        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        self._shutdown_event.set()
        self._release_model()
        self.get_logger().info('Shut down')
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def _read_parameters(self):
        self.model_name = self.get_parameter('model_name').value
        self.image_topic = self.get_parameter('image_topic').value
        self.image_encoding = self.get_parameter('image_encoding').value
        self.caption_interval = self.get_parameter('caption_interval').value
        self.task_prompt = self.get_parameter('task_prompt').value
        self.similarity_threshold = self.get_parameter('similarity_threshold').value
        self.device = self.get_parameter('device').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> bool:
        """Load the Florence-2 model and processor."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.torch_dtype = (
                torch.float16 if 'cuda' in self.device else torch.float32
            )

            self.get_logger().info(
                f'Loading Florence-2 from {self.model_name} '
                f'(device={self.device}, dtype={self.torch_dtype})...'
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)

            # Warm up with a dummy image
            dummy = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            self._run_inference(dummy)

            self.get_logger().info(
                f'Florence-2 loaded on {self.device} '
                f'({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)'
            )
            return True

        except ImportError as e:
            self.get_logger().error(
                f'Missing dependency: {e}. '
                'Run: pip3 install transformers einops timm'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load Florence-2: {e}')
        return False

    def _release_model(self):
        """Release GPU memory."""
        self._shutdown_event.set()
        if self._caption_thread and self._caption_thread.is_alive():
            self._caption_thread.join(timeout=3.0)
        self.model = None
        self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image):
        """Store the latest camera frame (non-blocking)."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.image_encoding
            )
            with self._frame_lock:
                self._latest_frame = cv_image
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    # ------------------------------------------------------------------ #
    # Captioning loop
    # ------------------------------------------------------------------ #

    def _caption_loop(self):
        """Background thread: periodically caption the latest frame."""
        # Wait for first frame
        while not self._shutdown_event.is_set():
            with self._frame_lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.5)

        if self._shutdown_event.is_set():
            return

        self.get_logger().info('First frame received — beginning caption loop')

        while not self._shutdown_event.is_set():
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is not None and self.model is not None:
                pil_image = PILImage.fromarray(frame)
                caption = self._run_inference(pil_image)

                if caption and not self._is_similar(caption, self._prev_caption):
                    msg = String()
                    msg.data = caption
                    self.pub_caption.publish(msg)
                    self.get_logger().info(f'Caption: "{caption}"')
                    self._prev_caption = caption
                elif caption:
                    self.get_logger().debug('Scene unchanged — skipping')

            # Sleep in small increments so we can respond to shutdown
            for _ in range(int(self.caption_interval * 2)):
                if self._shutdown_event.is_set():
                    return
                time.sleep(0.5)

    def _run_inference(self, pil_image: PILImage.Image) -> str:
        """Run Florence-2 inference and return the caption string."""
        try:
            import torch

            inputs = self.processor(
                text=self.task_prompt,
                images=pil_image,
                return_tensors='pt',
            ).to(self.device, self.torch_dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    pixel_values=inputs['pixel_values'],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                )

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                text,
                task=self.task_prompt,
                image_size=(pil_image.width, pil_image.height),
            )

            caption = parsed.get(self.task_prompt, '')
            if isinstance(caption, dict):
                caption = str(caption)

            return caption.strip()

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return ''

    # ------------------------------------------------------------------ #
    # Scene change detection
    # ------------------------------------------------------------------ #

    def _is_similar(self, new_caption: str, old_caption: str) -> bool:
        """Check if two captions describe roughly the same scene.

        Uses word-set overlap: if more than `similarity_threshold` of words
        are shared, the scene hasn't changed enough to announce.
        """
        if not old_caption:
            return False

        # Strip common filler words for better comparison
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'in', 'on', 'of', 'and',
            'with', 'to', 'it', 'that', 'this', 'at', 'by', 'from',
            'image', 'shows', 'there', 'which', 'has', 'have', 'can',
            'be', 'been', 'was', 'were',
        }

        def meaningful_words(text):
            return {
                w for w in text.lower().split()
                if w not in stop_words and len(w) > 1
            }

        new_words = meaningful_words(new_caption)
        old_words = meaningful_words(old_caption)

        if not new_words or not old_words:
            return False

        overlap = len(new_words & old_words) / max(len(new_words), len(old_words))
        return overlap > self.similarity_threshold


def main(args=None):
    rclpy.init(args=args)
    node = CaptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
