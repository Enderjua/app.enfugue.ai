from __future__ import annotations

from typing import Any, Callable, Iterator, TYPE_CHECKING

from enfugue.util import logger
from contextlib import contextmanager

if TYPE_CHECKING:
    from PIL.Image import Image

from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

__all__ = ["PoseDetector"]

class InstantiationException(Exception):
    def __init__(self, exception: Exception):
        super(InstantiationException, self).__init__("Could not instantiated DWPose: {0}({1})".format(
            type(exception).__name__,
            exception
        ))

class PoseImageProcessor(SupportModelImageProcessor):
    """
    Uses a pose detector to detect human poses, hands, and faces
    """
    def __init__(self, detector: Callable[[Image], Image], **kwargs: Any) -> None:
        super(PoseImageProcessor, self).__init__(**kwargs)
        self.detector = detector

    def detail_mask(
        self,
        image: Image,
        include_hands: bool=True,
        include_face: bool=True,
        isolated: bool=False,
    ) -> Union[Image, List[Image]]:
        """
        Calls the detector and draws the detail mask (hands and face)
        """
        if include_hands and include_face:
            draw_type = "mask"
        elif include_hands:
            draw_type = "handmask"
        elif include_face:
            draw_type = "facemask"
        else:
            # Return black
            from PIL import Image
            black_image = Image.new(image.mode, image.size)
            if isolated:
                return [black_image]
            return black_image
        detected = self.detector( # type: ignore[call-arg]
            image,
            include_body=False,
            include_hand=include_hands,
            include_face=include_face,
            draw_type=draw_type,
            isolated=isolated
        )
        if isolated:
            for i in range(len(detected)):
                detected[i] = detected[i].resize(image.size)
        else:
            detected = detected.resize(image.size)
        return detected

    def __call__(self, image: Image) -> Image:
        """
        Calls the detector
        """
        return self.detector( # type: ignore[call-arg]
            image,
            include_body=True,
            include_hand=True,
            include_face=True
        ).resize(image.size)

class PoseDetector(SupportModel):
    """
    Uses OpenPose to predict human poses.
    """
    BODY_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
    HAND_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
    FACE_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

    DW_DETECT_MODEL_PATH = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    DW_POSE_MODEL_PATH = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"

    @contextmanager
    def dwpose(self) -> Iterator[SupportModelImageProcessor]:
        """
        Gets and runs the DWPose detector.
        """
        try:
            from enfugue.diffusion.support.pose.dwpose import DWposeDetector  # type: ignore
            has_yielded = False
            with self.context():
                detect_model_path = self.get_model_file(self.DW_DETECT_MODEL_PATH)
                pose_model_path = self.get_model_file(self.DW_POSE_MODEL_PATH)
                detector = DWposeDetector(
                    det_ckpt=detect_model_path,
                    pose_ckpt=pose_model_path
                )
                detector.to(self.device)
                processor = PoseImageProcessor(detector)
                has_yielded = True
                yield processor
                del processor
                del detector
        except Exception as ex:
            if not has_yielded:
                raise InstantiationException(ex)
            else:
                raise

    @contextmanager
    def openpose(self) -> Iterator[SupportModelImageProcessor]:
        """
        Gets and runs the OpenPose detector.
        """
        from enfugue.diffusion.support.pose.openpose.helper import OpenposeDetector  # type: ignore

        with self.context():
            body_model_path = self.get_model_file(self.BODY_MODEL_PATH)
            hand_model_path = self.get_model_file(self.HAND_MODEL_PATH)
            face_model_path = self.get_model_file(self.FACE_MODEL_PATH)
            detector = OpenposeDetector.from_pretrained(
                body_model_path,
                hand_model_path,
                face_model_path
            )
            detector.to(self.device)
            processor = PoseImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def best(self) -> Iterator[SupportModelImageProcessor]:
        """
        Uses DWPose if available, otherwise uses OpenPose.
        """
        try:
            with self.dwpose() as processor:
                logger.debug("Using DWPose for pose detection.")
                yield processor
        except InstantiationException:
            with self.openpose() as processor:
                logger.debug("Using OpenPose for pose detection.")
                yield processor
