import math
from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional
from typing_extensions import Self
import cv2
import numpy as np


from viam.components.camera import Camera
from viam.media.video import ViamImage, CameraMimeType
from viam.media.utils import pil
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision, CaptureAllResult
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes
from viam.logging import getLogger




LOGGER = getLogger("MotionDetectorLogger")


class MotionDetector(Vision, Reconfigurable):
    """
    MotionDetector implements a vision service that only supports detections
    and classifications.

    It inherits from the built-in resource subtype Vision and conforms to the
    ``Reconfigurable`` protocol, which signifies that this component can be
    reconfigured. Additionally, it specifies a constructor function
    ``MotionDetector.new_service`` which confirms to the
    ``resource.types.ResourceCreator`` type required for all models.
    """

    # Here is where we define our new model's colon-delimited-triplet
    # (viam:vision:motion-detector) viam = namespace, vision = family, motion-detector = model name.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "motion-detector")

    def __init__(self, name: str):
        super().__init__(name=name)

    # Constructor
    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        source_cam = config.attributes.fields["cam_name"].string_value
        if source_cam == "":
            raise ValueError("Source camera must be provided as 'cam_name'")

        # You must specify either a min_box_size or a min_box_percent.
        min_box_size    = config.attributes.fields.get("min_box_size")
        min_box_percent = config.attributes.fields.get("min_box_percent")
        if min_box_size is None:
            if min_box_percent is None:
                raise ValueError("Must specify the minimum box in either pixels or percentages")
            min_box_val = min_box_percent.number_value
            if min_box_val < 0.0 or min_box_val > 1.0:
                raise ValueError("Minimum bounding box percent should be between 0.0 and 1.0")
        else:  # min_box_size is not None
            if min_box_percent is not None:
                raise ValueError("Cannot specify the minimum box in both pixels and percentages")
            min_box_val = min_box_size.number_value
            if min_box_val < 0:
                raise ValueError("Minimum bounding box size should be a non-negative integer")

        sensitivity = config.attributes.fields.get("sensitivity")
        if sensitivity is not None:
            sensitivity = sensitivity.number_value
            if sensitivity < 0 or sensitivity > 1:
                raise ValueError("Sensitivity should be a number between 0 and 1")

        max_box_size    = config.attributes.fields.get("max_box_size")
        max_box_percent = config.attributes.fields.get("max_box_percent")
        if max_box_size is None:
            if max_box_percent is None:
                raise ValueError("Must specify the maximum box in either pixels or percentages")
            max_box_val = max_box_percent.number_value
            if max_box_val < 0.0 or max_box_val > 1.0:
                raise ValueError("Maximum bounding box percent should be between 0.0 and 1.0")
        else:  # max_box_size is not None
            if max_box_percent is not None:
                raise ValueError("Cannot specify the maximum box in both pixels and percentages")
            max_box_val = max_box_size.number_value
            if max_box_val < 0:
                raise ValueError("Maximum bounding box size should be a non-negative integer")

        return [source_cam]

    # Handles attribute reconfiguration
    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.cam_name = config.attributes.fields["cam_name"].string_value
        self.camera = dependencies[Camera.get_resource_name(self.cam_name)]
        self.sensitivity = config.attributes.fields["sensitivity"].number_value
        if self.sensitivity == 0:
            self.sensitivity = 0.9
        self.min_box_size = config.attributes.fields["min_box_size"].number_value
        self.max_box_size = config.attributes.fields.get("max_box_size")
        if self.max_box_size is not None:
            self.max_box_size = self.max_box_size.number_value

    # This will be the main method implemented in this module.
    # Given a camera. Perform frame differencing and return how much of the image is moving
    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        # Grab and grayscale 2 images
        input1 = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        if input1.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input1.mime_type
            )
        img1 = pil.viam_to_pil_image(input1)
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)

        input2 = await self.camera.get_image()
        if input2.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input2.mime_type
            )
        img2 = pil.viam_to_pil_image(input2)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)

        return self.classification_from_gray_imgs(gray1=gray1, gray2=gray2)

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        if camera_name != self.cam_name:
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'cam_name'",
                self.cam_name,
            )
        image = await self.camera.get_image()
        return await self.get_classifications(image=image, count=count)

    # Not implemented for now. Eventually want this to return the location of the movement
    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        # Grab and grayscale 2 images
        input1 = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        if input1.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input1.mime_type
            )
        img1 = pil.viam_to_pil_image(input1)
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)

        input2 = await self.camera.get_image()
        if input2.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input2.mime_type
            )
        img2 = pil.viam_to_pil_image(input2)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)

        return self.detections_from_gray_imgs(gray1, gray2)

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        if camera_name != self.cam_name:
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'cam_name':",
                self.cam_name,
            )
        return await self.get_detections(image=None)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    # The linter doesn't like the vision service API, which we can't change.
    async def capture_all_from_camera(  # pylint: disable=too-many-positional-arguments
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        if camera_name not in (self.cam_name, ""):
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'cam_name':",
                self.cam_name,
            )
        img = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        if return_image:
            result.image = img
        if return_classifications:
            classifs = await self.get_classifications(img, 1)
            result.classifications = classifs
        if return_detections:
            dets = await self.get_detections(img)
            result.detections = dets
        # No object point clouds
        return result

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def classification_from_gray_imgs(self, gray1, gray2):
        # Frame difference
        diff = cv2.absdiff(gray2, gray1)

        # Simple noise filtering via threshold (~10% of 255)
        k = math.floor((1 - self.sensitivity) * 255)
        diff[diff < k] = 0
        diff[diff > k] = 1

        # Confidence = percent of activated pixels (after thresholding)
        conf = np.sum(diff) / (gray1.shape[0] * gray1.shape[1])

        classifications = [{"class_name": "motion", "confidence": conf}]
        return classifications

    def detections_from_gray_imgs(self, gray1, gray2):
        detections = []
        # Frame difference
        diff = cv2.absdiff(gray2, gray1)

        # Simple noise filtering via threshold (~10% of 255)
        k = math.floor((1 - self.sensitivity) * 255)
        diff[diff < k] = 0
        diff[diff > k] = 255

        # Morphological operations to remove noise and blob
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((15, 15), np.uint8)
        img = cv2.erode(diff, kernel)
        img2 = cv2.dilate(img, kernel)
        img3 = cv2.dilate(img2, kernel2)
        img_out = cv2.erode(img3, kernel2)

        # List points around the remaining blobs
        contours, _ = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Make boxes from the contours
        for c in contours:
            # Each contour should be a box.
            xs = [pt[0][0] for pt in c]
            ys = [pt[0][1] for pt in c]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

            # Ignore this detection if it's the wrong size
            area = (ymax - ymin) * (xmax - xmin)
            if area < self.min_box_size:
                continue
            if self.max_box_size is not None and area > self.max_box_size:
                continue

            detections.append(
                {
                    "confidence": 0.5,
                    "class_name": "motion",
                    "x_min": int(xmin),
                    "y_min": int(ymin),
                    "x_max": int(xmax),
                    "y_max": int(ymax),
                }
            )

        return detections
