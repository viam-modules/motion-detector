import math
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import PIL
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.utils import pil
from viam.media.video import CameraMimeType, ViamImage
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

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
        validate_cam_name = config.attributes.fields["cam_name"].string_value
        validate_camera_name = config.attributes.fields["camera_name"].string_value

        if validate_cam_name == "" and validate_camera_name == "":
            raise ValueError(
                "Source camera must be provided as 'cam_name' or 'camera_name', "
                "but neither was provided"
            )
        if validate_cam_name != "" and validate_camera_name != "":
            raise ValueError(
                "Source camera must be provided as 'cam_name' or 'camera_name', "
                "but both were provided"
            )
        source_cam = (
            validate_cam_name if validate_cam_name != "" else validate_camera_name
        )

        min_box_size = config.attributes.fields["min_box_size"].number_value
        min_box_percent = config.attributes.fields["min_box_percent"].number_value
        if min_box_size < 0:
            raise ValueError(
                "Minimum bounding box size should be a non-negative integer"
            )
        if min_box_percent < 0.0 or min_box_percent > 1.0:
            raise ValueError(
                "Minimum bounding box percent should be between 0.0 and 1.0"
            )
        if min_box_size != 0 and min_box_percent != 0.0:
            raise ValueError(
                "Cannot specify the minimum box in both pixels and percentages"
            )

        sensitivity = config.attributes.fields["sensitivity"].number_value
        if sensitivity < 0 or sensitivity > 1:
            raise ValueError("Sensitivity should be a number between 0.0 and 1.0")

        max_box_size = config.attributes.fields["max_box_size"].number_value
        max_box_percent = config.attributes.fields["max_box_percent"].number_value
        if max_box_size < 0:
            raise ValueError(
                "Maximum bounding box size should be a non-negative integer"
            )
        if max_box_percent < 0.0 or max_box_percent > 1.0:
            raise ValueError(
                "Maximum bounding box percent should be between 0.0 and 1.0"
            )
        if max_box_size != 0 and max_box_percent != 0.0:
            raise ValueError(
                "Cannot specify the maximum box in both pixels and percentages"
            )

        if config.attributes.fields["crop_region"].struct_value:
            crop_region = dict(
                config.attributes.fields["crop_region"].struct_value.fields
            )
            x1_rel = float(crop_region["x1_rel"].number_value)
            x2_rel = float(crop_region["x2_rel"].number_value)
            y1_rel = float(crop_region["y1_rel"].number_value)
            y2_rel = float(crop_region["y2_rel"].number_value)

            if x1_rel < 0.0 or x1_rel > 1.0:
                raise ValueError("x1_rel should be between 0.0 and 1.0")
            if x2_rel < 0.0 or x2_rel > 1.0:
                raise ValueError("x2_rel should be between 0.0 and 1.0")
            if y1_rel < 0.0 or y1_rel > 1.0:
                raise ValueError("y1_rel should be between 0.0 and 1.0")
            if y2_rel < 0.0 or y2_rel > 1.0:
                raise ValueError("y2_rel should be between 0.0 and 1.0")
            if x1_rel >= x2_rel:
                raise ValueError("x1_rel should be less than x2_rel")
            if x1_rel > x2_rel:
                raise ValueError("x1_rel should be less than x2_rel")
            if y1_rel > y2_rel:
                raise ValueError("y1_rel should be less than y2_rel")
        return [source_cam]

    # Handles attribute reconfiguration
    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        # either "camera_name" or "cam_name" is used to specify the camera
        self.cam_name = config.attributes.fields["cam_name"].string_value
        if self.cam_name == "":
            self.cam_name = config.attributes.fields["camera_name"].string_value

        self.camera = dependencies[Camera.get_resource_name(self.cam_name)]
        self.sensitivity = config.attributes.fields["sensitivity"].number_value
        if self.sensitivity == 0:
            self.sensitivity = 0.9

        # Store all possible box size constraints
        self.min_box_size = config.attributes.fields["min_box_size"].number_value
        self.min_box_percent = config.attributes.fields["min_box_percent"].number_value
        self.max_box_size = config.attributes.fields["max_box_size"].number_value
        self.max_box_percent = config.attributes.fields["max_box_percent"].number_value

        # Crop region is optional, so we need to check if it exists
        if config.attributes.fields["crop_region"].struct_value:
            self.crop_region = dict(
                config.attributes.fields["crop_region"].struct_value.fields
            )
            self.crop_region["x1_rel"] = float(self.crop_region["x1_rel"].number_value)
            self.crop_region["y1_rel"] = float(self.crop_region["y1_rel"].number_value)
            self.crop_region["x2_rel"] = float(self.crop_region["x2_rel"].number_value)
            self.crop_region["y2_rel"] = float(self.crop_region["y2_rel"].number_value)
        else:
            self.crop_region = None

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
        images = await self.camera.get_images()
        if len(images) == 0:
            raise ValueError("No images returned by get_images")
        input1 = images[0]
        if input1.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input1.mime_type
            )
        img1 = pil.viam_to_pil_image(input1)
        img1, _, _ = self.crop_image(img1)
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)

        camera_images = await self.camera.get_images()
        if len(camera_images) == 0:
            raise ValueError("No images were returned by get_images")
        input2 = camera_images[0]
        if input2.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input2.mime_type
            )
        img2 = pil.viam_to_pil_image(input2)
        img2, _, _ = self.crop_image(img2)
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
        if camera_name == "":
            camera_name = self.cam_name
        elif camera_name != self.cam_name:
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'cam_name'",
                self.cam_name,
            )
        return await self.get_classifications(image=None, count=count)

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
        images = await self.camera.get_images()
        if len(images) == 0:
            raise ValueError("No images returned by get_images")
        input1 = images[0]
        if input1.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input1.mime_type
            )
        img1 = pil.viam_to_pil_image(input1)
        img1, width, height = self.crop_image(img1)
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)

        camera_images = await self.camera.get_images()
        if len(camera_images) == 0:
            raise ValueError("No images were returned by get_images")
        input2 = camera_images[0]
        if input2.mime_type not in [CameraMimeType.JPEG, CameraMimeType.PNG]:
            raise ValueError(
                "image mime type must be PNG or JPEG, not ", input2.mime_type
            )
        img2 = pil.viam_to_pil_image(input2)
        img2, width, height = self.crop_image(img2)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
        return self.detections_from_gray_imgs(gray1, gray2, width, height)

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        if camera_name == "":
            camera_name = self.cam_name
        elif camera_name != self.cam_name:
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
        imgs = await self.camera.get_images()
        if len(imgs) == 0 and (return_image or return_classifications or return_detections):
            raise ValueError("No images returned by get_images")
        img = imgs[0]
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

    def detections_from_gray_imgs(self, gray1, gray2, width=None, height=None):
        detections = []
        # Frame difference
        diff = cv2.absdiff(gray2, gray1)

        include_normalized = True
        if diff.shape[0] == 0 or diff.shape[1] == 0:
            include_normalized = False

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
        contours, _ = cv2.findContours(
            img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

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
            if self.max_box_size > 0 and area > self.max_box_size:
                continue
            area_percent = area / np.prod(diff.shape)
            if area_percent < self.min_box_percent:
                continue
            if self.max_box_percent > 0 and area_percent > self.max_box_percent:
                continue

            if self.crop_region:
                # Adjust coordinates based on crop region
                x_offset = int(self.crop_region.get("x1_rel") * width)
                y_offset = int(self.crop_region.get("y1_rel") * height)

                # Convert back to original image coordinates
                xmin = min(width - 1, xmin + x_offset)
                ymin = min(height - 1, ymin + y_offset)
                xmax = min(width - 1, xmax + x_offset)
                ymax = min(height - 1, ymax + y_offset)

            detection = {
                "confidence": 0.5,
                "class_name": "motion",
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
            }

            if include_normalized:
                detection.update(
                    {
                        "x_min_normalized": xmin / diff.shape[1],
                        "y_min_normalized": ymin / diff.shape[0],
                        "x_max_normalized": xmax / diff.shape[1],
                        "y_max_normalized": ymax / diff.shape[0],
                    }
                )
            detections.append(detection)

        return detections

    def crop_image(self, image: PIL.Image.Image):
        if not self.crop_region:
            return image, None, None
        width, height = image.size
        x1 = int(self.crop_region["x1_rel"] * width)
        y1 = int(self.crop_region["y1_rel"] * height)
        x2 = int(self.crop_region["x2_rel"] * width)
        y2 = int(self.crop_region["y2_rel"] * height)
        return image.crop((x1, y1, x2, y2)), width, height

    def retrieve_original_coordinates(self, x_normalized, y_normalized, width, height):
        pass
