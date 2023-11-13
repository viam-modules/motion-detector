from io import BytesIO
from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union, cast
from typing_extensions import Self
from PIL import Image

from viam.components.camera import Camera
from viam.media.video import RawImage, CameraMimeType
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes
from viam.logging import getLogger

import cv2
import numpy as np

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
    def new_service(cls,
                 config: ServiceConfig,
                 dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        source_cam = config.attributes.fields["cam_name"].string_value
        return [source_cam]
    

    # Handles attribute reconfiguration
    def reconfigure(self,
                    config: ServiceConfig,
                    dependencies: Mapping[ResourceName, ResourceBase]):

        self.cam_name = config.attributes.fields["cam_name"].string_value
        self.camera = dependencies[Camera.get_resource_name(self.cam_name)]

        
    """
    Implement the methods the Viam RDK defines for the vision service API
    (rdk:service:vision)
    """

    # This will be the main method implemented in this module. 
    # Given a camera. Perform frame differencing and return how much of the image is moving
    async def get_classifications(self,
                                 image: Union[Image.Image, RawImage],
                                 count: int,
                                 *, 
                                 extra: Optional[Dict[str, Any]] = None,
                                 timeout: Optional[float] = None,
                                 **kwargs) -> List[Classification]:
        # Grab and grayscale 2 images
        img1 = await self.camera.get_image()
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
        img2 = await self.camera.get_image()
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
        
        # Frame difference
        diff = cv2.absdiff(gray2,gray1)
        
        # Simple noise filtering via threshold (~10% of 255)
        k = 25
        diff[diff<k] = 0
        diff[diff>k] = 1

        # Confidence = percent of activated pixels (after thresholding)
        conf = np.sum(diff) / (img1.size[0] * img1.size[1])

        classifications = [{"class_name": "motion", "confidence": conf}]
        return classifications

    async def get_classifications_from_camera(self, 
                                              camera_name: str, 
                                              count: int, 
                                              *,
                                              extra: Optional[Dict[str, Any]] = None,
                                              timeout: Optional[float] = None,
                                              **kwargs) -> List[Classification]:
        if camera_name != self.cam_name:
            raise Exception(
                "Camera name passed to method:",camera_name, "is not the configured source_cam:", self.cam_name)
        return await self.get_classifications(image=None, count=count)

    # Not implemented for now. Eventually want this to return the location of the movement 
    async def get_detections(self,
                            image: Union[Image.Image, RawImage],
                            *,
                            extra: Optional[Dict[str, Any]] = None,
                            timeout: Optional[float] = None,
                            **kwargs) -> List[Detection]:
       # For now.
       raise NotImplementedError

    async def get_detections_from_camera(self,
                                        camera_name: str,
                                        *,
                                        extra: Optional[Dict[str, Any]] = None,
                                        timeout: Optional[float] = None,
                                        **kwargs) -> List[Detection]:

        # For now.
        raise NotImplementedError
    
    async def get_object_point_clouds(self,
                                      camera_name: str,
                                      *,
                                      extra: Optional[Dict[str, Any]] = None,
                                      timeout: Optional[float] = None,
                                      **kwargs) -> List[PointCloudObject]:
        raise NotImplementedError
    
    async def do_command(self,
                        command: Mapping[str, ValueTypes],
                        *,
                        timeout: Optional[float] = None,
                        **kwargs):
        raise NotImplementedError
    

