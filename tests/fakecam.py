from typing import Any, Coroutine, Final, List, Optional, Tuple, Dict
from viam.components.camera import Camera
from viam.gen.component.camera.v1.camera_pb2 import GetPropertiesResponse
from viam.media.video import NamedImage, ViamImage, CameraMimeType
from viam.media.utils import pil
from viam.proto.common import ResponseMetadata
from PIL import Image



class FakeCamera(Camera):

    def __init__(self, name: str):
        super().__init__(name=name)
        self.count = -1
        img1 = Image.open("tests/img1.jpg")
        img2 = Image.open("tests/img2.jpg")
        self.images = [img1, img2]

    async def get_image(self, mime_type: str = "") -> Coroutine[Any, Any, ViamImage]:
        self.count +=1
        return pil.pil_to_viam_image(self.images[self.count%2], CameraMimeType.JPEG)

    async def get_images(self) -> Coroutine[Any, Any, Tuple[List[NamedImage] | ResponseMetadata]]:
        #self.count +=1
        #return [pil.pil_to_viam_image(self.images[self.count%2], CameraMimeType.JPEG)]
        raise NotImplementedError

    async def get_properties(self) -> Coroutine[Any, Any, GetPropertiesResponse]:
        raise NotImplementedError

    async def get_point_cloud(self) -> Coroutine[Any, Any, Tuple[bytes | str]]:
        raise NotImplementedError
